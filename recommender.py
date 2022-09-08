import osu_utils
from data.model import *
from pass_feature import PassFeature
import xgboost
from scipy import stats, integrate


class PPRecommender:

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection):
        self.pass_feature = PassFeature(config, connection)
        self.config = config
        self.connection = connection
        self.pass_rate_model = xgboost.Booster(
            model_file=os.path.join("result", "pass_xgboost", "model"))

    def map_beatmap_name(self, name, version):
        name = name + " - " + version
        max_length = 50
        name = name[:max_length - 10] + "..." + name[-10:] if len(name) > max_length else name
        return name

    def recall(self, uid):
        # stage 1: recall the maps with the highest possible pp
        base_projection = [
            BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.BEATMAP_ID,
            ModEmbedding.SPEED, UserEmbedding.VARIANT,
            # for pp
            Beatmap.OD,
            Beatmap.COUNT_SLIDERS,
            Beatmap.COUNT_CIRCLES,
            Beatmap.HT_STAR, Beatmap.STAR, Beatmap.DT_STAR,
            # for name
            Beatmap.VERSION, Beatmap.NAME
        ]
        data_list = []
        st = time.time()
        cursor = list(osu_utils.predict_score(self.connection, {
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: ('=', uid),
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT: ('=', '4k'),
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: ('=', self.config.game_mode),
            Beatmap.CS: ('=', 4),
            ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC: ('=', '0')
        }, self.config.embedding_size, projection=base_projection))
        # cursor += list(osu_utils.predict_score(self.connection, {
        #     UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: ('=', uid),
        #     UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT: ('=', '7k'),
        #     UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: ('=', self.config.game_mode),
        #     Beatmap.CS: ('=', 7),
        #     ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC: ('=', '0')
        # }, self.config.embedding_size, projection=base_projection))
        for x in cursor:
            score, bid, speed, variant, od, count1, count2, \
            star_ht, star_nm, star_dt, version, name = x[:len(base_projection) + 1]
            if speed == 1:
                star = star_dt
                mod = 'DT'
            elif speed == 0:
                star = star_nm
                mod = 'NM'
            else:
                star = star_ht
                mod = 'HT'
            data_list.append([bid, mod, star, score, 0, self.map_beatmap_name(name, version),
                              od, count1 + count2, speed, variant])
        print(f"DB time: {time.time() - st}")
        data = pd.DataFrame(data_list,
                            columns=['id', "mod", "star", "pred_score", "pred_pp", "name",
                                     'od', 'count', 'speed', 'variant'])
        data['pred_score'] = np.round(
            osu_utils.map_osu_score(data['pred_score'].to_numpy(), real_to_train=False)
        ).astype(int)
        data['pred_pp'] = osu_utils.mania_pp(data['pred_score'].to_numpy(), data['od'].to_numpy(),
                                             data['star'].to_numpy(),
                                             data['count'].to_numpy())
        measure_time(data.sort_values)(by="pred_pp", ascending=False, inplace=True)
        data: pd.DataFrame = data.iloc[:1000, :]
        data.set_index(['id', 'mod'], inplace=True)
        return data

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance):
        user_true_pp = user_bp.get_pp()

        def score_with_prob(x, pred_score, pred_score_std):
            # print(x)
            prob = np.exp(-(x - pred_score) ** 2 / (2 * pred_score_std ** 2)) / (
                    pred_score_std * np.sqrt(2 * np.pi))
            x_real = osu_utils.map_osu_score(x, real_to_train=False)
            return prob * x_real

        # stage 2: rank precisely with pass rate, break prob, pp recom, etc.
        pass_features_list, probable_scores, probable_pps, break_probs, \
        pp_gains, true_scores, true_pps = [], [], [], [], [], [], []
        for index, series in data.iterrows():
            bid, mod = index
            # pass rate
            pass_features, _ = self.pass_feature.get_pass_features(uid, series['variant'], bid, mod,
                                                                   is_predicting=True)
            pass_features_list.append(pass_features)
            # break prob
            true_speed, true_score, true_pp = user_bp.get_score_pp(bid)
            if true_score is None:
                break_prob = 1.0
                probable_score = series["pred_score"]
            else:
                true_score = int(round(true_score))
                std_result = osu_utils.predict_score_std(self.connection, uid,
                                                         series['variant'], self.config, bid, mod)
                if std_result is None or true_speed != series['speed']:
                    probable_score = series["pred_score"]
                    break_prob = 1.0
                else:
                    score_train, score_std_train = std_result
                    true_score_train = osu_utils.map_osu_score(true_score, real_to_train=True)
                    break_prob = 1 - stats.norm.cdf(true_score_train, loc=score_train,
                                                    scale=score_std_train)
                    probable_score = \
                    integrate.quad(score_with_prob, a=true_score_train, b=np.inf,
                                   args=(score_train, score_std_train),
                                   epsabs=1.0)[0] / break_prob
                    probable_score = round(probable_score)
            true_scores.append(true_score)
            true_pps.append(true_pp)
            probable_scores.append(probable_score)
            break_probs.append(round(break_prob, 6))
            # pp gain
            if probable_score is not None:
                pp_gain_beatmap = osu_utils.mania_pp(probable_score, series['od'], series['star'],
                                                     series['count'])
                pp_gain_beatmap = round(pp_gain_beatmap, 3)
                probable_pps.append(pp_gain_beatmap)
                user_bp2 = user_bp.copy()
                user_bp2.update(bid, 0, probable_score, pp_gain_beatmap, 0)
                pp_gain = user_bp2.get_pp() - user_true_pp
                pp_gain = max(0, pp_gain)
                pp_gains.append(pp_gain)
            else:
                probable_pps.append(None)
                pp_gains.append(None)

        data['pred_score (breaking)'] = probable_scores
        data['pred_pp (breaking)'] = probable_pps
        data['break_prob'] = break_probs
        data["pass_prob"] = self.pass_rate_model.predict(xgboost.DMatrix(pass_features_list))
        data["pp_gain (breaking)"] = pp_gains
        data['true_score'] = true_scores
        data['true_pp'] = true_pps
        data['pp_gain_expect'] = data['pp_gain (breaking)'] * data['break_prob'] * data['pass_prob']

        data = pd.DataFrame(data,
                                  columns=['name', 'star',
                                           'true_score', 'true_pp',
                                           'pred_score', 'break_prob',
                                           # 'pred_score (breaking)', 'pred_pp (breaking)',
                                           'pp_gain (breaking)', 'pass_prob',
                                           'pp_gain_expect'])
        data.sort_values(by="pp_gain_expect", ascending=False, inplace=True)
        return data

    def predict(self, uid):
        user_name = repository.select_first(self.connection, User.TABLE_NAME, project=[User.NAME],
                                            where={User.ID: uid})

        user_bp = osu_utils.get_user_bp(self.connection, uid, self.config)
        user_true_pp = user_bp.get_pp()
        print("current pp", user_true_pp)

        data = self.recall(uid)

def save_excel(df, filename):
    dfs = {'Sort by PP gain expect': df,
           'Sort by break prob': df.sort_values(by=['break_prob', 'pp_gain_expect'], ascending=False),
           'Sort by current BP': df.sort_values(by=['true_pp'], ascending=False),
           'Sort by pass prob': df.sort_values(by=['pass_prob', 'pp_gain_expect'])}
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheetname, df in dfs.items():  # loop through `dict` of dataframes
        df.to_excel(writer, sheet_name=sheetname)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
            worksheet.set_column(idx + 2, idx + 2, max_len)  # set column width
    writer.save()

if __name__ == "__main__":

    # uid = "10702235"  # ca
    # uid = "10500832"  # xz
    # uid = "7304075"  # ku
    # uid = "6701729" # serika
    # uid = "10817494"  # nickname
    # uid = "30281907" # potassium
    uid = "8586018" # luoxuan

    connection = repository.get_connection()
    recommender = PPRecommender(NetworkConfig(), connection)
    user_bp = osu_utils.get_user_bp(connection, uid, recommender.config)

    st = time.time()
    data = recommender.recall(uid)
    print(f"Recall time: {time.time() - st}")
    # print(data.to_string())

    st = time.time()
    data = recommender.rank(uid, data, user_bp)
    print(f"Rank time: {time.time() - st}")
    print(data.to_string())

    user_name = repository.select_first(connection, User.TABLE_NAME, project=[User.NAME],
                                        where={User.ID: uid})

    os.makedirs("report", exist_ok=True)
    save_excel(data, os.path.join("report", f"{user_name[0]} - PP report.xlsx"))
