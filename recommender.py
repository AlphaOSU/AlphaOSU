import osu_utils
from data.model import *
from pass_feature_feed import PassFeatureFeed
import xgboost
from scipy import stats, integrate


class PPRecommender:

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection):
        self.pass_feature_feed = PassFeatureFeed(config, connection)
        self.config = config
        self.connection = connection
        self.speed_to_pass_model = {}
        for speed in [-1, 0, 1]:
            path = get_pass_model_path(speed)
            if os.path.exists(path):
                self.speed_to_pass_model[speed] = xgboost.Booster(model_file=path)
            else:
                print("warning: pass model not found: speed =", speed)
                self.speed_to_pass_model[speed] = None

    def similarity(self, uid, variant):
        
        project = self.config.get_embedding_names(UserEmbedding.EMBEDDING)

        embedding = repository.select_first(self.connection, UserEmbedding.TABLE_NAME,
                                            project=project,
                                            where={
                                                UserEmbedding.GAME_MODE: self.config.game_mode,
                                                UserEmbedding.VARIANT: variant,
                                                UserEmbedding.USER_ID: uid
                                            })
        if embedding is None:
            return None
        embedding = np.asarray(embedding)

        cursor = repository.execute_sql(
            self.connection,
            f"SELECT {', '.join(project)}, UserEmbedding.{UserEmbedding.USER_ID}, {User.NAME}, {User.PP} "
            f"FROM UserEmbedding, User "
            f"WHERE UserEmbedding.{UserEmbedding.USER_ID} == User.{User.ID} "
            f"AND UserEmbedding.{UserEmbedding.GAME_MODE} == User.{User.GAME_MODE} "
            f"AND UserEmbedding.{UserEmbedding.VARIANT} == User.{User.VARIANT} "
            f"AND UserEmbedding.{UserEmbedding.GAME_MODE} == '{self.config.game_mode}' "
            f"AND UserEmbedding.{UserEmbedding.VARIANT} == '{variant}' "
        )
        data = []
        data_uid, data_name, data_pp = [], [], []
        data_embedding = []
        st = time.time()
        for x in cursor:
            # cur_embedding = np.asarray(x[:self.config.embedding_size])
            # cur_uid, cur_name, cur_pp = x[-3:]
            # distance = np.linalg.norm(embedding - cur_embedding)
            # data.append([cur_uid, cur_name, cur_pp, distance])

            data_embedding.append(x[:self.config.embedding_size])
            cur_uid, cur_name, cur_pp = x[-3:]
            data_uid.append(cur_uid)
            data_name.append(cur_name)
            data_pp.append(cur_pp)
        data_embedding = np.asarray(data_embedding)
        distance = np.linalg.norm(embedding.reshape(1, -1) - data_embedding, axis=-1, keepdims=False)
        print(f"similarity: {time.time() - st} s")

        # data_df = pd.DataFrame(data, columns=["id", "name", "pp", "distance"])
        data_df = pd.DataFrame({
            "id": data_uid,
            "name": data_name,
            "pp": data_pp,
            "distance": distance
        })
        data_df.sort_values(by=["distance", "id"], ascending=True, inplace=True)

        return data_df

    def map_beatmap_name(self, name, version):
        name = name + " - " + version
        max_length = 60
        end_length = 20
        name = name[:max_length - end_length] + "..." + name[-end_length:] if len(name) > max_length else name
        return name

    def recall(self, uid, key_count=[4], beatmap_ids=[]):
        # stage 1: recall the maps with the highest possible pp
        base_projection = [
            BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.BEATMAP_ID,
            ModEmbedding.SPEED, UserEmbedding.VARIANT,
            # for pp
            Beatmap.OD,
            Beatmap.COUNT_SLIDERS,
            Beatmap.COUNT_CIRCLES,
            Beatmap.HT_STAR, Beatmap.STAR, Beatmap.DT_STAR,
            Beatmap.CS, 
            # for name
            Beatmap.VERSION, Beatmap.NAME
        ]
        data_list = []
        cursor = []
        query_params = {
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: ('=', uid),
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: ('=', self.config.game_mode),
            ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC: ('=', '0')
        }
        if len(beatmap_ids) != 0:
            beatmap_ids = set(beatmap_ids)
        if 4 in key_count:
            query_params[UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT] = ('=', '4k')
            query_params[Beatmap.CS] = ('=', 4)
            cursor += list(osu_utils.predict_score(self.connection, query_params, self.config.embedding_size, projection=base_projection))
        if 7 in key_count:
            query_params[UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT] = ('=', '7k')
            query_params[Beatmap.CS] = ('=', 7)
            cursor += list(osu_utils.predict_score(self.connection, query_params, self.config.embedding_size, projection=base_projection))
        for x in cursor:
            score, bid, speed, variant, od, count1, count2, \
            star_ht, star_nm, star_dt, cs, version, name = x[:len(base_projection) + 1]
            # print(int(bid))
            if len(beatmap_ids) != 0 and int(bid) not in beatmap_ids:
                continue
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
                              od, count1 + count2, speed, variant, cs])
        data = pd.DataFrame(data_list,
                            columns=['id', "mod", "star", "pred_score", "pred_pp", "name",
                                     'od', 'count', 'speed', 'variant', 'cs'])
        data['pred_score'] = np.round(
            osu_utils.map_osu_score(data['pred_score'].to_numpy(), real_to_train=False)
        ).astype(int)
        data['pred_pp'] = osu_utils.mania_pp(data['pred_score'].to_numpy(), data['od'].to_numpy(),
                                             data['star'].to_numpy(),
                                             data['count'].to_numpy())
        measure_time(data.sort_values)(by="pred_pp", ascending=False, inplace=True)
        data: pd.DataFrame = data.iloc[:min(200, len(data)), :]
        data.set_index(['id', 'mod'], inplace=True)
        print(len(data))
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
        probable_scores, probable_pps, break_probs, \
        pp_gains, true_scores, true_pps, pass_probs, true_score_ids = [], [], [], [], [], [], [], []
        pass_feature_list = {-1: [], 0: [], 1: []}
        pass_feature_list_index = {-1: [], 0: [], 1: []}

        # estimate pp incre
        pass_features = None
        for i, row in enumerate(data.itertuples(name=None)):
            (bid, mod), star, pred_score, _, _, od, count, speed, variant, _ = row
            # pass rate feature. TODO: too slow!! 200ms
            pass_features, _ = self.pass_feature_feed.get_pass_features(uid, variant, bid, mod,
                                                                        is_predicting=True)
            pass_feature_list[speed].append(pass_features)
            pass_feature_list_index[speed].append(i)
            # break prob
            true_speed, true_score, true_pp, true_score_id = user_bp.get_score_pp(bid)
            if true_score is None:
                break_prob = 1.0
                probable_score = pred_score
            else:
                true_score = int(round(true_score))
                std_result = osu_utils.predict_score_std(self.connection, uid,
                                                         variant, self.config, bid, mod)
                if std_result is None or true_speed != speed:
                    probable_score = pred_score
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
            true_score_ids.append(int(true_score_id) if true_score_id is not None else None)
            # pp gain
            if probable_score is not None:
                pp_gain_beatmap = osu_utils.mania_pp(probable_score, od, star,
                                                     count)
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
        
        # calculate pass prob
        pass_probs = np.ones(len(probable_scores))
        for speed in [-1, 0, 1]:
            if len(pass_feature_list[speed]) != 0 and speed in self.speed_to_pass_model:
                dmatrix = xgboost.DMatrix(np.asarray(pass_feature_list[speed]))
                probs = self.speed_to_pass_model[speed].predict(dmatrix)
                pass_probs[pass_feature_list_index[speed]] = probs

        data['pred_score (breaking)'] = probable_scores
        data['pred_pp (breaking)'] = probable_pps
        data['break_prob'] = break_probs
        data["pass_prob"] = pass_probs
        data["pp_gain (breaking)"] = pp_gains
        data['true_score'] = true_scores
        data['true_pp'] = true_pps
        data['true_score_id'] = true_score_ids
        data['pp_gain_expect'] = data['pp_gain (breaking)'] * data['break_prob'] * data['pass_prob']

        data = pd.DataFrame(data,
                                  columns=['name', 'star',
                                           'true_score', 'true_pp',
                                           'pred_score', 'break_prob',
                                           # 'pred_score (breaking)', 'pred_pp (breaking)',
                                           'pp_gain (breaking)', 'pass_prob', 'cs',
                                           'pp_gain_expect', 'pred_pp', 'true_score_id'])
        data.sort_values(by="pp_gain_expect", ascending=False, inplace=True)
        return data

    def predict(self, uid, key_count=[4], beatmap_ids=[]):
        
        st = time.time()
        user_bp = osu_utils.get_user_bp(self.connection, uid, self.config, None)
        print(f"BP: {time.time() - st}")
        if len(user_bp.data) == 0:
            return None

        st = time.time()
        data = self.recall(uid, key_count, beatmap_ids)
        print(f"Recall time: {time.time() - st}")

        st = time.time()
        data = self.rank(uid, data, user_bp)
        print(f"Rank time: {time.time() - st}")
        return data

def save_excel(dfs, filename, idx_len):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheetname, df in dfs.items():  # loop through `dict` of dataframes
        df: pd.DataFrame
        df.to_excel(writer, sheet_name=sheetname, index=idx_len != 0)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
            worksheet.set_column(idx + idx_len, idx + idx_len, max_len)  # set column width
    writer.save()


if __name__ == "__main__":
    # uid = "10702235"  # ca
    # uid = "10500832"  # xz
    # uid = "7304075"  # ku
    # uid = "6701729" # serika
    # uid = "10817494"  # nickname
    # uid = "30281907"  # potassium
    # uid = "8586018"  # luoxuan

    import sys, json
    uid = sys.argv[-1]

    connection = repository.get_connection()
    recommender = PPRecommender(NetworkConfig(), connection)
    data = recommender.predict(uid)
    data_json = data.reset_index(inplace=False).to_json(orient='records', index=True)

    user_name = repository.select_first(connection, User.TABLE_NAME, project=[User.NAME],
                                        where={User.ID: uid})

    os.makedirs("report", exist_ok=True)
    save_excel({'Sort by PP gain expect': data,
                'Sort by break prob': data.sort_values(by=['break_prob', 'pp_gain_expect'],
                                                     ascending=False),
                'Sort by current BP': data.sort_values(by=['true_pp'], ascending=False),
                'Sort by pass prob': data.sort_values(by=['pass_prob', 'pp_gain_expect'])},
               os.path.join("report", f"{user_name[0]} - PP report.xlsx"), idx_len=2)

    st = time.time()
    similar_users = recommender.similarity(uid, "4k")
    print(f"Similarity time: {time.time() - st}")
    save_excel({
        "users": similar_users
    }, os.path.join("report", f"{user_name[0]} - similar users.xlsx"), 0)

    with open(os.path.join("report", f"{user_name[0]} - PP report.json"), "w") as f:
        f.write(data_json)
        # json.dump(data_json, f)