from abc import abstractmethod, ABCMeta
from collections import defaultdict

from scipy import stats, integrate

import osu_utils
import time
import train_pass_kernel
from data.model import *


class PPRuleSet(metaclass=ABCMeta):
    @abstractmethod
    def post_process_query_param(self, query_params: dict): pass

    @abstractmethod
    def train_to_real(self, x): pass

    @abstractmethod
    def real_to_train(self, x): pass

    @abstractmethod
    def pp(self, x, star, od, count): pass

    @abstractmethod
    def user_bp(self, connection, uid, config): pass

    @abstractmethod
    def db_column(self): pass

    def map_mod(self, mod):
        return mod


class ManiaPPV3(PPRuleSet):

    def post_process_query_param(self, query_params: dict):
        query_params[ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC] = ('=', '0')

    def train_to_real(self, x):
        return np.round(
            osu_utils.map_osu_score(x, real_to_train=False)
        ).astype(int)

    def real_to_train(self, x):
        return osu_utils.map_osu_score(x, real_to_train=True)

    def pp(self, x, star, od, count):
        return osu_utils.mania_pp(x, od, star, count)

    def user_bp(self, connection, uid, config):
        return osu_utils.get_user_bp(connection, uid, config, is_acc=False)

    def db_column(self):
        return Score.SCORE


class ManiaPPV4(PPRuleSet):

    def post_process_query_param(self, query_params: dict):
        query_params[ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC] = ('=', '1')

    def train_to_real(self, x):
        return osu_utils.map_osu_acc(x, real_to_train=False)

    def real_to_train(self, x):
        return osu_utils.map_osu_acc(x, real_to_train=True)

    def pp(self, x, star, od, count):
        return osu_utils.mania_pp_v4(x, star, count)

    def user_bp(self, connection, uid, config):
        return osu_utils.get_user_bp(connection, uid, config, is_acc=True)

    def map_mod(self, mod):
        return mod + "-ACC"

    def db_column(self):
        return Score.CUSTOM_ACCURACY


class PPRecommender:

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection, result_path="result",
                 rule=3):
        self.config = config
        self.connection = connection
        self.speed_to_pass_model = {}
        if rule == 3:
            self.pp_rule_set: PPRuleSet = ManiaPPV3()
        elif rule == 4:
            self.pp_rule_set: PPRuleSet = ManiaPPV4()
        else:
            raise

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
        st = time.time()
        for x in cursor:
            cur_embedding = np.asarray(x[:self.config.embedding_size])
            cur_uid, cur_name, cur_pp = x[-3:]
            distance = np.linalg.norm(embedding - cur_embedding)
            data.append([cur_uid, cur_name, cur_pp, distance])
        print(f"similarity: {time.time() - st} s")

        data_df = pd.DataFrame(data, columns=["id", "name", "pp", "distance"])
        data_df.sort_values(by=["distance", "id"], ascending=True, inplace=True)

        return data_df

    def map_beatmap_name(self, name, version):
        name = name + " - " + version
        # max_length = 60
        # end_length = 20
        # name = name[:max_length - end_length] + "..." + name[-end_length:] if len(name) > max_length else name
        return name

    def recall(self, uid, key_count=[4], beatmap_ids=[], max_star=None, max_size=300, min_star=0,
               min_pp=None):
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
            Beatmap.VERSION, Beatmap.NAME, Beatmap.SET_ID,
            # for validation
            BeatmapEmbedding.COUNT_HT, BeatmapEmbedding.COUNT_NM, BeatmapEmbedding.COUNT_DT
        ]
        data_list = []
        cursor = []
        query_params = {
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: ('=', uid),
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: ('=', self.config.game_mode),
        }
        self.pp_rule_set.post_process_query_param(query_params)
        if len(beatmap_ids) != 0:
            beatmap_ids = set(beatmap_ids)
        if 4 in key_count:
            query_params[UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT] = ('=', '4k')
            query_params[Beatmap.CS] = ('=', 4)
            cursor += list(
                osu_utils.predict_score(self.connection, query_params, self.config.embedding_size,
                                        projection=base_projection))
        if 7 in key_count:
            query_params[UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT] = ('=', '7k')
            query_params[Beatmap.CS] = ('=', 7)
            cursor += list(
                osu_utils.predict_score(self.connection, query_params, self.config.embedding_size,
                                        projection=base_projection))
        for x in cursor:
            score, bid, speed, variant, od, count1, count2, \
            star_ht, star_nm, star_dt, cs, version, name, set_id, \
            count_ht, count_nm, count_dt = x[:len(base_projection) + 1]
            # print(int(bid))
            if len(beatmap_ids) != 0 and int(bid) not in beatmap_ids:
                continue
            if speed == 1:
                star = star_dt
                mod = 'DT'
                count = count_dt
            elif speed == 0:
                star = star_nm
                mod = 'NM'
                count = count_nm
            else:
                star = star_ht
                mod = 'HT'
                count = count_ht
            if max_star is not None:
                if star > max_star:
                    continue
            if star < min_star:
                continue
            data_list.append([bid, mod, star, score, 0, self.map_beatmap_name(name, version),
                              od, count1 + count2, speed, variant, cs, set_id, count])
        data = pd.DataFrame(data_list,
                            columns=['id', "mod", "star", "pred_score", "pred_pp", "name",
                                     'od', 'count', 'speed', 'variant', 'cs', 'set_id',
                                     'valid_count'])
        data['pred_score'] = self.pp_rule_set.train_to_real(data['pred_score'].to_numpy())
        data['pred_pp'] = self.pp_rule_set.pp(data['pred_score'].to_numpy(),
                                              data['star'].to_numpy(),
                                              data['od'].to_numpy(),
                                              data['count'].to_numpy())
        measure_time(data.sort_values)(by="pred_pp", ascending=False, inplace=True)

        if min_pp is not None:
            pp_inverse = data['pred_pp'].to_numpy()[::-1]
            pp_pos = len(data) - bisect.bisect_left(list(pp_inverse), min_pp)
            max_size = max(max_size, pp_pos)
        max_size = min(len(data), max_size)

        data: pd.DataFrame = data.iloc[:max_size, :]
        data.set_index(['id', 'mod'], inplace=True)
        print(len(data))
        return data

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance):
        user_true_pp = user_bp.get_pp()

        def score_with_prob(x, pred_score, pred_score_std):
            # print(x)
            prob = np.exp(-(x - pred_score) ** 2 / (2 * pred_score_std ** 2)) / (
                    pred_score_std * np.sqrt(2 * np.pi))
            x_real = self.pp_rule_set.train_to_real(x)
            return prob * x_real

        # stage 2: rank precisely with pass rate, break prob, pp recom, etc.
        probable_scores, probable_pps, break_probs, \
        pp_gains, true_scores, true_pps, pass_probs, true_score_ids = [], [], [], [], [], [], [], []
        true_speeds = []
        pass_feature_list = defaultdict(lambda: [])
        pass_feature_list_index = defaultdict(lambda: [])

        # estimate pp incre
        for i, row in enumerate(data.itertuples(name=None)):
            (bid, mod), star, pred_score, _, _, od, count, speed, variant, _, _, _ = row
            pass_feature_list[f"{speed}/{variant}"].append(bid)
            pass_feature_list_index[f"{speed}/{variant}"].append(i)
            # break prob
            true_speed, true_score, true_pp, true_star, true_score_id = user_bp.get_score_pp(bid)
            if true_score is not None:
                true_pp = self.pp_rule_set.pp(true_score, true_star, od, count)
            if true_score is None:
                break_prob = 1.0
                probable_score = pred_score
            else:
                std_result = osu_utils.predict_score_std(self.connection, uid,
                                                         variant, self.config, bid,
                                                         self.pp_rule_set.map_mod(mod))
                if std_result is None or true_speed != speed:
                    probable_score = pred_score
                    break_prob = 1.0
                else:
                    score_train, score_std_train = std_result
                    true_score_train = self.pp_rule_set.real_to_train(true_score)
                    break_prob = 1 - stats.norm.cdf(true_score_train, loc=score_train,
                                                    scale=score_std_train)
                    probable_score = \
                        integrate.quad(score_with_prob, a=true_score_train, b=np.inf,
                                       args=(score_train, score_std_train),
                                       epsabs=1.0)[0] / break_prob
            true_scores.append(true_score)
            true_pps.append(true_pp)
            true_speeds.append(true_speed)
            probable_scores.append(probable_score)
            break_probs.append(round(break_prob, 6))
            true_score_ids.append(int(true_score_id) if true_score_id is not None else None)
            # pp gain
            if probable_score is not None:
                pp_gain_beatmap = self.pp_rule_set.pp(probable_score, star, od, count)
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
        for key in pass_feature_list:
            speed, variant = key.split("/")
            probs = train_pass_kernel.estimate_pass_probability(uid, variant,
                                                                pass_feature_list[key], speed,
                                                                self.config, self.connection)
            pass_probs[pass_feature_list_index[key]] = probs

        data['pred_score (breaking)'] = probable_scores
        data['pred_pp (breaking)'] = probable_pps
        data['break_prob'] = break_probs
        data["pass_prob"] = pass_probs
        data["pp_gain (breaking)"] = pp_gains
        data['true_score'] = true_scores
        data['true_pp'] = true_pps
        data['true_speed'] = true_speeds
        data['true_score_id'] = true_score_ids
        data['pp_gain_expect'] = data['pp_gain (breaking)'] * data['break_prob'] * data['pass_prob']

        data = pd.DataFrame(data,
                            columns=['name', 'star',
                                     'true_score', 'true_pp',
                                     'pred_score', 'break_prob',
                                     'pred_score (breaking)', 'pred_pp (breaking)',
                                     'pp_gain (breaking)', 'pass_prob', 'cs', 'set_id',
                                     'valid_count', 'true_speed',
                                     'pp_gain_expect', 'pred_pp', 'true_score_id'])
        data.sort_values(by="pp_gain_expect", ascending=False, inplace=True)
        return data

    def predict(self, uid, key_count=[4], beatmap_ids=[], max_star=None, max_size=300, min_star=0,
                min_pp=None):

        st = time.time()
        user_bp = self.pp_rule_set.user_bp(self.connection, uid, self.config)
        print(f"BP: {time.time() - st}")
        if len(user_bp.data) == 0:
            return None

        st = time.time()
        data = self.recall(uid, key_count, beatmap_ids, max_star, max_size, min_star, min_pp)
        print(f"Recall time: {time.time() - st}")

        st = time.time()
        data = self.rank(uid, data, user_bp)
        print(f"Rank time: {time.time() - st}")
        return data

    def draw_prediction_distribution_diagram(self, user_id, variant, beatmap_id, mod):
        score_train, score_std_train = osu_utils.predict_score_std(self.connection, user_id,
                                                                   variant, self.config,
                                                                   beatmap_id,
                                                                   self.pp_rule_set.map_mod(mod))
        if mod == "DT":
            speed = 1
        elif mod == 'HT':
            speed = -1
        else:
            speed = 0
        true_score = repository.select_first(self.connection, table_name=Score.TABLE_NAME,
                                             project=[self.pp_rule_set.db_column()],
                                             where={
                                                 Score.USER_ID: user_id,
                                                 Score.BEATMAP_ID: beatmap_id,
                                                 Score.SPEED: speed
                                             })
        max_score_train = score_train + score_std_train * 3
        min_score_train = score_train - score_std_train * 3
        max_score = self.pp_rule_set.train_to_real(max_score_train)
        min_score = self.pp_rule_set.train_to_real(min_score_train)
        x = np.linspace(min_score, max_score, num=100)
        x_train = self.pp_rule_set.real_to_train(x)
        probs = np.exp(-(x_train - score_train) ** 2 / (2 * score_std_train ** 2))
        probs = [round(x, ndigits=4) for x in probs.tolist()]

        return min_score, max_score, probs, true_score[0] if true_score is not None else None



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
    import sys
    import matplotlib.pyplot as plt

    uid = sys.argv[-1]

    connection = repository.get_connection()
    recommender = PPRecommender(NetworkConfig(), connection, rule=4)
    data = recommender.predict(uid)
    data_json = data.reset_index(inplace=False).to_json(orient='records', index=True)

    for i in range(10):
        x = data.iloc[i]
        bid, mod = x.name

        min_score, max_score, probs, true_score = recommender.draw_prediction_distribution_diagram(uid, '4k', str(bid), mod)
        x = np.linspace(min_score, max_score, num=len(probs))

        true_score_index = 0
        if true_score is not None:
            true_score_index = int((true_score - min_score) / (max_score - min_score) * len(x))
            if true_score_index >= len(x):
                true_score_index = len(x) - 1

        plt.clf()
        plt.ylim([0, 1.05])
        plt.fill_between(x[true_score_index:], probs[true_score_index:], color='b', alpha=0.3)
        if true_score is not None:
            plt.text(x[true_score_index], probs[true_score_index] - 0.05,
                     ' ← Current', fontsize=10, color='black')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.plot(x, probs)
        plt.savefig(f"{bid}.png")

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
