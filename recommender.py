import pickle
import time

import osu_utils
from data.model import *
from recommend import PPRuleSet, ManiaPPV3, ManiaPPV4, STDPP


class PPRecommender:

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection, result_path="result",
                 rule=3):
        self.config = config
        self.connection = connection
        self.speed_to_pass_model = {}
        if config.game_mode == "osu":
            self.pp_rule_set = STDPP(config, connection)
        elif config.game_mode == 'mania':
            if rule == 3:
                self.pp_rule_set: PPRuleSet = ManiaPPV3(config, connection)
            elif rule == 4:
                self.pp_rule_set: PPRuleSet = ManiaPPV4(config, connection)
            else:
                raise
        else:
            raise

    def similarity_tree(self, uid, variant):

        embedding_names = self.config.get_embedding_names(UserEmbedding.EMBEDDING)
        embedding = repository.select_first(self.connection, UserEmbedding.TABLE_NAME,
                                            project=embedding_names,
                                            where={
                                                UserEmbedding.GAME_MODE: self.config.game_mode,
                                                UserEmbedding.VARIANT: variant,
                                                UserEmbedding.USER_ID: uid
                                            })
        if embedding is None:
            return None
        embedding = np.asarray([embedding])

        with open(self.config.ball_tree_path, 'rb') as f:
            nbrs, user_ids, first_variant = pickle.load(f)

        if first_variant != variant:
            # trick: add a large value to the first dimension if variant is different
            embedding[0, 0] += 50

        nbrs_distance, nbrs_index = nbrs.kneighbors(embedding)
        nbrs_user_ids = user_ids[nbrs_index[0]]
        nbrs_distance = nbrs_distance[0]
        id_to_distance = dict(zip(nbrs_user_ids, nbrs_distance))

        cursor = repository.execute_sql(
            self.connection,
            (f"SELECT {User.ID}, {User.NAME}, {User.PP} "
             f"FROM User "
             f"WHERE {User.ID} IN ({','.join(map(str, id_to_distance.keys()))}) "
             f"AND {User.GAME_MODE} == '{self.config.game_mode}' "
             f"AND {User.VARIANT} == '{variant}' ")
        )
        data = []
        for x in cursor:
            cur_uid, cur_name, cur_pp = x
            distance = id_to_distance.get(int(cur_uid), -100000)
            data.append([cur_uid, cur_name, cur_pp, distance])
        data_df = pd.DataFrame(data, columns=["id", "name", "pp", "distance"])
        data_df.sort_values(by=["distance", "id"], ascending=True, inplace=True)

        return data_df

    def similarity(self, uid, variant):

        result = \
            repository.select(self.connection, UserEmbedding.TABLE_NAME,
                              [UserEmbedding.NEIGHBOR_ID,
                               UserEmbedding.NEIGHBOR_DISTANCE],
                              where={
                                  UserEmbedding.USER_ID: uid,
                                  UserEmbedding.VARIANT: variant,
                                  UserEmbedding.GAME_MODE: self.config.game_mode
                              }).fetchone()
        if result is None:
            return None
        cur_nbrs_ids, cur_nbrs_distance = result
        cur_nbrs_ids = list(repository.db_to_np(cur_nbrs_ids))
        cur_nbrs_distance = list(repository.db_to_np(cur_nbrs_distance))
        id_to_distance = dict(zip(cur_nbrs_ids, cur_nbrs_distance))

        cursor = repository.execute_sql(
            self.connection,
            (f"SELECT {User.ID}, {User.NAME}, {User.PP} "
             f"FROM User "
             f"WHERE {User.ID} IN ({','.join(map(str, cur_nbrs_ids))}) "
             f"AND {User.GAME_MODE} == '{self.config.game_mode}' "
             f"AND {User.VARIANT} == '{variant}' ")
        )
        st = time.time()
        data = []
        for x in cursor:
            cur_uid, cur_name, cur_pp = x
            distance = id_to_distance.get(int(cur_uid), -100000)
            data.append([cur_uid, cur_name, cur_pp, distance])
        print(f"similarity: {time.time() - st} s")

        data_df = pd.DataFrame(data, columns=["id", "name", "pp", "distance"])
        data_df.sort_values(by=["distance", "id"], ascending=True, inplace=True)

        return data_df

    def recall(self, uid, key_count=None, beatmap_ids=None, max_star=None, max_size=300, min_star=0,
               required_mods=None,
               min_pp=None):
        # stage 1: recall the maps with the highest possible pp
        if beatmap_ids is None:
            beatmap_ids = []
        if key_count is None:
            key_count = []
        if required_mods is None:
            required_mods = []
        if len(beatmap_ids) != 0:
            beatmap_ids = set(beatmap_ids)
        data = self.pp_rule_set.generate_recall_table(uid, key_count, beatmap_ids, max_star,
                                                      min_star, required_mods, min_pp)
        measure_time(data.sort_values)(by="pred_pp", ascending=False, inplace=True)

        if min_pp is not None:
            pp_inverse = data['pred_pp'].to_numpy()[::-1]
            pp_pos = len(data) - bisect.bisect_left(list(pp_inverse), min_pp)
            max_size = max(max_size, pp_pos)
        max_size = min(len(data), max_size)

        data: pd.DataFrame = data.iloc[:max_size, :]
        data.set_index(['id', 'mod'], inplace=True)
        print("recall length:", len(data))
        return data

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance):
        return self.pp_rule_set.rank(uid, data, user_bp)

    def predict(self, uid, key_count, beatmap_ids=[], max_star=None, max_size=300, min_star=0,
                min_pp=None, required_mods=None):

        with self.pp_rule_set.timing("[Get-BP]"):
            user_bp = self.pp_rule_set.user_bp(uid)
        if len(user_bp.data) == 0:
            return None

        with self.pp_rule_set.timing("[Recall-total]"):
            data = self.recall(uid, key_count, beatmap_ids, max_star, max_size, min_star, required_mods, min_pp)

        with self.pp_rule_set.timing("[Rank-total]"):
            data = self.rank(uid, data, user_bp)
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


def test_mania():
    import sys
    import matplotlib.pyplot as plt

    uid = sys.argv[-1]
    config = NetworkConfig.from_config("config/mania.json")

    connection = repository.get_connection()
    recommender = PPRecommender(config, connection, rule=4)
    data = recommender.predict(uid, key_count=[4])
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


def test_std():
    import sys

    uid = sys.argv[-1]
    config = NetworkConfig.from_config("config/osu.json")

    connection = repository.get_connection()
    user_name = repository.select_first(connection, User.TABLE_NAME, project=[User.NAME],
                                        where={User.ID: uid})
    print(user_name)
    recommender = PPRecommender(config, connection)
    user_beatmap_ids = connection.execute(
        "SELECT beatmap_id, pp FROM Score WHERE user_id == " + uid)
    beatmap_ids_to_real_pp = {}
    for x in user_beatmap_ids:
        beatmap_ids_to_real_pp[x[0]] = x[1]

    results = {}
    results_recommend = {}
    user_bp = recommender.pp_rule_set.user_bp(uid)
    for mod in osu_utils.STD_MODS.values():
        data = recommender.recall(uid, beatmap_ids=[], required_mods=mod,
                                  min_pp=min(user_bp.pp_order_list), max_size=3000)
        results["".join(mod)] = data.copy()
        data = recommender.rank(uid, data, user_bp)
        results_recommend["".join(mod)] = data.copy()
    save_excel(results, f"report/std/{user_name[0]}_bp.xlsx", 0)
    save_excel(results_recommend, f"report/std/{user_name[0]}_recommend.xlsx", 0)

    st = time.time()
    similar_users = recommender.similarity(uid, "")
    print(f"Similarity time: {time.time() - st}")

    st = time.time()
    similar_users_db = recommender.similarity_tree(uid, "")
    print(f"Similarity time db: {time.time() - st}")

    save_excel({
        "users": similar_users,
        "db": similar_users_db,
    }, os.path.join("report", "std", f"{user_name[0]} - similar users.xlsx"), 0)


if __name__ == "__main__":
    test_mania()
    # test_std()
