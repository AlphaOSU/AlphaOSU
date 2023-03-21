import time
import pickle

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

import osu_utils
from data import data_process
from data.model import *
from score import ManiaScoreDataProvider, STDScoreDataProvider


def linear_square(scores, x, weights, epoch, config):
    """
    Find a W to minimize ||scores - W^T x||^2, using Bayesian Linear Square.
    @param scores: shape = [N, ]
    @param x: shape = [N, E]
    @param weights: shape = [N, ], the weight for each data sample
    @param epoch: training epoch
    @param config: training config
    @return: (emb, sigma, alpha, (r2, mse, r2_adj))
        - emb: shape = [E, ], the optimal W
        - sigma, alpha: shape = [E, E] and [1], the Bayesian parameters for estimating uncertainty
        - r2, mse, r2_adj: training statistics
    """
    y = scores

    regr = BayesianRidge(fit_intercept=False)
    regr.fit(x, y, weights)
    y_pred = regr.predict(x)
    r2 = r2_score(y, y_pred, sample_weight=weights)
    n = len(x)
    if n > config.embedding_size + 1:
        r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - config.embedding_size - 1)
    else:
        r2_adj = r2
    mse = mean_absolute_error(y, y_pred, sample_weight=weights)
    emb = regr.coef_
    sigma = regr.sigma_
    alpha = regr.alpha_

    return (emb, sigma, alpha, (r2, mse, r2_adj))


class TrainingStatistics:

    def __init__(self, desc, total):
        self.io_times = []
        self.r2_list = []
        self.mse_list = []
        self.ls_times = []
        self.as_times = []
        self.r2_adj_list = []
        self.desc = desc
        self.total = total


cache = {}


def mean(arr):
    l = len(arr)
    if l == 0:
        return 0
    return sum(arr) / l


def train_embedding(key, get_data_method, epoch, config,
                    embedding_data: EmbeddingData,
                    training_statistics: TrainingStatistics, pbar,
                    other_embedding_data: EmbeddingData,
                    other_embedding_data2: EmbeddingData,
                    cachable=True):
    """
    A common method to train embedding (user / beatmap / mod). Traning results will be saved in
    embedding_data.
    For example, when training user embedding, other_embedding_data / other_embedding_data2 are
    the beatmap / mod embedding data. key is the user_key.
    @param key: target key
    @param get_data_method: the method to get training data for key
    @param epoch: training epoch
    @param config: config
    @param embedding_data: an EmbeddingData to save the embedding for key
    @param training_statistics: a TrainingStatistics to save the statistics
    @param pbar: progress bar
    @param other_embedding_data: an EmbeddingData oppository to embedding_data
    @param other_embedding_data2: another EmbeddingData oppository to embedding_data
    @param cachable: can the data returned by get_data_method be cached to save time
    @return: nothing
    """
    time_io = time.time()
    global cache
    if key in cache:
        data = cache[key]
    else:
        data = get_data_method(key, epoch)
        if cachable:
            cache[key] = data
    if data is None:
        return None
    time_io = time.time() - time_io

    scores, other_emb_id, other_emb_id2, regression_weights = data
    other_embs = other_embedding_data.embeddings[0][other_emb_id]
    other_embs2 = other_embedding_data2.embeddings[0][other_emb_id2]
    x = other_embs * other_embs2

    time_ls = time.time()
    (emb, sigma, alpha, metrics) = linear_square(scores, x, regression_weights, epoch, config)
    time_ls = time.time() - time_ls

    emb_id = embedding_data.key_to_embed_id[key]
    embedding_data.embeddings[0][emb_id] = emb
    if sigma is not None:
        embedding_data.sigma[emb_id] = sigma
    if alpha is not None:
        embedding_data.alpha[emb_id] = alpha

    if metrics[0] >= -2 and metrics[2] >= -2:
        training_statistics.r2_list.append(metrics[0])
        training_statistics.mse_list.append(metrics[1])
        training_statistics.r2_adj_list.append(metrics[2])
        training_statistics.io_times.append(time_io)
        training_statistics.ls_times.append(time_ls)

    time_assign = time.time()

    if len(training_statistics.io_times) % 100 == 0:
        io_time_mean = np.mean(training_statistics.io_times) * 1000
        ls_time_mean = np.mean(training_statistics.ls_times) * 1000
        time_assign = time.time() - time_assign
        training_statistics.as_times.append(time_assign)
        as_times_mean = np.mean(training_statistics.as_times) * 1000
        pbar.set_description(f"[{epoch}] {training_statistics.desc} io:{io_time_mean:.2f}ms "
                             f"ls:{ls_time_mean:.2f}ms "
                             f"as:{as_times_mean:.2f}ms "
                             f"r2:{mean(training_statistics.r2_list):.4f} "
                             f"r2_adj:{mean(training_statistics.r2_adj_list):.4f} "
                             f"mse:{mean(training_statistics.mse_list):.4f}")


def train_personal_embedding(key, weights, config, connection,
                             epoch, embedding_data: EmbeddingData,
                             other_embedding_data: EmbeddingData,
                             other_embedding_data2: EmbeddingData):
    """
    A common method to train user embedding. Training results will be saved in embedding_data.
    @param key: a key including user_id, game_mode and variant in combination with '-'
    @param weights: ScoreModelWeight
    @param config: training config
    @param connection: database connection
    @param epoch: training epoch, but it only needs to calculate once because the model had been convergence
    @param embedding_data: user embedding data
    @param other_embedding_data: beatmap embedding data
    @param other_embedding_data2: mod embedding data
    @return: length of score, for counting how many scores are used for training user embedding
    """
    if config.game_mode == 'osu':
        provider = STDScoreDataProvider(weights, config, connection)
    elif config.game_mode == 'mania':
        provider = ManiaScoreDataProvider(weights, config, connection)
    else:
        raise
    data = provider.provide_user_data(key, epoch, ignore_less=False)
    if data is None:
        return 0

    scores, other_emb_id, other_emb_id2, regression_weights = data
    other_embs = other_embedding_data.embeddings[0][other_emb_id]
    other_embs2 = other_embedding_data2.embeddings[0][other_emb_id2]
    x = other_embs * other_embs2

    (emb, sigma, alpha, metrics) = linear_square(scores, x, regression_weights, epoch, config)

    emb_id = embedding_data.key_to_embed_id[key]
    embedding_data.embeddings[0][emb_id] = emb
    embedding_data.sigma[emb_id] = sigma
    embedding_data.alpha[emb_id] = alpha

    return len(scores)


def train_personal_embedding_online(config: NetworkConfig, key, connection):
    """
    integrate functions among training personal embedding,
    updating neighbors for pass probability estimation and saving the results
    @param config: training config
    @param key: a key including user_id, game_mode and variant in combination with '-'
    @param connection: database connection
    """
    user_key = key

    weights = data_process.load_weight_online(config, user_key, connection)
    weights.beatmap_embedding.key_to_embed_id = weights.beatmap_embedding.key_to_embed_id.to_dict()
    weights.user_embedding.key_to_embed_id = weights.user_embedding.key_to_embed_id.to_dict()
    weights.mod_embedding.key_to_embed_id = weights.mod_embedding.key_to_embed_id.to_dict()
    constraint = UserEmbedding.construct_where_with_key(user_key)

    # train the user embedding
    count = train_personal_embedding(user_key, weights, config, connection, 100,
                                     weights.user_embedding, weights.beatmap_embedding,
                                     weights.mod_embedding)

    # update neighbors for pass probability estimation
    if os.path.isfile(config.ball_tree_path):
        embedding = weights.user_embedding.embeddings[0][
            weights.user_embedding.key_to_embed_id[user_key]
        ]
        embedding = np.copy(embedding)
        embedding = np.reshape(embedding, (1, -1))
        with open(config.ball_tree_path, 'rb') as f:
            nbrs, user_ids, first_variant = pickle.load(f)

        if first_variant != user_key.split("-")[-1]:
            # trick: add a large value to the first dimension if variant is different
            embedding[0, 0] += 50

        nbrs_distance, nbrs_index = nbrs.kneighbors(embedding)

        neighbor_id = user_ids[nbrs_index[0]].tolist()
        neighbor_id.insert(0, user_key.split("-")[0])
        neighbor_id = np.asarray(neighbor_id, dtype=np.int32)

        neighbor_distance = nbrs_distance[0].tolist()
        neighbor_distance.insert(0, 0.0)
        neighbor_distance = np.asarray(neighbor_distance, dtype=np.float64)

    else:
        #TODO: Compat codes. Remove it in the future
        print("WARNING: ball tree path not exists. Fall back to brute force!")

        embedding = weights.user_embedding.embeddings[0][
            weights.user_embedding.key_to_embed_id[user_key]]
        project = config.get_embedding_names(UserEmbedding.EMBEDDING)
        cursor = repository.select(connection, UserEmbedding.TABLE_NAME,
                                   project=project + [UserEmbedding.USER_ID],
                                   where={
                                       UserEmbedding.GAME_MODE: constraint[UserEmbedding.GAME_MODE],
                                       UserEmbedding.VARIANT: constraint[UserEmbedding.VARIANT]
                                   })
        data = []
        for x in cursor:
            cur_embedding = np.asarray(x[:config.embedding_size])
            cur_uid = x[-1]
            distance = np.linalg.norm(embedding - cur_embedding)
            data.append([cur_uid, distance])
        data_df = pd.DataFrame(data, columns=["id", "distance"])
        data_df.sort_values(by=["distance"], ascending=True, inplace=True)
        data_df = data_df[:150]
        neighbor_id = data_df['id'].to_numpy().astype(np.int32)
        neighbor_distance = data_df['distance'].to_numpy()

    # save the results
    with connection:
        primary_keys = ", ".join(UserEmbedding.PRIMARY_KEYS)
        values = user_key.split("-")
        sql = (f"INSERT OR IGNORE INTO `{UserEmbedding.TABLE_NAME}` "
               f"({primary_keys}) "
               f'VALUES ({values[0]}, "{values[1]}", "{values[2]}")')
        repository.execute_sql(connection, sql)
        data_process.save_embedding(connection, weights.user_embedding, config,
                                    UserEmbedding.TABLE_NAME,
                                    UserEmbedding.EMBEDDING)
        repository.update(connection, UserEmbedding.TABLE_NAME, puts=[{
            UserEmbedding.COUNT: count,
            UserEmbedding.NEIGHBOR_ID: repository.np_to_db(neighbor_id),
            UserEmbedding.NEIGHBOR_DISTANCE: repository.np_to_db(neighbor_distance),
        }], wheres=[constraint])

def debug_mod_embedding(mod_embedding: EmbeddingData, config: NetworkConfig):
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) + 1e-6) / (np.linalg.norm(b) + 1e-6)
    for idx_i, i in enumerate(mod_embedding.key_to_embed_id.keys()):
        for idx_j, j in enumerate(mod_embedding.key_to_embed_id.keys()):
            if idx_i <= idx_j:
                continue
            ii = mod_embedding.key_to_embed_id[i]
            jj = mod_embedding.key_to_embed_id[j]
            sim = cos_sim(mod_embedding.embeddings[0][ii],
                          mod_embedding.embeddings[0][jj])
            if config.game_mode == 'osu':
                mod_i = "".join(osu_utils.STD_MODS[str(i)])
                mod_j = "".join(osu_utils.STD_MODS[str(j)])
            else:
                mod_i = i
                mod_j = j
            print(f"{mod_i} <-> {mod_j}: {sim:.5f}  ", end="")
        print()

def train_score_by_als(config: NetworkConfig, connection: sqlite3.Connection):

    with connection:
        # repository.execute_sql(connection, f"DROP TABLE IF EXISTS {BeatmapEmbedding.TABLE_NAME}")
        # repository.execute_sql(connection, f"DROP TABLE IF EXISTS {UserEmbedding.TABLE_NAME}")
        # repository.execute_sql(connection, f"DROP TABLE IF EXISTS {ModEmbedding.TABLE_NAME}")
        repository.create_index(connection, "score_user", Score.TABLE_NAME, [Score.USER_ID])
        repository.create_index(connection, "score_beatmap", Score.TABLE_NAME, [Score.BEATMAP_ID])
    previous_r2 = -10000

    weights = data_process.load_weight(config)
    weights.beatmap_embedding.key_to_embed_id = weights.beatmap_embedding.key_to_embed_id.to_dict()
    weights.user_embedding.key_to_embed_id = weights.user_embedding.key_to_embed_id.to_dict()
    weights.mod_embedding.key_to_embed_id = weights.mod_embedding.key_to_embed_id.to_dict()

    if config.game_mode == 'osu':
        provider_cls = STDScoreDataProvider
    elif config.game_mode == 'mania':
        provider_cls = ManiaScoreDataProvider
    else:
        raise
    provider = provider_cls(weights, config, connection)

    def provide_user_data(k, e):
        return provider.provide_user_data(k, e)

    def provide_beatmap_data(k, e):
        return provider.provide_beatmap_data(k, e)

    def provide_mod_data(k, e):
        return provider.provide_mod_data(k, e)

    debug_mod_embedding(weights.mod_embedding, config)

    for epoch in range(200):

        # train beatmap
        statistics = TrainingStatistics("beatmap", len(weights.beatmap_embedding.key_to_embed_id))
        pbar = tqdm(weights.beatmap_embedding.key_to_embed_id.keys(), desc=statistics.desc)
        for beatmap_key in pbar:
            train_embedding(beatmap_key, provide_beatmap_data, epoch, config,
                            weights.beatmap_embedding, statistics, pbar, weights.user_embedding,
                            weights.mod_embedding, cachable=False)
        tqdm.write(pbar.desc)

        # train user
        statistics = TrainingStatistics("user", len(weights.user_embedding.key_to_embed_id))
        pbar = tqdm(weights.user_embedding.key_to_embed_id.keys(), desc=statistics.desc)
        for user_key in pbar:
            train_embedding(user_key, provide_user_data, epoch, config,
                            weights.user_embedding, statistics, pbar, weights.beatmap_embedding,
                            weights.mod_embedding, cachable=False)
        tqdm.write(pbar.desc)

        cur_r2 = mean(statistics.r2_adj_list)
        if previous_r2 >= cur_r2:
            break
        previous_r2 = cur_r2
        data_process.save_embedding(connection, weights.beatmap_embedding, config,
                                    BeatmapEmbedding.TABLE_NAME,
                                    BeatmapEmbedding.ITEM_EMBEDDING)
        data_process.save_embedding(connection, weights.user_embedding, config,
                                    UserEmbedding.TABLE_NAME,
                                    UserEmbedding.EMBEDDING)
        data_process.save_embedding(connection, weights.mod_embedding, config,
                                    ModEmbedding.TABLE_NAME,
                                    ModEmbedding.EMBEDDING)
        Meta.save(connection, "score_embedding_version",
                  time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        connection.commit()

        # train mod
        if epoch >= 2 and epoch <= 15:
            statistics = TrainingStatistics("mod", len(weights.mod_embedding.key_to_embed_id))
            pbar = tqdm(weights.mod_embedding.key_to_embed_id.keys(), desc=statistics.desc)
            for mod_key in pbar:
                train_embedding(mod_key, provide_mod_data, epoch, config,
                                weights.mod_embedding, statistics, pbar, weights.user_embedding,
                                weights.beatmap_embedding, cachable=False)
            tqdm.write(pbar.desc)

            debug_mod_embedding(weights.mod_embedding, config)

def update_score_count(config: NetworkConfig, conn: sqlite3.Connection):
    """
    Update pc for users and beatmaps.
    @param config: config
    @param conn: db connection
    @return: None
    """
    print("Update score count")
    repository.ensure_column(conn, UserEmbedding.TABLE_NAME, [("count", "integer", 0)])
    repository.ensure_column(conn, BeatmapEmbedding.TABLE_NAME, [("count_HT", "integer", 0)])
    repository.ensure_column(conn, BeatmapEmbedding.TABLE_NAME, [("count_NM", "integer", 0)])
    repository.ensure_column(conn, BeatmapEmbedding.TABLE_NAME, [("count_DT", "integer", 0)])
    for speed in [-1, 0, 1]:
        mod = ['HT', 'NM', 'DT'][speed + 1]
        repository.execute_sql(conn,
                               f"UPDATE BeatmapEmbedding SET count_{mod} = ("
                               f"SELECT COUNT(1) FROM Score "
                               f"WHERE Score.beatmap_id == BeatmapEmbedding.id "
                               f"AND Score.speed == {speed})")
    user_sql = ("UPDATE UserEmbedding SET count = ("
                "SELECT COUNT(1) FROM Score "
                "WHERE Score.user_id == UserEmbedding.id "
                "AND Score.game_mode == UserEmbedding.game_mode ")
    if config.game_mode == 'mania':
        user_sql += "AND (Score.cs || 'k') == UserEmbedding.variant)"
    else:
        user_sql += ")"
    repository.execute_sql(conn, user_sql)
    conn.commit()


def prepare_var_param(config: NetworkConfig):
    connection = repository.get_connection()
    x = repository.select(connection, Score.SCORE, project=[
        Score.BEATMAP_ID, Score.CS, Score.USER_ID, Score.SCORE, Score.IS_EZ, Score.PP, Score.SPEED,
        Score.CUSTOM_ACCURACY
    ])
    data = []
    for cursor in tqdm(x, total=3750000):
        bid, cs, uid, score, is_ez, pp, speed, acc = cursor
        if pp < 1 or is_ez:
            continue
        if speed == -1:
            score = score * 2
        if score < osu_utils.min_score or acc < osu_utils.min_acc:
            continue
        score_std_re = osu_utils.predict_score_std(connection, uid,
                                                   f"{cs}k",
                                                   config, bid,
                                                   ['HT', 'NM', 'DT'][
                                                       speed + 1])
        if score_std_re is None:
            continue
        predict, var_beatmap, var_user, var_mod = score_std_re
        score_train = osu_utils.map_osu_score(score, real_to_train=True)
        data.append((score_train, predict, var_beatmap, var_user, var_mod))

    data = pd.DataFrame(data, columns=['y', 'predict', 'var_b', 'var_u', 'var_m'])
    data.to_sql("VarTest", connection, if_exists='replace')


def estimate_var_param(config):
    connection = repository.get_connection()
    data = pd.read_sql_query("SELECT * FROM VarTest", connection)
    diff = np.abs(data['y'].to_numpy() - data['predict'].to_numpy())
    var_b = data['var_b'].to_numpy()
    var_u = data['var_u'].to_numpy()
    var_m = data['var_m'].to_numpy()
    scale = 0.001

    def error(x):
        b, u, m = list(x / scale)
        std = np.sqrt(var_b * b + var_u * u + var_m * m)
        prob = np.mean(diff < std)
        print(b, u, m, prob)
        return (0.68269 - prob) ** 2

    from scipy import optimize
    initial = np.asarray((1 / 3, 1 / 3, 1 / 3))

    optimize.minimize(error, initial * scale)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        _config = NetworkConfig(json.load(open(os.path.join(sys.argv[1], "config.json"))))
    else:
        _config = NetworkConfig.from_config("config/osu.json")
    conn = repository.get_connection()
    train_score_by_als(_config, conn)
    update_score_count(_config, conn)
    # estimate_var_param(config)
