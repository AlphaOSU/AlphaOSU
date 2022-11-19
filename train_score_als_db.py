import json
import math
import time

from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

import osu_utils
from data import data_process
from data.model import *


def linear_square(scores, x, weights, epoch, config):
    """
    Find a W to minimize ||scores - W^T x||^2, using Bayesian Linear Square.
    :param scores: shape = [N, ]
    :param x: shape = [N, E]
    :param weights: shape = [N, ], the weight for each data sample
    :param epoch: training epoch
    :param config: training config
    :return: (emb, sigma, alpha, (r2, mse, r2_adj))
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


speed_to_mod_map = ['HT', 'NM', 'DT']


def get_user_train_data(user_key, weights: ScoreModelWeight,
                        config: NetworkConfig, connection, epoch):
    """
    get the training data (scores) for a user from database
    :param user_key: the target user, in the format of {user_id}-{game_mode}-{variant}
    :param weights: model weights
    :param config: training config
    :param connection: database connection
    :param epoch: training epoch
    :return: (scores, beatmap_id_array, mod_id_array, weights)
        - scores: shape = [N, ]
        - beatmap_id_array: shape = [N, ], the embedding id of weights.beatmap_embedding
        - mod_id_array: shape = [N, ], the embedding id of weights.mod_embedding
        - weights: shape = [N, ], training weight for each data sample.
    """
    user_id, user_mode, user_variant = user_key.split("-")
    if user_mode != config.game_mode:
        return None
    sql = (
        f"SELECT {Score.BEATMAP_ID}, s.speed, s.score, s.pp, s.{Score.CUSTOM_ACCURACY} "
        f"FROM {Score.TABLE_NAME} as s "
        f"WHERE s.{Score.USER_ID} == {user_id} "
        f"AND s.{Score.CS} == {user_variant[0]} "
        f"AND s.{Score.GAME_MODE} == '{config.game_mode}' "
        f"AND s.{Score.PP} >= 1 "
        f"AND NOT s.{Score.IS_EZ}")
    scores = []
    mod_emb_id = []
    beatmap_emb_id = []
    pps = []
    for x in repository.execute_sql(connection, sql):
        score = x[2] * 2 if x[1] == -1 else x[2]
        if score < osu_utils.min_score:
            continue
        acc = x[-1]
        if acc < osu_utils.min_acc:
            continue

        # scores:
        beatmap_emb_id.append(weights.beatmap_embedding.key_to_embed_id[str(x[0])])
        mod_emb_id.append(weights.mod_embedding.key_to_embed_id[speed_to_mod_map[x[1] + 1]])
        scores.append(
            osu_utils.map_osu_score(score, real_to_train=True, arctanh=math.atanh, tanh=math.tanh))
        pps.append(x[3])

        # accs:
        beatmap_emb_id.append(weights.beatmap_embedding.key_to_embed_id[str(x[0])])
        mod_emb_id.append(
            weights.mod_embedding.key_to_embed_id[speed_to_mod_map[x[1] + 1] + "-ACC"])
        scores.append(
            osu_utils.map_osu_acc(acc, real_to_train=True, arctanh=math.atanh, tanh=math.tanh))
        pps.append(x[3])
    if len(scores) < config.embedding_size:
        return None
    pps = np.asarray(pps)

    regression_weights = np.clip(pps / np.mean(pps), 1 / config.pp_weight_clip,
                                 config.pp_weight_clip)
    return (np.asarray(scores, dtype=np.float32),
            np.asarray(beatmap_emb_id, dtype=np.int32),
            np.asarray(mod_emb_id, dtype=np.int32),
            regression_weights)


def get_beatmap_train_data(beatmap_key, weights: ScoreModelWeight,
                           config: NetworkConfig, connection, epoch):
    """
    get the training data (scores) for a beatmap from database
    :param beatmap_key: the target beatmap, in the format of beatmap_id
    :param weights: model weights
    :param config: training config
    :param connection: database connection
    :param epoch: training epoch
    :return: (scores, user_id_array, mod_id_array, weights)
        - scores: shape = [N, ]
        - user_id_array: shape = [N, ], the embedding id of weights.user_embedding
        - mod_id_array: shape = [N, ], the embedding id of weights.mod_embedding
        - weights: shape = [N, ], training weight for each data sample.
    """
    sql = (
        f"SELECT s.user_id, s.cs, s.speed, s.score, s.{Score.CUSTOM_ACCURACY} "
        f"FROM {Score.TABLE_NAME} as s "
        f"WHERE s.{Score.BEATMAP_ID} == {beatmap_key} "
        f"AND s.{Score.GAME_MODE} == '{config.game_mode}' "
        f"AND s.{Score.PP} >= 1 "
        f"AND NOT s.{Score.IS_EZ}")
    scores = []
    mod_emb_id = []
    user_emb_id = []
    for x in repository.execute_sql(connection, sql):
        score = x[3] * 2 if x[2] == -1 else x[3]
        if score < osu_utils.min_score:
            continue
        acc = x[-1]
        if acc < osu_utils.min_acc:
            continue
        user_key = f"{x[0]}-{config.game_mode}-{x[1]}k"
        if user_key not in weights.user_embedding.key_to_embed_id:
            continue

        # scores
        user_emb_id.append(weights.user_embedding.key_to_embed_id[user_key])
        mod_emb_id.append(weights.mod_embedding.key_to_embed_id[speed_to_mod_map[x[2] + 1]])
        scores.append(
            osu_utils.map_osu_score(score, real_to_train=True, arctanh=math.atanh, tanh=math.tanh))

        # accs
        user_emb_id.append(weights.user_embedding.key_to_embed_id[user_key])
        mod_emb_id.append(
            weights.mod_embedding.key_to_embed_id[speed_to_mod_map[x[2] + 1] + "-ACC"])
        scores.append(
            osu_utils.map_osu_acc(acc, real_to_train=True, arctanh=math.atanh, tanh=math.tanh))

    mod_emb_id = np.asarray(mod_emb_id, dtype=np.int32)

    def get_weight(mod_embedding_int):
        count = np.sum(mod_emb_id == mod_embedding_int)
        if count <= 0:
            return 0
        else:
            return np.clip(len(mod_emb_id) / count / 2, 1, 10)

    sample_weights = np.asarray([
        get_weight(i) for i in weights.mod_embedding.key_to_embed_id.values()
    ])[mod_emb_id]

    if len(scores) < config.embedding_size:
        return None
    return (np.asarray(scores, dtype=np.float32),
            np.asarray(user_emb_id, dtype=np.int32),
            mod_emb_id,
            sample_weights)


def get_mod_train_data(mod_key, weights: ScoreModelWeight,
                       config: NetworkConfig, connection, epoch):
    """
    get the training data (scores) for a mod from database.
    WARNING: this may
    :param mod_key: the target mod. For example, DT.
    :param weights: model weights
    :param config: training config
    :param connection: database connection
    :param epoch: training epoch
    :return: (scores, user_id_array, mod_id_array, weights)
        - scores: shape = [N, ]
        - user_id_array: shape = [N, ], the embedding id of weights.user_embedding
        - mod_id_array: shape = [N, ], the embedding id of weights.mod_embedding
        - weights: shape = [N, ], training weight for each data sample.
    """
    speed = 0
    if mod_key.startswith("HT"):
        speed = -1
    elif mod_key.startswith("DT"):
        speed = 1
    is_acc = False
    if mod_key.endswith("-ACC"):
        is_acc = True
    sql = (
        f"SELECT s.beatmap_id, s.user_id, s.cs, s.{Score.SCORE}, s.{Score.CUSTOM_ACCURACY} "
        f"FROM {Score.TABLE_NAME} as s "
        f"WHERE s.{Score.SPEED} == {speed} "
        f"AND s.{Score.GAME_MODE} == '{config.game_mode}' "
        f"AND s.{Score.PP} >= 1 "
        f"AND NOT s.{Score.IS_EZ} ")
    sql += f"AND s.{Score.SCORE_ID} % 30 == {epoch}"
    scores = []
    beatmap_emb_id = []
    user_emb_id = []
    for x in repository.execute_sql(connection, sql):
        beatmap_id, user_id, cs, score, acc = x
        if speed == -1:
            score = score * 2
        if score < osu_utils.min_score or acc < osu_utils.min_acc:
            continue
        user_key = f"{user_id}-{config.game_mode}-{cs}k"
        if user_key not in weights.user_embedding.key_to_embed_id:
            continue
        beatmap_key = str(beatmap_id)

        user_emb_id.append(weights.user_embedding.key_to_embed_id[user_key])
        beatmap_emb_id.append(weights.beatmap_embedding.key_to_embed_id[beatmap_key])

        if is_acc:
            scores.append(
                osu_utils.map_osu_acc(acc, real_to_train=True, arctanh=math.atanh, tanh=math.tanh))
        else:
            scores.append(osu_utils.map_osu_score(score, real_to_train=True, arctanh=math.atanh,
                                                  tanh=math.tanh))

    if len(scores) < config.embedding_size:
        return None
    return (np.asarray(scores, dtype=np.float32),
            np.asarray(user_emb_id, dtype=np.int32),
            np.asarray(beatmap_emb_id, dtype=np.int32),
            None)


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
    return sum(arr) / len(arr)


def train_embedding(key, get_data_method, weights: ScoreModelWeight, config, connection, epoch,
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
    :param key: target key
    :param get_data_method: the method to get training data for key
    :param weights: ScoreModelWeight
    :param config: training config
    :param connection: database connection
    :param epoch: training epoch
    :param embedding_data: an EmbeddingData to save the embedding for key
    :param training_statistics: a TrainingStatistics to save the statistics
    :param pbar: progress bar
    :param other_embedding_data: an EmbeddingData oppository to embedding_data
    :param other_embedding_data2: another EmbeddingData oppository to embedding_data
    :param cachable: can the data returned by get_data_method be cached to save time
    :return: nothing
    """
    time_io = time.time()
    global cache
    if key in cache:
        data = cache[key]
    else:
        data = get_data_method(key, weights, config, connection, epoch)
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
    embedding_data.sigma[emb_id] = sigma
    embedding_data.alpha[emb_id] = alpha

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


def train_personal_embedding(key, get_data_method, weights, config, connection,
                             epoch, embedding_data: EmbeddingData,
                             other_embedding_data: EmbeddingData,
                             other_embedding_data2: EmbeddingData):
    data = get_data_method(key, weights, config, connection, epoch)

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
    user_key = key

    weights = data_process.load_weight_online(config, user_key, connection)
    weights.beatmap_embedding.key_to_embed_id = weights.beatmap_embedding.key_to_embed_id.to_dict()
    weights.user_embedding.key_to_embed_id = weights.user_embedding.key_to_embed_id.to_dict()
    weights.mod_embedding.key_to_embed_id = weights.mod_embedding.key_to_embed_id.to_dict()

    # train the user embedding
    count = train_personal_embedding(user_key, get_user_train_data, weights, config, connection, 0,
                                     weights.user_embedding, weights.beatmap_embedding,
                                     weights.mod_embedding)

    # update neighbors for pass probability estimation
    constraint = UserEmbedding.construct_where_with_key(user_key)
    embedding = weights.user_embedding.embeddings[0][weights.user_embedding.key_to_embed_id[user_key]]
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
        data_process.save_embedding(connection, weights.user_embedding, config,
                                    UserEmbedding.TABLE_NAME,
                                    UserEmbedding.EMBEDDING)
        repository.update(connection, UserEmbedding.TABLE_NAME, puts=[{
            UserEmbedding.COUNT: count,
            UserEmbedding.NEIGHBOR_ID: repository.np_to_db(neighbor_id),
            UserEmbedding.NEIGHBOR_DISTANCE: repository.np_to_db(neighbor_distance),
        }], wheres=[constraint])


def train_score_by_als(config: NetworkConfig):
    connection = repository.get_connection()

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

    for epoch in range(200):

        # train beatmap
        statistics = TrainingStatistics("beatmap", len(weights.beatmap_embedding.key_to_embed_id))
        pbar = tqdm(weights.beatmap_embedding.key_to_embed_id.keys(), desc=statistics.desc)
        for beatmap_key in pbar:
            train_embedding(beatmap_key, get_beatmap_train_data, weights, config, connection, epoch,
                            weights.beatmap_embedding, statistics, pbar, weights.user_embedding,
                            weights.mod_embedding)
        print(pbar.desc)

        # train user
        statistics = TrainingStatistics("user", len(weights.user_embedding.key_to_embed_id))
        pbar = tqdm(weights.user_embedding.key_to_embed_id.keys(), desc=statistics.desc)
        for user_key in pbar:
            train_embedding(user_key, get_user_train_data, weights, config, connection, epoch,
                            weights.user_embedding, statistics, pbar, weights.beatmap_embedding,
                            weights.mod_embedding)
        print(pbar.desc)

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
        # test_score.test_predict_with_sql()

        # train mod
        if epoch >= 2 and epoch <= 15:
            statistics = TrainingStatistics("mod", len(weights.mod_embedding.key_to_embed_id))
            pbar = tqdm(weights.mod_embedding.key_to_embed_id.keys(), desc=statistics.desc)
            for mod_key in pbar:
                train_embedding(mod_key, get_mod_train_data, weights, config, connection, epoch,
                                weights.mod_embedding, statistics, pbar, weights.user_embedding,
                                weights.beatmap_embedding, cachable=False)
            print(pbar.desc)

            def cos_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) + 1e-6) / (np.linalg.norm(b) + 1e-6)

            for idx_i, i in enumerate(weights.mod_embedding.key_to_embed_id.keys()):
                for idx_j, j in enumerate(weights.mod_embedding.key_to_embed_id.keys()):
                    if idx_i <= idx_j:
                        continue
                    ii = weights.mod_embedding.key_to_embed_id[i]
                    jj = weights.mod_embedding.key_to_embed_id[j]
                    sim = cos_sim(weights.mod_embedding.embeddings[0][ii],
                                  weights.mod_embedding.embeddings[0][jj])
                    print(f"{i} <-> {j}: {sim}  ", end="")
                print()


def update_score_count(conn):
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
    repository.execute_sql(conn,
                           "UPDATE UserEmbedding SET count = ("
                           "SELECT COUNT(1) FROM Score "
                           "WHERE Score.user_id == UserEmbedding.id "
                           "AND Score.game_mode == UserEmbedding.game_mode "
                           "AND (Score.cs || 'k') == UserEmbedding.variant)")
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
        config = NetworkConfig(json.load(open(os.path.join(sys.argv[1], "config.json"))))
    else:
        config = NetworkConfig()
    train_score_by_als(config)
    # estimate_var_param(config)
