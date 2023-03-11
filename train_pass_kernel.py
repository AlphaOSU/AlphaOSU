from sklearn.neighbors import NearestNeighbors

from data import data_process
from data.model import *


def construct_nearest_neighbor(config: NetworkConfig):
    """
    We estimate the pass probability using k-nearest neighbors. This function constructs Top-k
    nearest neighbors for each player in the database, which are saved in UserEmbedding.
    Neighbors in UserEmbedding.
    @param config: configuration
    @return: None
    """
    connection = repository.get_connection()
    print("Prepare weights")
    weights = data_process.load_weight(config)
    key_to_embed_id: dict = weights.user_embedding.key_to_embed_id.to_dict()
    user_embs = weights.user_embedding.embeddings[0]
    user_ids = np.zeros((len(user_embs),), dtype=np.int32)

    first_variant = None
    for key, emb_id in key_to_embed_id.items():
        e = key.split("-")
        v = e[-1]
        user_ids[emb_id] = int(e[0])
        if first_variant is None:
            first_variant = v
        # trick: add a large value to the first dimension if variant is different
        if v != first_variant:
            user_embs[emb_id, 0] += 50

    # find NearestNeighbors
    print("Fitting")
    nbrs = NearestNeighbors(n_neighbors=150, n_jobs=-1).fit(user_embs)
    nbrs_distance, nbrs_index = nbrs.kneighbors(user_embs)

    # save into database
    repository.ensure_column(connection, UserEmbedding.TABLE_NAME, [
        (UserEmbedding.NEIGHBOR_ID, "TEXT", None),
        (UserEmbedding.NEIGHBOR_DISTANCE, "TEXT", None),
    ])
    puts = []
    wheres = []
    print("Saving")
    for key, emb_id in key_to_embed_id.items():
        e = key.split("-")
        puts.append({
            UserEmbedding.NEIGHBOR_ID: repository.np_to_db(user_ids[nbrs_index[emb_id]]),
            UserEmbedding.NEIGHBOR_DISTANCE: repository.np_to_db(nbrs_distance[emb_id]),
        })
        wheres.append({
            UserEmbedding.USER_ID: e[0],
            UserEmbedding.GAME_MODE: e[1],
            UserEmbedding.VARIANT: e[2]
        })
    with connection:
        repository.update(connection, UserEmbedding.TABLE_NAME, puts, wheres)


def estimate_pass_probability(uid, variant, beatmap_ids, speed, config: NetworkConfig,
                              connection, is_hr=False):
    """
    Using k-nearest neighbors algorithm to estimate the pass probability.
    Should be called after construct_nearest_neighbor()
    @param uid: user id, should be int
    @param variant: mania: 4k / 7k, else: ""
    @param beatmap_ids: beatmap ids that want to calculate
    @param speed: -1/0/1 = HT/NM/DT
    @param config: configuration
    @param connection: db connection
    @param is_hr: HR mod (std only)
    @return: probabilities for each beatmap id.
    """
    cur_nbrs_ids, cur_nbrs_distance = repository.select(connection, UserEmbedding.TABLE_NAME,
                                                        [UserEmbedding.NEIGHBOR_ID,
                                                         UserEmbedding.NEIGHBOR_DISTANCE],
                                                        where={
                                                            UserEmbedding.USER_ID: uid,
                                                            UserEmbedding.VARIANT: variant,
                                                            UserEmbedding.GAME_MODE: config.game_mode
                                                        }).fetchone()
    cur_nbrs_ids = repository.db_to_np(cur_nbrs_ids)
    cur_nbrs_distance = repository.db_to_np(cur_nbrs_distance)
    cur_nbrs_weights = np.exp(-cur_nbrs_distance / (2 * config.pass_band_width ** 2))
    scores = np.zeros(len(beatmap_ids), dtype=np.float32)
    played_flag = np.zeros(len(beatmap_ids), dtype=np.float32)
    beatmap_id_to_score_index = dict([(b, i) for (i, b) in enumerate(beatmap_ids)])
    user_id_to_weight = dict(zip(cur_nbrs_ids, cur_nbrs_weights))

    with connection:
        sql = (f"SELECT {Score.BEATMAP_ID}, {Score.USER_ID} "
               f"FROM Score "
               f"WHERE Score.{Score.USER_ID} IN ({','.join(map(str, cur_nbrs_ids))}) "
               f"AND Score.{Score.GAME_MODE} == ? "
               f"AND ({Score.SPEED} > ? OR ({Score.SPEED} == ? AND {Score.IS_HR} >= ? )) "
               )
        # Compare speed first, then compare HR if speeds are equal.
        sql_params = [config.game_mode, speed, speed, int(is_hr)]

        if config.game_mode == 'mania':
            sql += f"AND Score.{Score.CS} == ? "
            sql_params.append(variant[0])
        cursor = connection.execute(sql, sql_params)
        for test_bid, test_uid in cursor:
            if test_bid not in beatmap_id_to_score_index:
                continue
            test_beatmap_index = beatmap_id_to_score_index[test_bid]
            if test_uid == uid:
                played_flag[test_beatmap_index] = 1
            scores[test_beatmap_index] += user_id_to_weight[test_uid]
    scores /= np.sum(cur_nbrs_weights)
    weight_played = config.pass_basic_weight_played
    scores = np.where(played_flag == 1, scores * (1 - weight_played) + weight_played, scores)
    scores = np.power(scores, config.pass_power)
    # scores = scores * 0.9999 + 0.0001
    return scores


if __name__ == "__main__":
    construct_nearest_neighbor(NetworkConfig({"game_mode": "osu", "embedding_size": 20}))
