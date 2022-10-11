from sklearn.neighbors import NearestNeighbors

from data import data_process
from data.model import *


def construct_nearest_neighbor(config: NetworkConfig):
    connection = repository.get_connection()
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
    nbrs = NearestNeighbors(n_neighbors=150, n_jobs=-1).fit(user_embs)
    nbrs_distance, nbrs_index = nbrs.kneighbors(user_embs)

    # save into database
    repository.ensure_column(connection, UserEmbedding.TABLE_NAME, [
        (UserEmbedding.NEIGHBOR_ID, "TEXT", None),
        (UserEmbedding.NEIGHBOR_DISTANCE, "TEXT", None),
    ])
    puts = []
    wheres = []
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


def estimate_pass_probability(uid, variant, beatmap_ids, speed, config: NetworkConfig, connection):
    cur_nbrs_ids, cur_nbrs_distance = repository.select(connection, UserEmbedding.TABLE_NAME,
                                                        [UserEmbedding.NEIGHBOR_ID, UserEmbedding.NEIGHBOR_DISTANCE],
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
        cursor = connection.execute(f"SELECT {Score.BEATMAP_ID}, {Score.USER_ID} "
                              "FROM Score "
                              f"WHERE Score.user_id IN ({','.join(map(str, cur_nbrs_ids))}) "
                              f"AND Score.{Score.GAME_MODE} == ? "
                              f"AND Score.{Score.CS} == ? "
                              f"AND Score.SPEED == ? ",
                              [config.game_mode, variant[0], speed])
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

    return scores


if __name__ == "__main__":
    construct_nearest_neighbor(NetworkConfig())
    print(estimate_pass_probability(7304075, '4k', [
        767046,  # triumph
        3525702,  # eternel
        1920615,  # blue
        992512, # galaxy
    ], 1, NetworkConfig()))

    print(estimate_pass_probability(10500832, '4k', [
        767046,  # triumph
        3525702,  # eternel
        1920615,  # blue
        992512, # galaxy
    ], 1, NetworkConfig()))

    print(estimate_pass_probability(8611484, '4k', [
        767046,  # triumph
        3525702,  # eternel
        1920615,  # blue
        992512, # galaxy
    ], 1, NetworkConfig()))

    print(estimate_pass_probability(26407244, '4k', [
        767046,  # triumph
        3525702,  # eternel
        1920615,  # blue
        992512, # galaxy
    ], 1, NetworkConfig()))

