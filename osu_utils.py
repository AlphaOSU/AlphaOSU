from sklearn.linear_model import LinearRegression

from data.model import *
from data import api

import subprocess
import json


def estimate_star_from_acc(pp, custom_acc, objects):
    acc_pp = max(5 * custom_acc - 4, 0)
    if acc_pp == 0:
        return 0
    length_pp = 1 + 0.1 * min(1.0, objects / 1500)
    diff_pp = pp / acc_pp / length_pp / 8.0
    star = diff_pp ** (1 / 2.2) + 0.15
    return star


def invoke_osu_tools(beatmap_path, dt_star=False, ht_star=False):
    cmd = list(api.get_secret_value("osu_tools_command", []))
    assert len(cmd) > 0
    cmd.extend(["difficulty", beatmap_path, "-j"])
    if dt_star:
        cmd.extend(["-m", "DT"])
    elif ht_star:
        cmd.extend(["-m", "HT"])
    print(f"Invoke osu tools: {cmd}")
    result = json.loads(subprocess.check_output(cmd))
    if dt_star or ht_star:
        return result['results'][0]['attributes']['star_rating']
    return result


def estimate_star_from_score(pp, score, objects, od):
    l = 1 + 0.1 * min(1500, objects) / 1500
    if score < 500000:
        return -1
    elif score < 600000:
        s = (score - 500000) / 100000 * 0.3
    elif score < 700000:
        s = (score - 600000) / 100000 * 0.25 + 0.3
    elif score < 800000:
        s = (score - 700000) / 100000 * 0.2 + 0.55
    elif score < 900000:
        s = (score - 800000) / 100000 * 0.15 + 0.75
    else:
        s = (score - 900000) / 100000 * 0.1 + 0.9
    a = 0
    if od != 0 and score > 960000:
        a = (0.2 - 3 * (10 - od) * 0.006667) * ((score - 960000) / 40000) ** 1.1
    # print(a, s, l, pp, score, objects, od)
    diff_pp = pp / 0.8 / (((l * s) ** 1.1 + (l * s * a) ** 1.1) ** (1 / 1.1))
    return ((diff_pp * 135) ** (1 / 2.2) + 4) / 5 * 0.2


# @measure_time
def mania_pp(score, od, star, objects, power=np.power, where=np.where, abs=np.abs):
    d = power((25 * star - 4), 2.2) / 135
    d = where(star < 0.2, 1 / 135, d)
    l = 1 + 0.1 * where(1500.0 < objects, 1500.0, objects) / 1500

    s5 = 0.0
    s6 = (score - 500000) / 100000 * 0.3
    s7 = (score - 600000) / 100000 * 0.25 + 0.3
    s8 = (score - 700000) / 100000 * 0.2 + 0.55
    s9 = (score - 800000) / 100000 * 0.15 + 0.75
    s10 = (score - 900000) / 100000 * 0.1 + 0.9
    s = where(score - 500000 < 0, s5,
              where(score - 600000 < 0, s6,
                    where(score - 700000 < 0, s7,
                          where(score - 800000 < 0, s8,
                                where(score - 900000 < 0, s9, s10)))))
    base_pp = d * l * s

    a = (0.2 - 3 * (10 - od) * 0.006667)
    a = a * power(abs(score - 960000) / 40000, 1.1)
    a = a * base_pp
    accuracy_pp = where(score > 960000, a, 0)

    return 0.8 * power(power(base_pp, 1.1) + power(accuracy_pp, 1.1), 1 / 1.1)


def mania_pp_v4(custom_acc, star, objects, power=np.power, max=np.maximum, min=np.minimum):
    diff_pp = power(max(star - 0.15, 0.05), 2.2)
    acc_pp = max(5 * custom_acc - 4, 0)
    length_pp = 1 + 0.1 * min(1.0, objects / 1500)
    return 8 * diff_pp * acc_pp * length_pp


ep = 0.2

max_score = 1_000_000
min_score = 500_000
max_acc = 1
min_acc = 0.8

k_score = (2 - 2 * ep) / (max_score - min_score)
b_score = -k_score * min_score - 1 + ep

k_acc = (2 - 2 * ep) / (max_acc - min_acc)
b_acc = -k_acc * min_acc - 1 + ep


def map_osu_score(score, real_to_train: bool, arctanh=np.arctanh, tanh=np.tanh):
    global k_score, b_score, min_score, max_score
    if real_to_train:
        return arctanh(k_score * np.clip(score, min_score, max_score) + b_score)
    else:
        return np.clip((tanh(score) - b_score) / k_score, min_score, max_score)


def map_osu_acc(acc, real_to_train: bool, arctanh=np.arctanh, tanh=np.tanh):
    global k_acc, b_acc, min_acc, max_acc
    if real_to_train:
        return arctanh(k_acc * np.clip(acc, min_acc, max_acc) + b_acc)
    else:
        return np.clip((tanh(acc) - b_acc) / k_acc, min_acc, max_acc)


def predict_score_std(connection, uid, variant, config: NetworkConfig, bid, mod):
    x = repository.select(connection, [UserEmbedding.TABLE_NAME, BeatmapEmbedding.TABLE_NAME,
                                       ModEmbedding.TABLE_NAME],
                          project=[
                                      UserEmbedding.TABLE_NAME + "." + UserEmbedding.EMBEDDING + "_alpha",
                                      UserEmbedding.TABLE_NAME + "." + UserEmbedding.EMBEDDING + "_sigma",
                                      BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.ITEM_EMBEDDING + "_alpha",
                                      BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.ITEM_EMBEDDING + "_sigma",
                                      ModEmbedding.TABLE_NAME + "." + ModEmbedding.EMBEDDING + "_alpha",
                                      ModEmbedding.TABLE_NAME + "." + ModEmbedding.EMBEDDING + "_sigma"] +
                                  [
                                      UserEmbedding.TABLE_NAME + "." + UserEmbedding.EMBEDDING + "_" + str(
                                          i) for i in
                                      range(config.embedding_size)] +
                                  [
                                      BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.ITEM_EMBEDDING + "_" + str(
                                          i) for i in
                                      range(config.embedding_size)] +
                                  [
                                      ModEmbedding.TABLE_NAME + "." + ModEmbedding.EMBEDDING + "_" + str(
                                          i) for i in
                                      range(config.embedding_size)],
                          where={
                              UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: uid,
                              UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT: variant,
                              UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: config.game_mode,
                              BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.BEATMAP_ID: bid,
                              ModEmbedding.TABLE_NAME + "." + ModEmbedding.MOD: mod
                          }, return_first=True)
    if x is None:
        return None
    user_alpha, user_sigma, beatmap_alpha, beatmap_sigma, mod_alpha, mod_sigma = x[:6]
    if user_alpha == 0 or beatmap_alpha == 0 or mod_alpha == 0:
        return None
    user_sigma = repository.db_to_np(user_sigma)
    beatmap_sigma = repository.db_to_np(beatmap_sigma)
    mod_sigma = repository.db_to_np(mod_sigma)

    user_embedding = np.asarray(x[6: 6 + config.embedding_size])
    beatmap_embedding = np.asarray(x[6 + config.embedding_size: 6 + config.embedding_size * 2])
    mod_embedding = np.asarray(x[6 + config.embedding_size * 2: 6 + config.embedding_size * 3])

    x = np.dot(user_embedding, beatmap_embedding * mod_embedding)

    user_mod_embedding = user_embedding * mod_embedding
    var_beatmap = (np.dot(user_mod_embedding, beatmap_sigma) * user_mod_embedding).sum() + (
            1.0 / beatmap_alpha)

    beatmap_mod_embedding = beatmap_embedding * mod_embedding
    var_user = (np.dot(beatmap_mod_embedding, user_sigma) * beatmap_mod_embedding).sum() + (
            1.0 / user_alpha)

    user_beatmap_embedding = user_embedding * beatmap_embedding
    var_mod = (np.dot(user_beatmap_embedding, mod_sigma) * user_beatmap_embedding).sum() + (
            1.0 / mod_alpha)

    std = 0.06018866012651341 * var_beatmap + 0.29444196224551056 * var_user + 0.23866568666941756 * var_mod
    std = np.sqrt(std)
    return x, std


def predict_score(connection, where_dict, embedding_size, projection, limit=None):
    project_templates = []
    for i in range(embedding_size):
        project_templates.append(
            f"{BeatmapEmbedding.TABLE_NAME}.{BeatmapEmbedding.ITEM_EMBEDDING}_{i} * "
            f"{UserEmbedding.TABLE_NAME}.{UserEmbedding.EMBEDDING}_{i} * "
            f"{ModEmbedding.TABLE_NAME}.{ModEmbedding.EMBEDDING}_{i}"
        )

    where_items = list(where_dict.items())
    sql = """SELECT {score} AS score, {projection} 
FROM BeatmapEmbedding, UserEmbedding, ModEmbedding, Beatmap
WHERE {wheres} AND Beatmap.id == BeatmapEmbedding.id
""".format(
        score=" + ".join(project_templates), projection=', '.join(projection),
        wheres=' AND '.join(map(lambda x: f"{x[0]} {x[1][0]} ?", where_items))
    )
    if limit is not None:
        sql += " LIMIT " + str(limit)

    return repository.execute_sql(connection, sql,
                                  parameters=list(
                                      map(lambda x: x[1][1], where_items))).fetchall()


@measure_time
def get_user_bp(connection, user_id,
                config: NetworkConfig, max_length=100,
                is_acc=False):  # return (bid, speed) -> (score, pp)
    if is_acc:
        score = Score.CUSTOM_ACCURACY
    else:
        score = Score.SCORE
    sql = f"""SELECT Beatmap.id, Score.speed, Score.PP, Score.{score}, Score.{Score.SCORE_ID}, Beatmap.{Beatmap.HT_STAR}, 
Beatmap.{Beatmap.STAR}, Beatmap.{Beatmap.DT_STAR}, Beatmap.{Beatmap.CS}
FROM Score
JOIN Beatmap ON Score.beatmap_id == Beatmap.id
WHERE Score.user_id == "{user_id}" 
AND Beatmap.game_mode == "{config.game_mode}"
ORDER BY Score.pp DESC
"""
    if max_length is not None:
        sql += f"LIMIT {max_length + 30}"
    cursor = list(repository.execute_sql(connection, sql).fetchall())
    user_bp = BestPerformance(max_length)
    for tuple in cursor[::-1]:
        bid, speed, pp, score, score_id, ht_star, nm_star, dt_star, cs = tuple
        star = [ht_star, nm_star, dt_star][speed + 1]
        user_bp.update(int(bid), int(speed), score, pp, star, score_id=score_id,
                       cs=cs)  # , embeddings)
    return user_bp


def estimate_user_embedding(user_bp: BestPerformance):
    embeddings = np.asarray(user_bp.data['embedding'].tolist())  # [L, E]
    scores = map_osu_score(user_bp.data['score'].to_numpy().astype(np.float32),
                           real_to_train=True)  # [L]
    pps = user_bp.data['pp'].to_numpy()

    # y = kx + b
    y = scores - embeddings[:, -1]
    x = embeddings[:, :-1]
    regr = LinearRegression()
    regr.fit(x, y, pps / np.mean(pps))
    emb = regr.coef_.tolist()
    emb.append(float(regr.intercept_))
    return emb


if __name__ == "__main__":
    # with repository.get_connection() as conn:
    #     bp = get_user_bp(conn, uid, NetworkConfig())
    #     print(estimate_user_embedding(bp))
    # print(mania_pp(898231, 9.0, 7.00, 2404 + 57))
    # print(mania_pp(986114, 8.2, 6.40, 2259 + 1349))
    #
    # print(mania_pp(816323, 9, 8.289408218280933, 257 + 1671))
    # print(estimate_star_from_score(603.614, 816323, 257 + 1671, 9))

    # print(mania_pp(1_000_000, 8.0, 5.0, 10000))
    # print(mania_pp(500_000, 8.0, 5.0, 10000))
    # print(mania_pp(960_000, 8.0, 5.0, 10000))
    #
    # print(mania_pp_v4(1, 5.0, 10000))
    # print(mania_pp_v4(0.9, 5.0, 10000))
    # print(mania_pp_v4(0.8, 5.0, 10000))
    # print(map_osu_score(850000, real_to_train=True))
    # print(map_osu_score(960000, real_to_train=True))
    # print(map_osu_score(990000, real_to_train=True))
    # print(map_osu_score(999000, real_to_train=True))
    # print(map_osu_score(1000000, real_to_train=True))
    #
    # print(map_osu_score(1, real_to_train=False))

    print(estimate_star_from_acc(505.91, 0.945932922127987, 10000))
