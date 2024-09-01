from sklearn.linear_model import LinearRegression

from data.model import *
from data import api

import subprocess
import json


def invoke_osu_tools(beatmap_path, dt_star=False, ht_star=False, nm_star=False):
    """
    Invoke osu tools for sr calculation.
    See https://github.com/ppy/osu-tools/ for more details.
    @param beatmap_path: beatmap_path
    @param dt_star: if return DT sr only
    @param ht_star: if return HT sr only
    @param nm_star: if return NM sr only
    @return: result
    """
    cmd = list(api.get_secret_value("osu_tools_command", []))
    assert len(cmd) > 0
    cmd.extend(["difficulty", beatmap_path, "-j"])
    if dt_star:
        cmd.extend(["-m", "DT"])
    elif ht_star:
        cmd.extend(["-m", "HT"])
    result = json.loads(subprocess.check_output(cmd))
    if dt_star or ht_star or nm_star:
        return result['results'][0]['attributes']['star_rating']
    return result


def mania_pp(score, od, star, objects, power=np.power, where=np.where, abs=np.abs):
    """
    @deprecated this is old mania PP calculation. New version is mania_pp_v4().
    """
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
    """
    Calculate mania PP.
    @param custom_acc: PP acc
    @param star: sr
    @param objects: number of objects ( = rice notes + LN notes)
    @param power: power function
    @param max: max function
    @param min: min function
    @return: PP
    """
    diff_pp = power(max(star - 0.15, 0.05), 2.2)
    acc_pp = max(5 * custom_acc - 4, 0)
    length_pp = 1 + 0.1 * min(1.0, objects / 1500)
    return 8 * diff_pp * acc_pp * length_pp


ep = 0.2

max_score = 1_000_000
min_score = 500_000
max_acc = 1
min_acc = 0.8
max_pp_ratio = 1
min_pp_ratio = 0

k_score = (2 - 2 * ep) / (max_score - min_score)
b_score = -k_score * min_score - 1 + ep

k_acc = (2 - 2 * ep) / (max_acc - min_acc)
b_acc = -k_acc * min_acc - 1 + ep

k_pp = (2 - 2 * ep) / (max_pp_ratio - min_pp_ratio)
b_pp = -k_pp * min_pp_ratio - 1 + ep

def map_osu_score(score, real_to_train: bool, arctanh=np.arctanh, tanh=np.tanh):
    """
    Scale scores to [-1, 1] for stable training.
    @param score: if real_to_train, this is real osu score, otherwise the scores for training.
    @param real_to_train: if true, perform real -> training, otherwise training -> real
    @param arctanh: arctanh function
    @param tanh: tanh function
    @return: if real_to_train, return scores for training, else return real osu scores.
    """
    global k_score, b_score, min_score, max_score
    if real_to_train:
        return arctanh(k_score * np.clip(score, min_score, max_score) + b_score)
    else:
        return np.clip((tanh(score) - b_score) / k_score, min_score, max_score)


def map_osu_pp(pp, real_to_train: bool, max_pp: float, arctanh=np.arctanh, tanh=np.tanh):
    global k_pp, b_pp, min_pp_ratio, max_pp_ratio
    if real_to_train:
        return arctanh(k_pp * np.clip(pp / max_pp, min_pp_ratio, max_pp_ratio) + b_pp)
    else:
        return np.clip((tanh(pp) - b_pp) / k_pp, min_pp_ratio, max_pp_ratio) * max_pp

def map_osu_acc(acc, real_to_train: bool, arctanh=np.arctanh, tanh=np.tanh):
    """
    Scale acc to [-1, 1] for stable training.
    @param acc: if real_to_train, this is real osu acc, otherwise the acc for training.
    @param real_to_train: if true, perform real -> training, otherwise training -> real
    @param arctanh: arctanh function
    @param tanh: tanh function
    @return: if real_to_train, return acc for training, else return real osu acc.
    """
    global k_acc, b_acc, min_acc, max_acc
    if real_to_train:
        return arctanh(k_acc * np.clip(acc, min_acc, max_acc) + b_acc)
    else:
        return np.clip((tanh(acc) - b_acc) / k_acc, min_acc, max_acc)


MOD_INT_MAPPING = {
    'NM': 0,
    'DT': 64,
    'HD': 8,
    'HR': 16,
    'HT': 256
}

def mods_to_db_key(mods):
    if mods is None or len(mods) == 0:
        return "0"
    return str(sum(MOD_INT_MAPPING.get(m, 0) for m in mods))


def bools_to_db_key(is_dt, is_hr, is_hd):
    mod_int = 0
    if is_dt:
        mod_int += MOD_INT_MAPPING["DT"]
    if is_hr:
        mod_int += MOD_INT_MAPPING["HR"]
    if is_hd:
        mod_int += MOD_INT_MAPPING["HD"]
    return str(mod_int)

STD_MODS = dict(
    (mods_to_db_key(x), x) for x in
    [['NM'], ['HD'], ['HR'], ['DT'], ['HD', 'HR'], ['HD', 'DT'], ['HR', 'DT'], ['HR', 'HD', 'DT']]
)

def predict_score_std(connection, uid, variant, config: NetworkConfig, bid, mod):
    """
    Predict score with standard deviation using Beyasian Linear Regression.
    @param connection: db connection
    @param uid: user id
    @param variant: 4k/7k
    @param config: configuration
    @param bid: beatmap id
    @param mod: mod text, HT/DT/NM/...
    @return: score and std.
    """
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

    # These magic numbers are estimated in train_score_als_db.estimate_var_param()
    std = 0.06018866012651341 * var_beatmap + 0.29444196224551056 * var_user + 0.23866568666941756 * var_mod
    std = np.sqrt(std)
    return x, std


def predict_score(connection, where_dict, embedding_size, projection, limit=None):
    """
    Predict scores Using Linear Regression. This function is faster than predict_score_std*()
    @param connection: db connection
    @param where_dict: db query constraints
    @param embedding_size: embedding size
    @param projection: db query column projection
    @param limit: db query result limits.
    @return: scores
    """
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
                is_acc=False) -> BestPerformance:
    """
    Get user's best performance.
    @param connection: db connection
    @param user_id: user id
    @param config: configuration
    @param max_length: bp max length
    @param is_acc: fetch score or acc
    @return: BestPerformance
    """
    if config.game_mode == 'mania':
        if is_acc:
            score = Score.CUSTOM_ACCURACY
        else:
            score = Score.SCORE
        sql = (f"SELECT Beatmap.id, Score.speed, Score.PP, Score.{score}, Score.{Score.SCORE_ID}, Beatmap.{Beatmap.HT_STAR}, "
                f"Beatmap.{Beatmap.STAR}, Beatmap.{Beatmap.DT_STAR}, Beatmap.{Beatmap.CS} "
                f"FROM Score "
                f"JOIN Beatmap ON Score.beatmap_id == Beatmap.id "
                f'WHERE Score.user_id == "{user_id}" ' 
                f'AND Beatmap.game_mode == "{config.game_mode}" '
                f"ORDER BY Score.pp DESC ")
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
    elif config.game_mode == 'osu':
        sql = (f"SELECT Beatmap.id, Score.PP, Score.{Score.SCORE_ID}, Beatmap.{Beatmap.MOD_STAR}, "
                f"Score.{Score.IS_DT}, Score.{Score.IS_HD}, Score.{Score.IS_HR}, "
                f"Score.{Score.SCORE} "
                f"FROM Score "
                f"JOIN Beatmap ON Score.beatmap_id == Beatmap.id "
                f'WHERE Score.user_id == "{user_id}" ' 
                f'AND Beatmap.game_mode == "{config.game_mode}" '
                f"ORDER BY Score.pp DESC ")
        if max_length is not None:
            sql += f"LIMIT {max_length + 30}"
        cursor = list(repository.execute_sql(connection, sql).fetchall())
        user_bp = BestPerformance(max_length)
        for tuple in cursor[::-1]:
            bid, pp, score_id, mod_star, is_dt, is_hd, is_hr, score = tuple
            mod_int = bools_to_db_key(is_dt, is_hr, is_hd)
            star = json.loads(mod_star)[mod_int]
            user_bp.update(int(bid), mod_int, score, pp, star, score_id=score_id)  # , embeddings)
        return user_bp
    else:
        raise NotImplemented