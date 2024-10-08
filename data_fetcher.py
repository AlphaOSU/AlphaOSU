import json
import random
import time

import requests
from tqdm import tqdm
from dateutil import parser

import osu_utils
from train_score_als_db import train_personal_embedding_online
from data import api
from data.model import *
import subprocess


def ensure_beatmap_star(beatmap_id, ht_star, dt_star, star):
    """
    @deprecated use ensure_beatmap_attributes instead

    fetch the star of beatmap with HT or DT mod if it hadn't been fetched
    @param beatmap_id: the id of beatmap in osu
    @param ht_star:the star of beatmap with HT mod
    @param dt_star:the star of beatmap with DT mod
    @param star:the star of beatmap
    @return:ht_star and dt_star
    """
    if ht_star != 0 and dt_star != 0 and star != 0:
        return ht_star, dt_star, star
    osu_website = api.get_secret_value("osu_website", api.OSU_WEBSITE)
    r = api.request_api(f"osu/{beatmap_id}", "GET", osu_website, json=False)
    base_cache = os.path.join("result", "cache")
    os.makedirs(base_cache, exist_ok=True)
    beatmap_path = os.path.join(base_cache, f"{beatmap_id}.osu")
    beatmap_path = os.path.abspath(beatmap_path)
    with open(beatmap_path, 'wb') as fd:
        fd.write(r.content)

    if ht_star == 0:
        ht_star = osu_utils.invoke_osu_tools(beatmap_path, ht_star=True)
    if dt_star == 0:
        dt_star = osu_utils.invoke_osu_tools(beatmap_path, dt_star=True)
    if star == 0:
        star = osu_utils.invoke_osu_tools(beatmap_path, nm_star=True)

    os.remove(beatmap_path)
    print(f"Calculate star: {beatmap_id}, ht = {ht_star}, dt = {dt_star}, nm = {star}")
    return ht_star, dt_star, star


def ensure_beatmap_attributes(beatmap_id, mod_stars, mod_max_pp, mode):
    """
    fetch the sr / max pp of beatmap with various mods if it hadn't been fetched
    @param beatmap_id: bid
    @param mod_stars: a dict (mod_int -> sr)
    @param mod_max_pp: a dict (mod_int -> max pp)
    @param mode_int: game mode
    @return: mod_stars, mod_max_pp
    """
    if mode == 'mania':
        raise NotImplemented
    elif mode == 'osu':
        if mod_stars is not None and len(mod_stars) == len(osu_utils.STD_MODS) and mod_max_pp is not None:
            return mod_stars, mod_max_pp

        osu_website = api.get_secret_value("osu_website", api.OSU_WEBSITE)
        r = api.request_api(f"osu/{beatmap_id}", "GET", osu_website, json=False)
        base_cache = os.path.join("result", "cache")
        os.makedirs(base_cache, exist_ok=True)
        beatmap_path = os.path.join(base_cache, f"{beatmap_id}.osu")
        beatmap_path = os.path.abspath(beatmap_path)
        with open(beatmap_path, 'wb') as fd:
            fd.write(r.content)

        cmd = list(api.get_secret_value("osu_tools_command", []))
        assert len(cmd) > 0
        cmd.extend(["simulate", "osu", beatmap_path, "-m", "DT", "-m", "HD", "-m", "HR", "-p"])
        result = json.loads(subprocess.check_output(cmd))

        mod_stars = {}
        mod_max_pp = {}
        for x in result:
            mod_int = osu_utils.mods_to_db_key(list(map(lambda o: o['acronym'], x['score']['mods'])))
            mod_stars[mod_int] = round(x['difficulty_attributes']['star_rating'], 5)
            mod_max_pp[mod_int] = round(x['performance_attributes']['pp'], 5)
        os.remove(beatmap_path)
        return mod_stars, mod_max_pp
    else:
        raise NotImplemented

def parse_beatmap_data(beatmap_data, beatmapset_data, conn: sqlite3.Connection, mode=None,
                       mode_int=None):
    """
    filter useful beatmap data from the original one
    @param beatmap_data:original beatmap data from osu api
    @param beatmapset_data:original beatmapset data from osu api
    @param conn:database connection
    @param mode:osu game mode
    @param mode_int:a code name of osu game mode
    @return:filtered beatmap data as beatmap_db_data
    """
    if beatmap_data['status'] != 'ranked':
        return None
    if beatmap_data['convert']:
        return None
    if mode is not None and beatmap_data['mode'] != mode:
        return None
    if mode_int is not None and beatmap_data['mode_int'] != mode_int:
        return None
    if mode is None and mode_int is not None:
        mode = ['osu', 'taiko', 'fruits', 'mania'][mode_int]
    result = {
        Beatmap.ID: beatmap_data['id'],
        Beatmap.SET_ID: beatmap_data['beatmapset_id'],
        Beatmap.NAME: beatmapset_data['title'],
        Beatmap.VERSION: beatmap_data['version'],
        Beatmap.GAME_MODE: beatmap_data['mode'],
        Beatmap.CREATOR: beatmapset_data['creator'],
        Beatmap.LAST_UPDATED: int(parser.parse(beatmap_data['last_updated']).timestamp()),

        Beatmap.LENGTH: beatmap_data['hit_length'],
        Beatmap.BPM: beatmap_data['bpm'],
        Beatmap.CS: beatmap_data['cs'],
        Beatmap.HP: beatmap_data['drain'],
        Beatmap.OD: beatmap_data['accuracy'],
        Beatmap.AR: beatmap_data['ar'],
        Beatmap.STAR: beatmap_data['difficulty_rating'],
        Beatmap.COUNT_CIRCLES: beatmap_data['count_circles'],
        Beatmap.COUNT_SLIDERS: beatmap_data['count_sliders'],
        Beatmap.COUNT_SPINNERS: beatmap_data['count_spinners'],
        Beatmap.PASS_COUNT: beatmap_data['passcount'],
        Beatmap.PLAY_COUNT: beatmap_data['playcount'],
    }
    if mode == 'mania': # TODO: refactor it with mod_stars and mod_max_pp
        old = repository.select_first(conn, Beatmap.TABLE_NAME,
                                      project=[Beatmap.SUM_SCORES, Beatmap.HT_STAR, Beatmap.DT_STAR, Beatmap.STAR],
                                      where={Beatmap.ID: beatmap_data['id']})
        if old is None:
            old_sum, ht_star, dt_star, star = 0, 0, 0, 0
        else:
            old_sum, ht_star, dt_star, star = old
        try:
            ht_star, dt_star, star = ensure_beatmap_star(beatmap_data['id'], ht_star, dt_star, star)
        except:
            print(f"ensure_beatmap_star error in {beatmap_data['id']}!!!")
        result.update({
            Beatmap.SUM_SCORES: old_sum,
            Beatmap.HT_STAR: ht_star,
            Beatmap.DT_STAR: dt_star,
            Beatmap.STAR: star
        })
    elif mode == 'osu':
        old = repository.select_first(conn, Beatmap.TABLE_NAME,
                                      project=[Beatmap.SUM_SCORES, Beatmap.MOD_STAR,
                                               Beatmap.MOD_MAX_PP],
                                      where={Beatmap.ID: beatmap_data['id']})
        if old is None:
            old_sum, mod_star, mod_max_pp = 0, {}, {}
        else:
            old_sum, mod_star, mod_max_pp = old
            mod_star = json.loads(mod_star) if mod_star is not None else None
            mod_max_pp = json.loads(mod_max_pp) if mod_max_pp is not None else None
        try:
            mod_star, mod_max_pp = ensure_beatmap_attributes(beatmap_data['id'], mod_star,
                                                             mod_max_pp, mode)

        except Exception:
            print(f"ensure_beatmap_attributes error in {beatmap_data['id']}!!!")
            # raise
        result.update({
            Beatmap.SUM_SCORES: old_sum,
            Beatmap.MOD_STAR: json.dumps(mod_star, separators=(',', ':')),
            Beatmap.MOD_MAX_PP: json.dumps(mod_max_pp, separators=(',', ':')),
        })
    else:
        raise NotImplemented


    return result


def parse_score_data(scores, beatmap_id, is_in_top_scores):
    """
    filter useful score data from the original one
    @param scores:related score data from osu api according to user's bp or top score of beatmap
    @param beatmap_id:the id of beatmap in osu
    @param is_in_top_scores:a control parameter
    @return:filtered score data as score_db_data
    """
    is_dt = 'DT' in scores['mods'] or 'NC' in scores['mods']
    is_ht = 'HT' in scores['mods']
    result = {
        Score.BEATMAP_ID: beatmap_id,
        Score.USER_ID: scores['user_id'],
        Score.SCORE_ID: scores['id'],
        Score.SPEED: 1 if is_dt else (-1 if is_ht else 0),
        Score.IS_DT: is_dt,
        Score.IS_HR: 'HR' in scores['mods'],
        Score.IS_HD: 'HD' in scores['mods'],
        Score.IS_FL: 'FL' in scores['mods'],
        Score.IS_EZ: 'EZ' in scores['mods'],
        Score.IS_MR: 'MR' in scores['mods'],
        Score.IS_HT: is_ht,
        Score.CREATE_AT: int(parser.parse(scores['created_at']).timestamp()),

        Score.ACCURACY: scores['accuracy'],
        Score.SCORE: scores['score'],
        Score.MAX_COMBO: scores['max_combo'],
        Score.COUNT_50: scores['statistics']['count_50'],
        Score.COUNT_100: scores['statistics']['count_100'],
        Score.COUNT_300: scores['statistics']['count_300'],
        Score.COUNT_geki: scores['statistics']['count_geki'],
        Score.COUNT_katu: scores['statistics']['count_katu'],
        Score.COUNT_miss: scores['statistics']['count_miss'],
        Score.PP: scores.get('pp', 0),
        Score.PP_WEIGHT: scores.get('weight', {}).get('percentage', 0),
    }
    if result[Score.PP] is None:
        result[Score.PP] = 0
    return result


def get_state(key, default):
    with repository.get_connection() as conn_progress:
        count = repository.count(conn_progress, Task.TABLE_NAME, where={Task.TASK_NAME: key})
        if count == 0:
            return default
        progress, t = repository.select_first(conn_progress, Task.TABLE_NAME,
                                              project=[Task.TASK_STATE, Task.TASK_TIME],
                                              where={Task.TASK_NAME: key})
        if t + 2 * 24 * 3600 >= time.time():
            return progress
        return default


def update_progress(connection: sqlite3.Connection, key, progress):
    repository.insert_or_replace(connection, Task.TABLE_NAME, [{
        Task.TASK_NAME: key,
        Task.TASK_STATE: progress,
        Task.TASK_TIME: time.time()
    }])


class ProgressControl:
    def __init__(self, key, total):
        self._key = key
        self._st = None
        self._st_count = 0
        # self._bar = tqdm.tqdm(desc=key)

    def get_state(self, default):
        with repository.get_connection() as conn_progress:
            count = repository.count(conn_progress, Task.TABLE_NAME,
                                     where={Task.TASK_NAME: self._key})
        if count == 0:
            return default
        progress, t = repository.select_first(conn_progress, Task.TABLE_NAME,
                                              project=[Task.TASK_STATE, Task.TASK_TIME],
                                              where={Task.TASK_NAME: self._key})
        if t + 2 * 24 * 3600 >= time.time():
            return progress
        return default

    def commit(self, state: str, current_progress: int, total: int, connection: sqlite3.Connection):
        if self._st is None:
            self._st = time.time()
            self._s_progress = current_progress
            time_per_step = 0
        else:
            time_per_step = (time.time() - self._st) / (current_progress - self._s_progress)
        seconds = int(round((total - current_progress) * time_per_step))

        repository.insert_or_replace(connection, Task.TABLE_NAME, [{
            Task.TASK_NAME: self._key,
            Task.TASK_STATE: state,
            Task.TASK_TIME: time.time()
        }])

        line = "%s: %d / %d, %.2lf%%, remain = %s" % (
            self._key, current_progress, total, current_progress / total * 100,
            f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
        print(line)


def fetch_user_ranking(game_mode, variant, max_page=10000, country=None):
    """
    fetch and save user's ranking in table User
    @param game_mode: osu game mode
    @param variant: the number of columns in mania mode
    @param max_page:the number of fetching ranking range, a page contains 50 users
    @param country:a parameter for fetching specialized country ranking
    """
    total_count = 10000
    progress_control = ProgressControl(
        "fetch_user_ranking_%s_%s_%s" % (game_mode, variant, country),
        total_count
    )
    current_count, current_page = tuple(map(int, progress_control.get_state("0 1").split(" ")))
    conn_ranking = repository.get_connection()
    while current_count < total_count and current_page < max_page:
        param = {
            "variant": variant,
            "cursor[page]": current_page
        }
        if country is not None:
            param['country'] = country
        data = api.request_auth_api("rankings/%s/performance" % game_mode, "GET", param)
        total_count = data['total']
        if 'cursor' not in data or data['cursor'] is None:
            break
        current_page = data['cursor']['page']

        db_data = list(map(lambda r: {
            User.ID: r['user']['id'],
            User.NAME: r['user']['username'],
            User.GAME_MODE: game_mode,
            User.VARIANT: variant,
            User.PP: r['pp'],
            User.RANK: r['global_rank'],
            User.PLAY_COUNT: r['play_count'],
            User.PLAY_TIME: r['play_time'],
            User.COUNTRY: r['user']['country']['name'],
            User.DIRTY: False
        }, data['ranking']))
        current_count += len(db_data)
        with conn_ranking:

            for user_data in db_data:
                old = repository.select_first(conn_ranking, User.TABLE_NAME,
                                              project=[User.PP, User.DIRTY],
                                              where={
                                                  User.ID: user_data[User.ID],
                                                  User.GAME_MODE: user_data[User.GAME_MODE],
                                                  User.VARIANT: user_data[User.VARIANT],
                                              })
                old_pp = 0
                old_dirty = False
                if old is not None:
                    old_pp = float(old[0])
                    old_dirty = old[1]
                new_pp = float(user_data[User.PP])
                user_data[User.DIRTY] = abs(old_pp - new_pp) > 1
                if user_data[User.DIRTY]:
                    print(
                        f"[Dirty] {user_data[User.NAME]} - {user_data[User.ID]}: old = {old_pp}, new = {new_pp}")
                if old_dirty:
                    user_data[User.DIRTY] = old_dirty

            repository.insert_or_replace(conn_ranking, User.TABLE_NAME, db_data)
            if current_count / total_count > current_page / max_page:
                current_progress = current_count
                total_progress = total_count
            else:
                current_progress = current_page
                total_progress = max_page
            progress_control.commit(str(current_count) + " " + str(current_page),
                                    current_progress, total_progress,
                                    conn_ranking)


def fetch_best_performance_for_user(game_mode, user_id, connection, enable_retry=True):
    """
    fetch personal specialized data from user's bp
    @param game_mode: osu game mode
    @param user_id: osu id
    @param connection: database connection
    @param enable_retry:a parameter controlling whether starting retry feature up
    @return:filtered beatmap data as beatmap_db_data, filtered score data as score_db_data
    """
    data = api.request_auth_api("users/%d/scores/best" % user_id, "GET", {
        "mode": game_mode,
        "limit": 100
    }, enable_retry=enable_retry)
    if 'error' in data:
        print("ERROR: " + api.recent_request)
        # user does not exist, skip!!
        return None, None

    def filter_beatmap(x):
        """
        filter convert maps from osu mode
        @param x:original personal best performance data from osu api
        @return:bool
        """
        beatmap = x['beatmap']
        if beatmap['convert']:
            return False
        return True

    data = list(filter(filter_beatmap, data))

    beatmap_db_data = list(
        filter(lambda x: x is not None,
               map(lambda x: parse_beatmap_data(x['beatmap'], x['beatmapset'], connection,
                                                mode=game_mode),
                   data)
               )
    )
    score_db_data = list(map(lambda x: parse_score_data(x, x['beatmap']['id'], False), data))

    return beatmap_db_data, score_db_data


def fetch_best_performance(game_mode, max_user=1000000):
    """
    fetch and save users' bp related data in table Score and table Beatmap
    @param game_mode:osu game mode
    @param max_user: the number of fetching user range
    """
    conn = repository.get_connection()
    with conn:
        user_id_set = set(map(lambda x: x[0], repository.select(
            conn, User.TABLE_NAME, [User.ID],
            where={
                User.GAME_MODE: game_mode,
                User.DIRTY: True
            }, limit=max_user, order_by=User.RANK)))
    progress_control = ProgressControl("fetch_best_performance_%s" % (game_mode), len(user_id_set))
    previous_state = int(progress_control.get_state(-1))
    for i, user_id in enumerate(user_id_set):
        print(f"[Best performance] {user_id}")
        if i <= previous_state:
            continue
        beatmap_db_data, score_db_data = fetch_best_performance_for_user(game_mode, user_id, conn)
        if beatmap_db_data is None or score_db_data is None:
            continue

        with conn:
            repository.insert_or_replace(conn, Beatmap.TABLE_NAME, beatmap_db_data)
            insert_scores(conn, score_db_data)
            progress_control.commit(str(i), i + 1, len(user_id_set), conn)
    with conn:
        repository.update(conn, User.TABLE_NAME, puts=[{
            User.DIRTY: False
        }], wheres=[{
            User.DIRTY: True
        }])
    # conn.close()


def fetch_ranked_beatmaps(mode_int, max_maps=1000000):
    """
    fetch and save ranked map related data in Table Beatmap
    @param mode_int: a code name of osu game mode
    @param max_maps: the number of fetching maps
    """
    progress_control = ProgressControl("fetch_ranked_beatmaps_%d_auth_2" % (mode_int), max_maps)
    state = json.loads(progress_control.get_state(json.dumps({
        'cur_count': 0,
        'total': max_maps,
        'params': {
            'm': mode_int,
            's': 'ranked'
        }
    })))
    beatmap_conn = repository.get_connection()
    while True:
        if state['cur_count'] >= max_maps:
            break
        data = api.request_auth_api('beatmapsets/search/', method='GET', params=state['params'])
        state['total'] = data['total']
        db_data = []
        for beatset in data["beatmapsets"]:
            for beatmap in beatset["beatmaps"]:
                x = parse_beatmap_data(beatmap, beatset, beatmap_conn, mode_int=mode_int)
                if x is None:
                    continue
                db_data.append(x)
            state['cur_count'] += 1
        with beatmap_conn:
            repository.insert_or_replace(beatmap_conn, Beatmap.TABLE_NAME, db_data)
            progress_control.commit(json.dumps(state), state['cur_count'],
                                    min(state['total'], max_maps),
                                    beatmap_conn)
        if 'cursor_string' in data and data['cursor_string'] is not None:
            state['params']["cursor_string"] = data["cursor_string"]
        else:
            break
    # beatmap_conn.close()


def fetch_beatmap_top_scores(game_mode, variant, max_beatmap=100000):
    """
    fetch and save top scores of ranked beatmaps related data in Table Score
    @param game_mode: osu game mode
    @param variant: the number of columns in mania mode but CS in osu mode
    @param max_beatmap: the number of fetching maps
    """
    connection = repository.get_connection()
    with connection:
        where = {
            Beatmap.GAME_MODE: game_mode
        }
        if game_mode == 'mania':
            where[Beatmap.CS] = int(variant[0])
        beatmap_ids = list(repository.select(
            connection, Beatmap.TABLE_NAME, [Beatmap.ID, Beatmap.SUM_SCORES],
            where=where, limit=max_beatmap))
    progress_control = ProgressControl("fetch_beatmap_top_scores_%s_%s" % (game_mode, variant),
                                       len(beatmap_ids) * 6)
    previous_state = int(progress_control.get_state(-1))
    for i, (beatmap_id, sum_scores) in enumerate(beatmap_ids):
        if i <= previous_state:
            continue
        j = 0
        dirty = True
        for mods in ['', 'DT', 'HT']:
            for type in ['global', 'country']:
                if dirty and random.random() > 0.5:
                    data = api.request_auth_api(
                        'beatmaps/{beatmap}/scores'.format(beatmap=beatmap_id),
                        'GET',
                        params={
                            'mode': game_mode,
                            'mods[]': mods,
                            'type': type
                        })
                    score_db_data = list(
                        map(lambda x: parse_score_data(x, beatmap_id, True), data['scores']))
                else:
                    score_db_data = None
                # check dirty
                if j == 0 and score_db_data is not None:
                    cur_sum_scores = sum(map(lambda x: x[Score.SCORE], score_db_data))
                    if cur_sum_scores == sum_scores:
                        dirty = False
                    else:
                        with connection:
                            repository.update(connection, Beatmap.TABLE_NAME, [{
                                Beatmap.SUM_SCORES: cur_sum_scores
                            }], [{
                                Beatmap.ID: beatmap_id
                            }])
                    print(beatmap_id, dirty)
                j += 1
                with connection:
                    if score_db_data is not None:
                        insert_scores(connection, score_db_data)
                    progress_control.commit(str(i), i * 6 + j,
                                            len(beatmap_ids) * 6, connection)
                # break


def filter_score_data(conn, score_db_data):
    """
    make a comparison between previous pp and current pp,then choosing the higher one
    @param conn:database connection
    @param score_db_data: filtered score data
    @return:compared filtered score data
    """
    data = []
    for score_dict in score_db_data:
        previous_pp = repository.select(conn, Score.TABLE_NAME, project=[Score.PP], where={
            Score.BEATMAP_ID: score_dict[Score.BEATMAP_ID],
            Score.USER_ID: score_dict[Score.USER_ID],
            Score.SPEED: score_dict[Score.SPEED],
        }).fetchone()
        # We only insert the scores with the highest pp.
        if previous_pp is None or previous_pp[0] < score_dict[Score.PP]:
            data.append(score_dict)
    return data


def insert_scores(conn, score_db_data):
    """
    save filtered score data in Table Score
    @param conn: database connection
    @param score_db_data:filtered score data
    """
    data = filter_score_data(conn, score_db_data)
    repository.insert_or_replace(conn, Score.TABLE_NAME, data, or_ignore=False)


def post_process_db_std(user_id=None, conn=None):
    """
    fill up the blanks about Score.GAME_MODE, Score.CS
    @param user_id:osu uid
    @param conn:database connection
    """
    if conn is None:
        conn = repository.get_connection()
    with conn:
        # insert game_mode and cs into Score
        repository.ensure_column(conn, Score.TABLE_NAME, [
            (Score.GAME_MODE, "TEXT", None),
            (Score.CS, "INTEGER", None),
        ])
        if user_id is not None:
            extra_cs = f" WHERE {Score.USER_ID} = {user_id}"
        else:
            extra_cs = ""
        repository.execute_sql(conn,
                               f"UPDATE {Score.TABLE_NAME} SET ({Score.GAME_MODE}, {Score.CS}) = "
                               f"(SELECT {Beatmap.GAME_MODE}, {Beatmap.CS} "
                               f"FROM {Beatmap.TABLE_NAME} "
                               f"WHERE {Beatmap.TABLE_NAME}.{Beatmap.ID} == "
                               f"{Score.TABLE_NAME}.{Score.BEATMAP_ID} "
                               f") {extra_cs}")
        # drop task table
        repository.execute_sql(conn, f"DROP TABLE IF EXISTS {Task.TABLE_NAME}")

def post_process_db_mania(user_id=None, conn=None):
    """
    fill up the blanks about Score.GAME_MODE, Score.CS(aka the number of columns in mania mode) and Score.CUSTOM_ACCURACY
    @param user_id:osu uid
    @param conn:database connection
    """
    if conn is None:
        conn = repository.get_connection()
    with conn:
        # insert game_mode and cs into Score
        repository.ensure_column(conn, Score.TABLE_NAME, [
            (Score.GAME_MODE, "TEXT", None),
            (Score.CS, "INTEGER", None),
            (Score.CUSTOM_ACCURACY, "FLOAT", None),
        ])
        if user_id is not None:
            extra_cs = f" WHERE {Score.USER_ID} = {user_id}"
            extra_update = f" AND {Score.USER_ID} = {user_id}"
        else:
            extra_cs = ""
            extra_update = ""
        repository.execute_sql(conn,
                               f"UPDATE {Score.TABLE_NAME} SET ({Score.GAME_MODE}, {Score.CS}) = "
                               f"(SELECT {Beatmap.GAME_MODE}, {Beatmap.CS} "
                               f"FROM {Beatmap.TABLE_NAME} "
                               f"WHERE {Beatmap.TABLE_NAME}.{Beatmap.ID} == "
                               f"{Score.TABLE_NAME}.{Score.BEATMAP_ID} "
                               f") {extra_cs}")
        # update custom acc for mania
        weight_mania = {
            Score.COUNT_geki: 320,
            Score.COUNT_300: 300,
            Score.COUNT_katu: 200,
            Score.COUNT_100: 100,
            Score.COUNT_50: 50,
            Score.COUNT_miss: 0,
        }
        custom_acc_sum = " + ".join([f"{k} * {v}" for k, v in weight_mania.items()])
        custom_acc_total = " + ".join(weight_mania.keys())
        repository.execute_sql(conn,
                               f"UPDATE {Score.TABLE_NAME} "
                               f"SET {Score.CUSTOM_ACCURACY} = ({custom_acc_sum}) * 1.0 / ({custom_acc_total}) / 320 "
                               f"WHERE {Score.GAME_MODE} == 'mania' {extra_update}")
        # drop task table
        repository.execute_sql(conn, f"DROP TABLE IF EXISTS {Task.TABLE_NAME}")


def fetch_best_performance_for_user_online(config, uid, connection):
    """
    update table Beatmap and table Score in data according to personal osu bp
    @param config: config
    @param uid: osu id
    @param connection: database connection
    @return: nothing
    """
    beatmap_db_data, score_db_data = fetch_best_performance_for_user(game_mode=config.game_mode,
                                                                     user_id=uid,
                                                                     connection=connection,
                                                                     enable_retry=False)
    if beatmap_db_data is None or score_db_data is None:
        return
    with connection:
        repository.insert_or_replace(connection, Beatmap.TABLE_NAME, beatmap_db_data)
        insert_scores(connection, score_db_data)

def set_auth_file(auth_file):
    api.auth_file = auth_file

def update_single_user(connection, config: NetworkConfig, user_name=None, user_id=None):
    """
    updating personal bp and training personal UserEmbedding
    @param connection: database connection
    @param config: training config
    @param user_name: osu username
    @param user_id: osu uid
    @return:a bool to explain whether update user's bp and embedding correctly
    """
    game_mode = config.game_mode
    print("Fetching user info...")
    if user_name is not None:
        r = api.request_auth_api(f'users/{user_name}/{game_mode}', method='GET', params={
            "key": "username"
        }, enable_retry=False)
    else:
        r = api.request_auth_api(f'users/{user_id}/{game_mode}', method='GET', params={
            "key": "id"
        }, enable_retry=False)
    if "error" in r:
        return False, None
    user_data = {
        User.ID: r['id'],
        User.NAME: r['username'],
        User.GAME_MODE: game_mode,
        User.RANK: r['statistics']['global_rank'],
        User.PLAY_COUNT: r['statistics']['play_count'],
        User.PLAY_TIME: r['statistics']['play_time'],
        User.COUNTRY: r['country']['name'],
        User.DIRTY: False
    }

    def process(u_data):
        variant = u_data[User.VARIANT]
        uid = u_data[User.ID]
        with connection:
            repository.insert_or_replace(connection, User.TABLE_NAME, [u_data])
        fetch_best_performance_for_user_online(config, uid, connection)
        if game_mode == 'mania':
            post_process_db_mania(uid, connection)
        elif game_mode == 'osu':
            post_process_db_std(uid, connection)
        else:
            raise
        train_personal_embedding_online(config, f"{uid}-{game_mode}-{variant}", connection)

    if game_mode == 'mania':
        for v in r['statistics']['variants']:
            if v['mode'] != game_mode:
                continue
            user_variant_data = user_data.copy()
            user_variant_data[User.VARIANT] = v['variant']
            user_variant_data[User.PP] = v['pp']
            if v['pp'] < 10:
                continue
            print(f"Fetching scores: ({v['variant']}) ...")
            process(user_variant_data)
    else:
        user_data[User.VARIANT] = ""
        user_data[User.PP] = r['statistics']['pp']
        process(user_data)

    return True, r['id']

def fetch_mania():
    """
    integrate functions among fetch users' ranking, users' bp, ranked beatmaps and top scores of ranked beatmaps
    """
    with repository.get_connection() as conn_:
        User.create(conn_)
    Beatmap.create(conn_)
    Score.create(conn_)
    Task.create(conn_)

    try:
        for variant in ['4k', '7k']:
            for country in [None, "CN", "US", "KR"]:
                fetch_user_ranking(game_mode='mania', variant=variant, country=country)

        fetch_ranked_beatmaps(3)
        fetch_best_performance(game_mode='mania')
        fetch_beatmap_top_scores(game_mode='mania', variant='4k')
        fetch_beatmap_top_scores(game_mode='mania', variant='7k')
        post_process_db_mania()
    except Exception as e:
        print("ERROR: " + str(api.recent_request))
        raise e

def fetch_std():
    """
    integrate functions among fetch users' ranking, users' bp, ranked beatmaps and top scores of ranked beatmaps
    """
    with repository.get_connection() as conn_:
        User.create(conn_)
        Beatmap.create(conn_)
        Score.create(conn_)
        Task.create(conn_)

    try:
        for country in [None, "US", "RU", "DE", "CA", "PL", "PH", "FR", "JP", "BR", "GB",
                        "ID", "AU", "TW", "CL", "MY", "KR", "UA", "MX", "CN"]:
            fetch_user_ranking(game_mode='osu', variant="", country=country)

        fetch_ranked_beatmaps(0)
        fetch_best_performance(game_mode='osu')
        post_process_db_std()

        conn = repository.get_connection()
        for x in repository.select(conn, Beatmap.TABLE_NAME, [Beatmap.ID, Beatmap.MOD_STAR, Beatmap.MOD_MAX_PP]):
            bid, mod_star, mod_max_pp = x
            if mod_star == "{}":
                mod_star, mod_max_pp = ensure_beatmap_attributes(bid, mod_star, mod_max_pp, "osu")
                repository.update(conn, Beatmap.TABLE_NAME, [{
                    Beatmap.MOD_STAR: json.dumps(mod_star, separators=(',', ':')),
                    Beatmap.MOD_MAX_PP: json.dumps(mod_max_pp, separators=(',', ':')),
                }], [{
                    Beatmap.ID: bid
                }])
                print(bid, mod_star, mod_max_pp)
                conn.commit()

    except Exception as e:
        print("ERROR: " + str(api.recent_request))
        raise e

if __name__ == "__main__":
    with repository.get_connection() as conn_:
        User.create(conn_)
    Beatmap.create(conn_)
    Score.create(conn_)
    Task.create(conn_)
    fetch_ranked_beatmaps(3)
