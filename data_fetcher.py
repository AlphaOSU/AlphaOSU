from data import api
import json
from data.model import *
import osu_utils
import random
import datetime


def get_state(key, default):
    with repository.get_connection() as conn_progress:
        count = repository.count(conn_progress, Task.TABLE_NAME, where={Task.TASK_NAME: key})
        if count == 0:
            return default
        progress, t = repository.select_first(conn_progress, Task.TABLE_NAME,
                                              project=[Task.TASK_STATE, Task.TASK_TIME],
                                              where={Task.TASK_NAME: key})
        if t + 3 * 24 * 3600 >= time.time():
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
                old = repository.select_first(conn_ranking, User.TABLE_NAME, project=[User.PP],
                                              where={
                                                  User.ID: user_data[User.ID],
                                                  User.GAME_MODE: user_data[User.GAME_MODE],
                                                  User.VARIANT: user_data[User.VARIANT],
                                              })[0]
                user_data[User.DIRTY] = old is None or abs(float(old) - float(user_data[User.PP])) > 0.1

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
    # conn_ranking.close()


def fetch_best_performance(game_mode, max_user=100000):
    conn = repository.get_connection()
    with conn:
        user_id_name = list(repository.select(
            conn, User.TABLE_NAME, ['DISTINCT ' + User.ID, User.NAME, User.PP],
            where={
                User.GAME_MODE: game_mode,
                User.DIRTY: True
            }, limit=max_user, order_by=User.RANK))
        user_id_name = list(filter(lambda x: x[2] >= 1000, user_id_name))
    progress_control = ProgressControl("fetch_best_performance_%s" % (game_mode), len(user_id_name))
    previous_state = int(progress_control.get_state(-1))
    for i, (user_id, user_name, pp) in enumerate(user_id_name):
        if i <= previous_state:
            continue
        data = api.request_auth_api("users/%d/scores/best" % user_id, "GET", {
            "mode": game_mode,
            "limit": 100
        })
        if 'error' in data:
            print("ERROR: " + api.recent_request)
            # user does not exist, skip!!
            continue

        def filter_beatmap(x):
            beatmap = x['beatmap']
            if beatmap['convert']:
                return False
            return True

        data = list(filter(filter_beatmap, data))

        beatmap_db_data = list(
            filter(lambda x: x is not None,
                   map(lambda x: parse_beatmap_data(x['beatmap'], x['beatmapset'], conn),
                       data)
                   )
        )
        score_db_data = list(map(lambda x: parse_score_data(x, x['beatmap']['id'], False), data))

        with conn:
            repository.insert_or_replace(conn, Beatmap.TABLE_NAME, beatmap_db_data)
            insert_scores(conn, score_db_data)
            progress_control.commit(str(i), i + 1, len(user_id_name), conn)
    with conn:
        repository.update(conn, User.TABLE_NAME, puts=[{
            User.DIRTY: False
        }], wheres=[{
            User.DIRTY: True
        }])
    # conn.close()


def fetch_ranked_beatmaps(mode_int, max_maps=100000):
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
                x = parse_beatmap_data(beatmap, beatset, beatmap_conn)
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
                if dirty or random.random() > 0.9:
                    data = api.request_auth_api('beatmaps/{beatmap}/scores'.format(beatmap=beatmap_id),
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
                if j == 0:
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
                j += 1
                with connection:
                    if score_db_data is not None:
                        insert_scores(connection, score_db_data)
                    progress_control.commit(str(i), i * 6 + j,
                                            len(beatmap_ids) * 6, connection)
                # break


def insert_scores(conn, score_db_data):
    data = []
    for score_dict in score_db_data:
        previous_score = repository.select(conn, Score.TABLE_NAME, project=[Score.SCORE], where={
            Score.BEATMAP_ID: score_dict[Score.BEATMAP_ID],
            Score.USER_ID: score_dict[Score.USER_ID],
            Score.SPEED: score_dict[Score.SPEED],
        }).fetchone()
        if previous_score is None or previous_score[0] < score_dict[Score.SCORE]:
            data.append(score_dict)
    repository.insert_or_replace(conn, Score.TABLE_NAME, data, or_ignore=False)


def apply_speed_on_beatmap(game_mode):
    connection = repository.get_connection()
    repository.ensure_column(connection, Beatmap.TABLE_NAME, [
        (Beatmap.HT_STAR, "REAL", "-1.0"),
        (Beatmap.DT_STAR, "REAL", "-1.0"),
    ])
    score_cursor = repository.select(connection, Score.TABLE_NAME,
                                     project=[Score.BEATMAP_ID, Score.SPEED,
                                              f"MAX({Score.SCORE})", Score.PP],
                                     where={Score.SPEED: 0, Score.PP: 0}, where_format="%s != ?",
                                     group_by=[Score.BEATMAP_ID, Score.SPEED])
    score_cursor = list(score_cursor)
    # beatmap_extra = []
    puts_ht = []
    wheres_ht = []
    puts_dt = []
    wheres_dt = []
    for beatmap_id, speed, score, pp in score_cursor:
        beatmap_cursor = repository.select(connection, Beatmap.TABLE_NAME,
                                           project=[Beatmap.COUNT_SLIDERS, Beatmap.COUNT_CIRCLES,
                                                    Beatmap.OD],
                                           where={Beatmap.ID: beatmap_id,
                                                  Beatmap.GAME_MODE: game_mode})

        values = beatmap_cursor.fetchone()
        if values is None:
            continue
        if speed == -1:  # HT
            score = score * 2
        objects = values[0] + values[1]
        od = values[2]
        star = osu_utils.estimate_star_from_score(pp, score, objects, od)
        if speed == -1:
            puts_ht.append({
                Beatmap.HT_STAR: star
            })
            wheres_ht.append({
                Beatmap.ID: beatmap_id
            })
        elif speed == 1:
            puts_dt.append({
                Beatmap.DT_STAR: star
            })
            wheres_dt.append({
                Beatmap.ID: beatmap_id
            })

    repository.update(connection, Beatmap.TABLE_NAME, puts_ht, wheres_ht)
    repository.update(connection, Beatmap.TABLE_NAME, puts_dt, wheres_dt)
    # repository.insert_or_replace(connection, Beatmap.TABLE_NAME, beatmap_extra)
    connection.commit()


def post_process_db():
    with repository.get_connection() as conn:
        repository.ensure_column(conn, Score.TABLE_NAME, [
            (Score.GAME_MODE, "TEXT", None),
            (Score.CS, "INTEGER", None),
        ])
        repository.execute_sql(conn,
                               f"UPDATE {Score.TABLE_NAME} SET ({Score.GAME_MODE}, {Score.CS}) = "
                               f"(SELECT {Beatmap.GAME_MODE}, {Beatmap.CS} "
                               f"FROM {Beatmap.TABLE_NAME} "
                               f"WHERE {Beatmap.TABLE_NAME}.{Beatmap.ID} == "
                               f"{Score.TABLE_NAME}.{Score.BEATMAP_ID})")


def fetch():
    with repository.get_connection() as conn_:
        User.create(conn_)
    Beatmap.create(conn_)
    Score.create(conn_)
    Task.create(conn_)

    try:
        fetch_user_ranking(game_mode='mania', variant='4k')  # , max_page=10)
        fetch_user_ranking(game_mode='mania', variant='4k', country="CN")  # , max_page=10)
        fetch_user_ranking(game_mode='mania', variant='7k')  # , max_page=1)
        fetch_user_ranking(game_mode='mania', variant='7k', country="CN")  # , country="CN", max_page=1)
        fetch_ranked_beatmaps(3)
        fetch_best_performance(game_mode='mania')  # , max_user=50)
        fetch_beatmap_top_scores(game_mode='mania', variant='4k')  # , max_beatmap=10)
        fetch_beatmap_top_scores(game_mode='mania', variant='7k')  # , max_beatmap=10)
        #
        # with repository.get_connection() as conn_:
        #     conn_.execute("DROP TABLE IF EXISTS BeatmapCount")
        #     conn_.execute("CREATE TABLE BeatmapCount AS SELECT beatmap_id, speed, count(1) as count "
        #                  "FROM Score GROUP BY beatmap_id, speed")
        #     conn_.execute("CREATE UNIQUE INDEX BeatmapCountIndex ON BeatmapCount (beatmap_id, speed)")
        #     conn_.execute(f"CREATE INDEX IF NOT EXISTS score_user ON {Score.TABLE_NAME}({Score.USER_ID})")
        #     repository.ensure_column(conn_, User.TABLE_NAME, [(User.BP_MEAN_PP, "REAL", None)])
        #     conn_.execute(f"""UPDATE {User.TABLE_NAME} SET {User.BP_MEAN_PP} = (
        #         SELECT avg(pp) FROM (
        #             SELECT Score.pp as pp FROM Score
        #             JOIN Beatmap ON Score.beatmap_id == Beatmap.id
        #             WHERE Score.user_id == User.id AND Beatmap.game_mode == User.game_mode
        #             ORDER BY Score.pp DESC LIMIT 100
        #         )
        #     )""")
        apply_speed_on_beatmap("mania")
        post_process_db()
    except Exception as e:
        print("ERROR: " + str(api.recent_request))
        raise e

    # os.system('zip -r "%s" "%s"' % ("result.zip", "result"))

    # repository.export_db_to_csv(Beatmap.TABLE_NAME, "beatmap.csv")
    # repository.export_db_to_csv(User.TABLE_NAME, "user.csv")
    # repository.export_db_to_csv(Score.TABLE_NAME, "score.csv")


if __name__ == "__main__":
    fetch()