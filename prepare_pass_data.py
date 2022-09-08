import matplotlib.pyplot as plt
from data import data_process
from data.model import *
import osu_utils
from tqdm import tqdm
import random
import time
import json
import math

tick = 0


def get_beatmap_key_to_info(connection, beatmap_data, config):
    transform = json.loads(Meta.get(connection, "tranform_" + config.game_mode, ""))

    def map_(raw, mapping_key):
        mapping = transform[mapping_key]
        return raw * mapping['std'] + mapping['mean']

    beatmap_key_to_info = {}
    for beatmap_key, x in beatmap_data.iterrows():
        od, star, count_circles, count_sliders, cs = x[
            ['od', 'star', 'count_circles', 'count_sliders', 'cs']]
        od = map_(od, 'od')
        star = map_(star, 'star')
        count_circles = map_(count_circles, 'count_circles')
        count_sliders = map_(count_sliders, 'count_sliders')
        cs = int(map_(cs, 'cs'))
        beatmap_key_to_info[beatmap_key] = (od, star, count_circles + count_sliders, cs)
    return beatmap_key_to_info


# @nb.jit(forceobj=False)
def view_score_hist():
    config = NetworkConfig()
    config.load_beyas = False
    weights = data_process.load_weight(config)
    scores, beatmap_data = data_process.load_score_data(config)

    scores = data_process.combine_data(scores, beatmap_data, weights)
    scores['score'] = osu_utils.map_osu_score(scores['score'].to_numpy(), real_to_train=False)

    user_keys = list(set(scores['user_key'].tolist()))
    random.shuffle(user_keys)

    beatmap_keys = set(scores['beatmap_key'].tolist())
    scores.set_index(['user_key', 'beatmap_key'], inplace=True)

    # for _, x in tqdm(scores.iterrows(), total=len(scores)):
    #     ubkey_to_score[x['user_key'] + x['beatmap_key']] = x['score']
    #     user_keys.update(x['user_key'])
    #     beatmap_keys.update(x['beatmap_key'])

    divide = 1000
    divide2 = 1
    grid_score_to_pass = [[0, 0] for _ in range(int(np.ceil(500000 / divide)))]
    bp_num_to_pass = [[0, 0] for _ in range(int(np.ceil(105 / divide2)))]
    index = set(scores.index)
    connection = repository.get_connection()

    beatmap_key_to_info = get_beatmap_key_to_info(connection, beatmap_data, config)

    for user_key in tqdm(user_keys):  # ['10702235-mania-4k']: #tqdm(user_keys[:10]):
        user_id, _, variant = user_key.split('-')
        user_bp = osu_utils.get_user_bp(connection, user_id, config, variant[0])
        pps = user_bp.data['pp'].to_numpy()[::-1]
        # print(pps)

        user_emb_id = int(weights.user_embedding.key_to_embed_id[user_key])
        user_embedding = weights.user_embedding.embeddings[0][user_emb_id]
        for beatmap_key in beatmap_keys:
            od, star, notes, cs = beatmap_key_to_info[beatmap_key]
            if str(cs) != variant[0]:
                continue
            t = (user_key, beatmap_key)

            if t in index:
                score = scores.loc[t, 'score'] + 1000
                # pp = scores.loc[t, 'pp']
                # print('true_pp', pp)
                passed = 0
            else:
                passed = 1

                map_emb_id = int(weights.beatmap_embedding.key_to_embed_id[beatmap_key])
                beatmap_embedding = weights.beatmap_embedding.embeddings[0][map_emb_id]
                predict = np.dot(user_embedding[:-1], beatmap_embedding[:-1]) + user_embedding[-1] + \
                          beatmap_embedding[-1]
                score = np.clip(osu_utils.map_osu_score(predict, real_to_train=False), 500000 + 1,
                                1000000 - 1)

                # score = 750000

            # od = 5
            # star = 6
            # count_sliders = count_circles = 1000
            pp = osu_utils.mania_pp(score, od, star, notes)
            pp_rank = len(pps) - bisect.bisect_left(pps, pp)

            # print(pp, pp_rank)
            i = int(pp_rank / divide2)
            if i >= len(grid_score_to_pass):
                i = len(grid_score_to_pass) - 1
            elif i < 0:
                i = 0
            bp_num_to_pass[i][passed] += 1
            # print(pp_rank)

            i = int((score - 500000) / divide)
            if i >= len(grid_score_to_pass):
                i = len(grid_score_to_pass) - 1
            elif i < 0:
                i = 0
            # print(i, score, grid_score_to_pass)
            grid_score_to_pass[i][passed] += 1
    print(grid_score_to_pass)
    print(bp_num_to_pass)

    x = np.arange(500000, 1000000, divide)
    y = list(map(lambda x: x[0] / (x[0] + x[1]) if x[0] + x[1] != 0 else 0.0, grid_score_to_pass))
    print(x, y)
    plt.plot(x, y)
    plt.show()

    plt.clf()
    x = np.arange(0, 105, divide2)
    y = list(map(lambda x: x[0] / (x[0] + x[1]) if x[0] + x[1] != 0 else 0.0, bp_num_to_pass))
    print(x, y)
    plt.plot(x, y)

    plt.show()
    # plt.hist(scores, bins=100, density=True)
    # plt.show()


def get_not_passed_candidates(conn, config: NetworkConfig, uid, variant, weights: ScoreModelWeight,
                              beatmap_embid_info: dict) -> set:
    user_bp = osu_utils.get_user_bp(conn, uid, config)
    if len(user_bp.data) < 100:
        return set()
    min_score = min(map(lambda x: x[1], user_bp.data.values()))
    min_pp = user_bp.pp_order_list[0]
    user_bp_data = set(map(lambda x: (x[0], x[1][0]), user_bp.data.items()))

    user_key = f'{uid}-{config.game_mode}-{variant}'
    user_emb = weights.user_embedding.embeddings[0][
        weights.user_embedding.key_to_embed_id[user_key]]
    cursor = []
    for key, mod_emb_id in weights.mod_embedding.key_to_embed_id.items():
        mod_emb = weights.mod_embedding.embeddings[0][mod_emb_id]
        if key == 'HT':
            speed = -1
            star = beatmap_embid_info[Beatmap.HT_STAR]
        elif key == 'NM':
            speed = 0
            star = beatmap_embid_info[Beatmap.STAR]
        elif key == 'DT':
            speed = 1
            star = beatmap_embid_info[Beatmap.DT_STAR]
        else:
            continue
        mod_score = measure_time(np.sum)(user_emb.reshape((1, -1)) * mod_emb.reshape((1, -1)) *
                                         weights.beatmap_embedding.embeddings[0],
                                         axis=1, keepdims=True)
        data = np.concatenate([
            mod_score,  # [B, 1]
            beatmap_embid_info[Beatmap.ID],
            speed * np.ones_like(mod_score),  # [B, 1]
            beatmap_embid_info[Beatmap.OD],
            beatmap_embid_info[Beatmap.COUNT_SLIDERS] + beatmap_embid_info[Beatmap.COUNT_CIRCLES],
            star
        ], axis=1)[beatmap_embid_info[Beatmap.CS] == int(variant[0])]  # [B, 9]
        cursor.append(data)
    cursor = np.concatenate(cursor, axis=0)
    cursor[:, 0] = osu_utils.map_osu_score(cursor[:, 0], real_to_train=False)
    not_pass_set = set()
    # data = cursor
    # for i in range(len(cursor)):
    #     # score = float(osu_utils.map_osu_score(x[0], real_to_train=False))
    #     score, bid, speed, od, count1, count2, star = cursor[i].tolist()
    #     data.append([bid, speed, score, od, star, count1 + count2])
    # data = np.asarray(data)
    pps = osu_utils.mania_pp(cursor[:, 0], cursor[:, 3], cursor[:, 5], cursor[:, 4])

    for i, pp in enumerate(pps):
        if pp > min_pp:
            bid = int(cursor[i, 1])
            speed = int(cursor[i, 2])
            score = cursor[i, 0]
            beatmap_key = (bid, speed)
            if beatmap_key in user_bp_data:
                continue
            not_pass_set.add(
                (bid, speed, score, pp, bisect.bisect_left(user_bp.pp_order_list, pp), False))

    if len(not_pass_set) >= 200:
        not_pass_set = set(random.sample(list(not_pass_set), 200))
    for i, bid in enumerate(user_bp.id_of_pp_order_list):
        (speed, score, pp, star) = user_bp.data[bid]
        not_pass_set.add((bid, speed, score, pp, i + 1, True))
    return not_pass_set


def prepare_not_passed_candidates(config: NetworkConfig):

    with repository.get_connection() as conn:
        repository.execute_sql(conn, f"DROP TABLE IF EXISTS {CannotPass.TABLE_NAME}")
        CannotPass.create(conn)
        user_sql = f"SELECT id, variant FROM User " \
                   f"WHERE User.pp >= 1000 AND {User.GAME_MODE} == '{config.game_mode}'"
        cursor = list(
            repository.execute_sql(conn, user_sql)
        )
        # repository.select(conn, User.TABLE_NAME, project=[User.ID, User.VARIANT], where={
        #     User.GAME_MODE: config.game_mode
        # }))
        count = 0
        total_count = 0
        pbar = tqdm(cursor)
        pass_count = 0

        weights = data_process.load_weight(config)
        cur_emb = -1
        id_list, od_list, slider_list, circle_list, ht_list, nm_list, dt_list, cs_list = \
            [], [], [], [], [], [], [], []
        for key, emb_id in sorted(weights.beatmap_embedding.key_to_embed_id.items(),
                                  key=lambda x: x[1], reverse=False):
            assert cur_emb == emb_id - 1
            cur_emb = emb_id
            result = repository.select_first(conn, Beatmap.TABLE_NAME,
                                             project=[Beatmap.ID, Beatmap.OD, Beatmap.COUNT_SLIDERS,
                                                      Beatmap.COUNT_CIRCLES, Beatmap.HT_STAR,
                                                      Beatmap.STAR, Beatmap.DT_STAR, Beatmap.CS],
                                             where={Beatmap.ID: key})
            id_list.append(result[0])
            od_list.append(result[1])
            slider_list.append(result[2])
            circle_list.append(result[3])
            ht_list.append(result[4])
            nm_list.append(result[5])
            dt_list.append(result[6])
            cs_list.append(result[7])
        beatmap_info = {
            Beatmap.ID: np.array(id_list).reshape((-1, 1)),
            Beatmap.OD: np.array(od_list).reshape((-1, 1)),
            Beatmap.COUNT_CIRCLES: np.array(circle_list).reshape((-1, 1)),
            Beatmap.COUNT_SLIDERS: np.array(slider_list).reshape((-1, 1)),
            Beatmap.HT_STAR: np.array(ht_list).reshape((-1, 1)),
            Beatmap.STAR: np.array(nm_list).reshape((-1, 1)),
            Beatmap.DT_STAR: np.array(dt_list).reshape((-1, 1)),
            Beatmap.CS: np.array(cs_list),
        }

        # data = []
        for (uid, variant) in pbar:
            candidates = list(
                get_not_passed_candidates(conn, config, uid, variant, weights, beatmap_info))
            # if len(candidates) >= 200:
            #     candidates = random.sample(candidates, 200)
            if len(candidates) == 0:
                continue
            # data += list(map(lambda x: [uid, config.game_mode, variant, int(x[0]),
            #                             int(x[1]), int(x[2]), int(x[3]), x[4], x[5]], candidates))
            repository.insert_or_replace(conn, CannotPass.TABLE_NAME, list(
                map(lambda x: {
                    CannotPass.USER_ID: uid,
                    CannotPass.USER_GAME_MODE: config.game_mode,
                    CannotPass.USER_VARIANT: variant,
                    CannotPass.BEATMAP_ID: x[0],
                    CannotPass.SPEED: x[1],
                    CannotPass.SCORE: int(x[2]),
                    CannotPass.PP: int(x[3]),
                    CannotPass.PP_RANK: int(x[4]),
                    CannotPass.PASS: int(x[5]),
                }, candidates)))
            count += 1
            total_count += len(candidates)
            pass_count += sum(map(lambda x: int(x[5]), candidates))
            if count == 100:
                conn.commit()
                count = 0
            pbar.set_description(f"{pass_count / total_count:.3f}", refresh=True)
        # data_df = pd.DataFrame(data, columns=[CannotPass.USER_ID, CannotPass.USER_GAME_MODE,
        #                                       CannotPass.USER_VARIANT, CannotPass.BEATMAP_ID,
        #                                       CannotPass.SPEED, CannotPass.SCORE,
        #                                       CannotPass.PP, CannotPass.PP_RANK,
        #                                       CannotPass.PASS])
        # data_df.to_sql(CannotPass.TABLE_NAME, conn, if_exists='replace')


def shuffle_cannot_pass():
    temp_name = f"temp_{CannotPass.TABLE_NAME}"
    with repository.get_connection() as conn:
        st = time.time()
        repository.execute_sql(conn, f"CREATE TABLE {temp_name} AS SELECT * FROM {CannotPass.TABLE_NAME} ORDER BY RANDOM()")
        print("create random temp", time.time() - st)

        st = time.time()
        repository.execute_sql(conn, f"DROP TABLE IF EXISTS {CannotPass.TABLE_NAME}")
        print("drop old", time.time() - st)

        st = time.time()
        repository.execute_sql(conn, f"CREATE TABLE {CannotPass.TABLE_NAME} AS SELECT * FROM {temp_name}")
        print("move new", time.time() - st)

        st = time.time()
        repository.execute_sql(conn, f"DROP TABLE IF EXISTS {temp_name}")
        print("drop new", time.time() - st)





if __name__ == "__main__":

    prepare_not_passed_candidates(NetworkConfig())
    shuffle_cannot_pass()

    # conn = repository.get_connection()
    # config = NetworkConfig()
    # weights = data_process.load_weight(config)
    # cur_emb = -1
    # id_list, od_list, slider_list, circle_list, ht_list, nm_list, dt_list, cs_list = \
    #     [], [], [], [], [], [], [], []
    # for key, emb_id in sorted(weights.beatmap_embedding.key_to_embed_id.items(),
    #                           key=lambda x: x[1], reverse=False):
    #     assert cur_emb == emb_id - 1
    #     cur_emb = emb_id
    #     result = repository.select_first(conn, Beatmap.TABLE_NAME,
    #                                      project=[Beatmap.ID, Beatmap.OD, Beatmap.COUNT_SLIDERS,
    #                                               Beatmap.COUNT_CIRCLES, Beatmap.HT_STAR,
    #                                               Beatmap.STAR, Beatmap.DT_STAR, Beatmap.CS],
    #                                      where={Beatmap.ID: key})
    #     id_list.append(result[0])
    #     od_list.append(result[1])
    #     slider_list.append(result[2])
    #     circle_list.append(result[3])
    #     ht_list.append(result[4])
    #     nm_list.append(result[5])
    #     dt_list.append(result[6])
    #     cs_list.append(result[7])
    # beatmap_info = {
    #     Beatmap.ID: np.array(id_list).reshape((-1, 1)),
    #     Beatmap.OD: np.array(od_list).reshape((-1, 1)),
    #     Beatmap.COUNT_CIRCLES: np.array(circle_list).reshape((-1, 1)),
    #     Beatmap.COUNT_SLIDERS: np.array(slider_list).reshape((-1, 1)),
    #     Beatmap.HT_STAR: np.array(ht_list).reshape((-1, 1)),
    #     Beatmap.STAR: np.array(nm_list).reshape((-1, 1)),
    #     Beatmap.DT_STAR: np.array(dt_list).reshape((-1, 1)),
    #     Beatmap.CS: np.array(cs_list),
    # }
    # lst = get_not_passed_candidates(conn, NetworkConfig(), 7304075, '4k', weights, beatmap_info)
    # for x in lst:
    #     name = repository.select_first(conn, Beatmap.TABLE_NAME, project=[Beatmap.NAME, Beatmap.VERSION], where={
    #         Beatmap.ID: x[0]
    #     })
    #     print(name, x)
