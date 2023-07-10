import math

import osu_utils
from score.provider import *


class ManiaScoreDataProvider(BaseScoreDataProvider):

    def __init__(self, weights: ScoreModelWeight, config: NetworkConfig,
                 connection: sqlite3.Connection):
        super().__init__(weights, config, connection)
        self.speed_to_mod_map = ['HT', 'NM', 'DT']

    def provide_user_data(self, user_key: str, epoch, ignore_less=True):
        user_id, user_mode, user_variant = user_key.split("-")
        if user_mode != self.config.game_mode:
            return None
        sql = (
            f"SELECT {Score.BEATMAP_ID}, s.speed, s.score, s.pp, s.{Score.CUSTOM_ACCURACY} "
            f"FROM {Score.TABLE_NAME} as s "
            f"WHERE s.{Score.USER_ID} == {user_id} "
            f"AND s.{Score.CS} == {user_variant[0]} "
            f"AND s.{Score.GAME_MODE} == '{self.config.game_mode}' "
            f"AND s.{Score.PP} >= 1 "
            f"AND NOT s.{Score.IS_EZ}")
        scores = []
        mod_emb_id = []
        beatmap_emb_id = []
        pps = []
        for x in repository.execute_sql(self.connection, sql):
            score = x[2] * 2 if x[1] == -1 else x[2]
            if score < osu_utils.min_score:
                continue
            acc = x[-1]
            if acc < osu_utils.min_acc:
                continue
            if str(x[0]) not in self.weights.beatmap_embedding.key_to_embed_id:
                continue

            # scores:
            beatmap_emb_id.append(self.weights.beatmap_embedding.key_to_embed_id[str(x[0])])
            mod_emb_id.append(
                self.weights.mod_embedding.key_to_embed_id[self.speed_to_mod_map[x[1] + 1]]
            )
            scores.append(
                osu_utils.map_osu_score(score, real_to_train=True, arctanh=math.atanh,
                                        tanh=math.tanh))
            pps.append(x[3])

            # accs:
            beatmap_emb_id.append(self.weights.beatmap_embedding.key_to_embed_id[str(x[0])])
            mod_emb_id.append(
                self.weights.mod_embedding.key_to_embed_id[
                    self.speed_to_mod_map[x[1] + 1] + "-ACC"])
            scores.append(
                osu_utils.map_osu_acc(acc, real_to_train=True, arctanh=math.atanh, tanh=math.tanh))
            pps.append(x[3])
        if len(scores) < self.config.embedding_size and ignore_less:
            return None
        if len(scores) == 0:
            return None
        
        # only train top-200 beatmaps
        pp_sorted_list = sorted(pps, reverse=True)
        if len(pp_sorted_list) >= 200:
            min_pp = pp_sorted_list[199]
        else:
            min_pp = 0
        pps = np.asarray(pps)

        regression_weights = np.clip(pps / np.mean(pps), 1 / self.config.pp_weight_clip,
                                     self.config.pp_weight_clip)
        regression_weights = np.where(
            pps >= min_pp,
            regression_weights, 
            regression_weights / 100
        )
        return (np.asarray(scores, dtype=np.float32),
                np.asarray(beatmap_emb_id, dtype=np.int32),
                np.asarray(mod_emb_id, dtype=np.int32),
                regression_weights)

    def provide_beatmap_data(self, beatmap_key: str, epoch, ignore_less=True):
        sql = (
            f"SELECT s.user_id, s.cs, s.speed, s.score, s.{Score.CUSTOM_ACCURACY} "
            f"FROM {Score.TABLE_NAME} as s "
            f"WHERE s.{Score.BEATMAP_ID} == {beatmap_key} "
            f"AND s.{Score.GAME_MODE} == '{self.config.game_mode}' "
            f"AND s.{Score.PP} >= 1 "
            f"AND NOT s.{Score.IS_EZ}")
        scores = []
        mod_emb_id = []
        user_emb_id = []
        for x in repository.execute_sql(self.connection, sql):
            score = x[3] * 2 if x[2] == -1 else x[3]
            if score < osu_utils.min_score:
                continue
            acc = x[-1]
            if acc < osu_utils.min_acc:
                continue
            user_key = f"{x[0]}-{self.config.game_mode}-{x[1]}k"
            if user_key not in self.weights.user_embedding.key_to_embed_id:
                continue

            # scores
            user_emb_id.append(self.weights.user_embedding.key_to_embed_id[user_key])
            mod_emb_id.append(
                self.weights.mod_embedding.key_to_embed_id[self.speed_to_mod_map[x[2] + 1]]
            )
            scores.append(
                osu_utils.map_osu_score(score, real_to_train=True, arctanh=math.atanh,
                                        tanh=math.tanh))

            # accs
            user_emb_id.append(self.weights.user_embedding.key_to_embed_id[user_key])
            mod_emb_id.append(
                self.weights.mod_embedding.key_to_embed_id[self.speed_to_mod_map[x[2] + 1] + "-ACC"]
            )
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
            get_weight(i) for i in self.weights.mod_embedding.key_to_embed_id.values()
        ])[mod_emb_id]

        if len(scores) < self.config.embedding_size:
            return None
        return (np.asarray(scores, dtype=np.float32),
                np.asarray(user_emb_id, dtype=np.int32),
                mod_emb_id,
                sample_weights)

    def provide_mod_data(self, mod_key: str, epoch, ignore_less=True):
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
            f"AND s.{Score.GAME_MODE} == '{self.config.game_mode}' "
            f"AND s.{Score.PP} >= 1 "
            f"AND NOT s.{Score.IS_EZ} ")
        sql += f"AND s.{Score.SCORE_ID} % 30 == {epoch}"
        scores = []
        beatmap_emb_id = []
        user_emb_id = []
        for x in repository.execute_sql(self.connection, sql):
            beatmap_id, user_id, cs, score, acc = x
            if speed == -1:
                score = score * 2
            if score < osu_utils.min_score or acc < osu_utils.min_acc:
                continue
            user_key = f"{user_id}-{self.config.game_mode}-{cs}k"
            if user_key not in self.weights.user_embedding.key_to_embed_id:
                continue
            beatmap_key = str(beatmap_id)

            user_emb_id.append(self.weights.user_embedding.key_to_embed_id[user_key])
            beatmap_emb_id.append(self.weights.beatmap_embedding.key_to_embed_id[beatmap_key])

            if is_acc:
                scores.append(
                    osu_utils.map_osu_acc(acc, real_to_train=True, arctanh=math.atanh,
                                          tanh=math.tanh))
            else:
                scores.append(osu_utils.map_osu_score(score, real_to_train=True, arctanh=math.atanh,
                                                      tanh=math.tanh))

        if len(scores) < self.config.embedding_size:
            return None
        return (np.asarray(scores, dtype=np.float32),
                np.asarray(user_emb_id, dtype=np.int32),
                np.asarray(beatmap_emb_id, dtype=np.int32),
                None)
