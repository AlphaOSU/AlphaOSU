import json

import osu_utils
from score.provider import *


class STDScoreDataProvider(BaseScoreDataProvider):

    def __init__(self, weights: ScoreModelWeight, config: NetworkConfig,
                 connection: sqlite3.Connection):
        super().__init__(weights, config, connection)
        self.dt_int = osu_utils.MOD_INT_MAPPING["DT"]
        self.hr_int = osu_utils.MOD_INT_MAPPING["HR"]
        self.hd_int = osu_utils.MOD_INT_MAPPING["HD"]
        self.beatmap_max_pp = {}

        cursor = repository.select(connection, Beatmap.TABLE_NAME,
                                   project=[Beatmap.ID, Beatmap.MOD_MAX_PP])
        for (bid, max_pp) in cursor:
            max_pp = json.loads(max_pp)
            self.beatmap_max_pp[bid] = max_pp

        # filter user/beatmap that have a low count
        self.dirty_user = set()
        self.dirty_beatmap = set()

    def get_mod_int(self, is_dt, is_hr, is_hd):
        mod_int = 0
        if is_dt:
            mod_int += self.dt_int
        if is_hr:
            mod_int += self.hr_int
        if is_hd:
            mod_int += self.hd_int
        return str(mod_int)

    def provide_user_data(self, user_key: str, epoch, ignore_less=True):
        user_id, user_mode, user_variant = user_key.split("-")
        if user_mode != self.config.game_mode:
            return None
        sql = (
            f"SELECT {Score.BEATMAP_ID}, {Score.PP}, {Score.IS_DT}, {Score.IS_HR}, {Score.IS_HD} "
            f"FROM {Score.TABLE_NAME} "
            f"WHERE {Score.USER_ID} == {user_id} "
            f"AND {Score.GAME_MODE} == '{self.config.game_mode}' "
            f"AND {Score.PP} >= 1 "
            f"AND NOT {Score.IS_EZ} AND NOT {Score.IS_FL} AND NOT {Score.IS_HT}"
        )
        mod_emb_id = []
        beatmap_emb_id = []
        pps = []
        for x in repository.execute_sql(self.connection, sql):
            bid, pp, is_dt, is_hr, is_hd = x
            mod_int = self.get_mod_int(is_dt, is_hr, is_hd)
            max_pp = self.beatmap_max_pp[bid][mod_int]
            # assert pp <= max_pp, f"PP = {pp}, mod = {mod_int}, but max_pp is {self.beatmap_max_pp[bid]}"
            pp = osu_utils.map_osu_pp(pp, real_to_train=True, max_pp=max_pp)

            if str(bid) not in self.weights.beatmap_embedding.key_to_embed_id:
                continue
            if int(bid) in self.dirty_beatmap:
                continue
            bid_emb = self.weights.beatmap_embedding.key_to_embed_id[str(bid)]

            beatmap_emb_id.append(bid_emb)
            mod_emb_id.append(
                self.weights.mod_embedding.key_to_embed_id[mod_int]
            )
            pps.append(pp)

        if len(pps) < self.config.embedding_size and ignore_less:
            self.dirty_user.add(int(user_id))
            return None
        if len(pps) == 0:
            self.dirty_user.add(int(user_id))
            return None
        pps = np.asarray(pps)

        regression_weights = np.clip(pps / np.mean(pps), 1 / self.config.pp_weight_clip,
                                     self.config.pp_weight_clip)
        return (np.asarray(pps, dtype=np.float32),
                np.asarray(beatmap_emb_id, dtype=np.int32),
                np.asarray(mod_emb_id, dtype=np.int32),
                regression_weights)

    def provide_beatmap_data(self, beatmap_key: str, epoch, ignore_less=True):
        sql = (
            f"SELECT {Score.USER_ID}, {Score.PP}, {Score.SCORE_ID}, "
            f"{Score.IS_DT}, {Score.IS_HR}, {Score.IS_HD} "
            f"FROM {Score.TABLE_NAME} "
            f"WHERE {Score.BEATMAP_ID} == {beatmap_key} "
            # f"AND {Score.GAME_MODE} == '{self.config.game_mode}' "
            f"AND {Score.PP} >= 1 "
            f"AND NOT {Score.IS_EZ} AND NOT {Score.IS_FL} AND NOT {Score.IS_HT}")
        mod_emb_id = []
        user_emb_id = []
        pps = []
        mod_max_pp = self.beatmap_max_pp[int(beatmap_key)]
        for x in repository.execute_sql(self.connection, sql):
            uid, pp, score_id, is_dt, is_hr, is_hd = x
            mod_int = self.get_mod_int(is_dt, is_hr, is_hd)
            if mod_int not in mod_max_pp:
                continue
            max_pp = mod_max_pp[mod_int]
            # assert pp <= max_pp, f"PP = {pp}, mod = {mod_int}, id = {score_id}, max_pp = {mod_max_pp}"
            pp = osu_utils.map_osu_pp(pp, real_to_train=True, max_pp=max_pp)

            user_key = f"{uid}-{self.config.game_mode}-"
            if user_key not in self.weights.user_embedding.key_to_embed_id:
                continue
            if int(uid) in self.dirty_user:
                continue

            user_emb_id.append(self.weights.user_embedding.key_to_embed_id[user_key])
            mod_emb_id.append(
                self.weights.mod_embedding.key_to_embed_id[mod_int]
            )
            pps.append(pp)

        if len(pps) < self.config.embedding_size:
            self.dirty_beatmap.add(int(beatmap_key))
            return None
        return (np.asarray(pps, dtype=np.float32),
                np.asarray(user_emb_id, dtype=np.int32),
                np.asarray(mod_emb_id, dtype=np.int32),
                None)

    def provide_mod_data(self, mod_key: str, epoch, ignore_less=True):
        is_dt = int(mod_key) & self.dt_int != 0
        is_hr = int(mod_key) & self.hr_int != 0
        is_hd = int(mod_key) & self.hd_int != 0
        mod_int = self.get_mod_int(is_dt, is_hr, is_hd)
        sql = (
            f"SELECT {Score.USER_ID}, {Score.BEATMAP_ID}, {Score.PP} "
            f"FROM {Score.TABLE_NAME} "
            f"WHERE {Score.IS_DT} == {int(is_dt)} "
            f"AND {Score.IS_HR} == {int(is_hr)} "
            f"AND {Score.IS_HD} == {int(is_hd)} "
            f"AND {Score.GAME_MODE} == '{self.config.game_mode}' "
            f"AND {Score.PP} >= 1 "
            f"AND NOT {Score.IS_EZ} AND NOT {Score.IS_FL} AND NOT {Score.IS_HT} ")
        sql += f"AND {Score.SCORE_ID} % 30 == {epoch}"
        pps = []
        beatmap_emb_id = []
        user_emb_id = []
        for x in repository.execute_sql(self.connection, sql):
            uid, bid, pp = x
            user_key = f"{uid}-{self.config.game_mode}-"
            if user_key not in self.weights.user_embedding.key_to_embed_id:
                continue
            beatmap_key = str(bid)
            if beatmap_key not in self.weights.beatmap_embedding.key_to_embed_id:
                continue
            max_pp = self.beatmap_max_pp[bid][mod_int]
            # assert pp <= max_pp, f"PP = {pp}, mod = {mod_int}, but max_pp is {self.beatmap_max_pp[beatmap_key]}"
            pp = osu_utils.map_osu_pp(pp, real_to_train=True, max_pp=max_pp)

            user_emb_id.append(self.weights.user_embedding.key_to_embed_id[user_key])
            beatmap_emb_id.append(self.weights.beatmap_embedding.key_to_embed_id[beatmap_key])
            pps.append(pp)

        if len(pps) < self.config.embedding_size:
            return None
        return (np.asarray(pps, dtype=np.float32),
                np.asarray(user_emb_id, dtype=np.int32),
                np.asarray(beatmap_emb_id, dtype=np.int32),
                None)
