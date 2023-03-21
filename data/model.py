import json
import os
import sqlite3
import bisect

import numpy as np
import pandas as pd

from data import repository


class User:
    TABLE_NAME = "User"
    PRIMARY_KEYS = ["id", "game_mode", "variant"]

    ID = 'id'
    NAME = 'name'
    GAME_MODE = "game_mode"
    VARIANT = "variant"
    PP = "pp"
    RANK = "rank"
    PLAY_COUNT = "play_count"
    PLAY_TIME = "play_time"
    COUNTRY = 'country'
    DIRTY = "dirty"

    BP_MEAN_PP = 'bp_mean_pp'

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=User.TABLE_NAME, columns={
            User.ID: "INTEGER NOT NULL",
            User.NAME: "TEXT",
            User.GAME_MODE: "TEXT NOT NULL",
            User.VARIANT: "TEXT NOT NULL",
            User.PP: "REAL",
            User.RANK: "INTEGER",
            User.PLAY_COUNT: "INTEGER",
            User.PLAY_TIME: "INTEGER",
            User.COUNTRY: "TEXT",
            User.DIRTY: "BOOLEAN DEFAULT FALSE"
        }, primary_keys=User.PRIMARY_KEYS)


class UserEmbedding:
    TABLE_NAME = "UserEmbedding"
    PRIMARY_KEYS = ["id", "game_mode", "variant"]

    USER_ID = 'id'
    GAME_MODE = 'game_mode'
    VARIANT = 'variant'

    COUNT = 'count'

    EMBEDDING = "embedding"
    EMBEDDING_BAYESIAN = "embedding_beyasian"

    NEIGHBOR_ID = "neighbor_id"
    NEIGHBOR_DISTANCE = "neighbor_distance"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=UserEmbedding.TABLE_NAME, columns={
            UserEmbedding.USER_ID: "INTEGER NOT NULL",
            UserEmbedding.GAME_MODE: "TEXT NOT NULL",
            UserEmbedding.VARIANT: "TEXT NOT NULL"
        }, primary_keys=UserEmbedding.PRIMARY_KEYS)

    @staticmethod
    def construct_where(uid, game_mode, variant):
        return {
            UserEmbedding.USER_ID: uid,
            UserEmbedding.GAME_MODE: game_mode,
            UserEmbedding.VARIANT: variant
        }

    @staticmethod
    def construct_where_with_key(user_key):
        user_id, game_mode, variant = user_key.split("-")
        return UserEmbedding.construct_where(user_id, game_mode, variant)


class BeatmapEmbedding:
    TABLE_NAME = "BeatmapEmbedding"
    PRIMARY_KEYS = ["id"]

    BEATMAP_ID = 'id'

    ITEM_EMBEDDING = "item_embedding"

    FINAL_EMBEDDING = "final_embedding"

    COUNT_HT = "count_HT"
    COUNT_NM = "count_NM"
    COUNT_DT = "count_DT"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=BeatmapEmbedding.TABLE_NAME, columns={
            BeatmapEmbedding.BEATMAP_ID: "INTEGER NOT NULL"
        }, primary_keys=BeatmapEmbedding.PRIMARY_KEYS)

    @staticmethod
    def construct_where(bid):
        return {
            BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.BEATMAP_ID: bid
        }


class ModEmbedding:
    TABLE_NAME = "ModEmbedding"
    PRIMARY_KEYS = ['mod']

    # TODO: in mania, mod stores text (e.g., DT, NM). In std, we stores mod int for convinience.
    MOD = 'mod'
    MOD_TEXT = "mod_text"
    SPEED = 'speed'
    IS_ACC = "is_acc"

    EMBEDDING = "mod_embedding"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=ModEmbedding.TABLE_NAME, columns={
            ModEmbedding.MOD: "TEXT NOT NULL",
            ModEmbedding.SPEED: "INTEGER NOT NULL",
            ModEmbedding.IS_ACC: "BOOLEAN NOT NULL",
            ModEmbedding.MOD_TEXT: "INTEGER DEFAULT -1",
        }, primary_keys=ModEmbedding.PRIMARY_KEYS)

    @staticmethod
    def construct_where(mod):
        return {
            ModEmbedding.MOD: mod
        }


class Score:
    TABLE_NAME = "Score"

    BEATMAP_ID = "beatmap_id"
    USER_ID = "user_id"
    SCORE_ID = "score_id"

    SPEED = "speed"  # -1: HT; 0: none; 1: DT
    IS_DT = "is_dt"
    IS_HR = "is_hr"
    IS_HD = "is_hd"
    IS_FL = "is_fl"
    IS_EZ = "is_ez"
    IS_MR = "is_mr"
    IS_HT = "is_ht"
    CREATE_AT = "create_at"

    ACCURACY = "accuracy"
    SCORE = "score"
    MAX_COMBO = "max_combo"
    COUNT_50 = "count_50"
    COUNT_100 = "count_100"
    COUNT_300 = "count_300"
    COUNT_geki = "count_geki"
    COUNT_katu = "count_katu"
    COUNT_miss = "count_miss"
    PP = "pp"
    PP_WEIGHT = "pp_weight"

    GAME_MODE = "game_mode"
    CS = "cs"

    CUSTOM_ACCURACY = "custom_accuracy"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=Score.TABLE_NAME, columns={
            Score.BEATMAP_ID: "INTEGER NOT NULL",
            Score.USER_ID: "INTEGER NOT NULL",
            Score.SCORE_ID: "INTEGER NOT NULL",
            Score.SPEED: "INTEGER",
            Score.IS_DT: "BOOLEAN",
            Score.IS_HR: "BOOLEAN",
            Score.IS_HD: "BOOLEAN",
            Score.IS_FL: "BOOLEAN",
            Score.IS_EZ: "BOOLEAN",
            Score.IS_MR: "BOOLEAN",
            Score.IS_HT: "BOOLEAN",
            Score.CREATE_AT: "INTEGER",  # unix timestamp (seconds)

            Score.ACCURACY: "REAL",
            Score.SCORE: "INTEGER",
            Score.MAX_COMBO: "INTEGER",
            Score.COUNT_geki: "INTEGER",
            Score.COUNT_300: "INTEGER",
            Score.COUNT_katu: "INTEGER",
            Score.COUNT_100: "INTEGER",
            Score.COUNT_50: "INTEGER",
            Score.COUNT_miss: "INTEGER",
            Score.PP: "REAL",
            Score.PP_WEIGHT: "REAL"
        }, primary_keys=[Score.BEATMAP_ID, Score.SPEED, Score.USER_ID])
        # repository.create_index(conn, index_name='index_user', table_name=Score.TABLE_NAME,
        #                         columns=[Score.USER_ID])


class Beatmap:
    TABLE_NAME = "Beatmap"
    PRIMARY_KEYS = ["id"]

    ID = "id"
    SET_ID = "set_id"
    NAME = "name"
    VERSION = "version"
    GAME_MODE = "game_mode"
    CREATOR = "creator"
    LAST_UPDATED = "last_updated"

    LENGTH = "length"
    BPM = "bpm"
    CS = "cs"
    HP = "hp"
    OD = "od"
    AR = "ar"
    STAR = "star" # @deprecated: use mod_star instead
    DT_STAR = "dt_star" # @deprecated: use mod_star instead
    HT_STAR = "ht_star" # @deprecated: use mod_star instead
    MOD_STAR = "mod_star"

    SUM_SCORES = "sum_scores"

    PASS_COUNT = "pass_count"
    PLAY_COUNT = "play_count"
    COUNT_CIRCLES = "count_circles"
    COUNT_SLIDERS = "count_sliders"
    COUNT_SPINNERS = "count_spinners"

    MOD_MAX_PP = "mod_max_pp"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=Beatmap.TABLE_NAME, columns={
            Beatmap.ID: "INTEGER NOT NULL",
            Beatmap.SET_ID: "INTEGER",
            Beatmap.NAME: "TEXT",
            Beatmap.VERSION: "TEXT",
            Beatmap.GAME_MODE: "TEXT",
            Beatmap.CREATOR: "TEXT",
            Beatmap.SUM_SCORES: "INTEGER DEFAULT 0",

            Beatmap.LENGTH: "REAL",
            Beatmap.BPM: "REAL",
            Beatmap.CS: "REAL",
            Beatmap.HP: "REAL",
            Beatmap.OD: "REAL",
            Beatmap.AR: "REAL",
            Beatmap.STAR: "REAL",
            Beatmap.DT_STAR: "REAL DEFAULT 0.0",
            Beatmap.HT_STAR: "REAL DEFAULT 0.0",
            Beatmap.MOD_STAR: "TEXT DEFAULT NULL",
            Beatmap.PASS_COUNT: "INTEGER",
            Beatmap.PLAY_COUNT: "INTEGER",
            Beatmap.COUNT_CIRCLES: "INTEGER",
            Beatmap.COUNT_SLIDERS: "INTEGER",
            Beatmap.COUNT_SPINNERS: "INTEGER",

            Beatmap.MOD_MAX_PP: "TEXT DEFAULT NULL"

        }, primary_keys=Beatmap.PRIMARY_KEYS)
        repository.ensure_column(conn, table_name=Beatmap.TABLE_NAME, name_type_default=[
            (Beatmap.MOD_STAR, "TEXT DEFAULT NULL", None),
            (Beatmap.MOD_MAX_PP, "TEXT DEFAULT NULL", None),
            (Beatmap.LAST_UPDATED, "INTEGER DEFAULT 0", None)
        ])

    @staticmethod
    def construct_where(bid):
        return {
            Beatmap.TABLE_NAME + "." + Beatmap.ID: bid
        }


class Task:
    TABLE_NAME = "Task"

    TASK_NAME = "task_name"
    TASK_STATE = "task_state"
    TASK_TIME = "task_time"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=Task.TABLE_NAME, columns={
            Task.TASK_NAME: "TEXT",
            Task.TASK_STATE: "INTEGER",
            Task.TASK_TIME: "INTEGER"
        }, primary_keys=[Task.TASK_NAME])


class Meta:
    TABLE_NAME = "Meta"

    KEY = "key"
    VALUE = "value"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=Meta.TABLE_NAME, columns={
            Meta.KEY: "TEXT",
            Meta.VALUE: "TEXT"
        }, primary_keys=[Meta.KEY])

    @staticmethod
    def get(conn: sqlite3.Connection, key: str, default: str):
        result = repository.select_first(conn, Meta.TABLE_NAME, [Meta.VALUE], where={Meta.KEY: key})
        if result is None:
            return default
        return result[0]

    @staticmethod
    def save(conn: sqlite3.Connection, key: str, value: str):
        repository.insert_or_replace(conn, Meta.TABLE_NAME, contents=[
            {
                Meta.KEY: key,
                Meta.VALUE: value
            }
        ])


def measure_time(fun):
    def decorate(*c, **d):
        # start_time = time.time()
        result = fun(*c, **d)
        # end_time = time.time()
        # print(fun.__name__ + ":", end_time - start_time, "s")
        return result

    return decorate


class NetworkConfig:

    @staticmethod
    def from_config(config_file):
        config = json.load(open(config_file))
        print("Load config:", config)
        return NetworkConfig(config)

    def __init__(self, data_dict=None):
        if data_dict is None:
            data_dict = {}
        self.game_mode = data_dict.get('game_mode', 'mania')
        self.embedding_size: int = data_dict.get('embedding_size', 15)
        self.embedding_size_beyas = (self.embedding_size - 1) ** 2 + 1
        self.pp_weight_clip = data_dict.get('pp_weight_clip', 10)
        self.pass_band_width = data_dict.get('pass_band_width', 1)
        self.pass_basic_weight_played = data_dict.get('pass_basic_weight_played', 0.7)
        self.pass_power = data_dict.get('pass_power', 0.8)
        self.ball_tree_path = data_dict.get('ball_tree_path', 'ball-tree.pkl')

    def get_embedding_names(self, name, is_sigma=False, is_alpha=False):
        if is_sigma:
            return name + "_sigma"
        if is_alpha:
            return name + "_alpha"
        return [name + "_" + str(i) for i in range(self.embedding_size)]


class EmbeddingData:

    def __init__(self, key_to_embed_id: pd.Series = None, embeddings: list = None,
                 sigma: np.ndarray = None, alpha: np.ndarray = None):
        """
        :param key_to_embed_id: a mapping {key -> i}, i is the index in embeddings (1 <= i <= N)
        :param embeddings: shape = (1, N, E)
        :param sigma: shape = (N, (E-1)^2)
        :param alpha: shape = (N, )
        """
        self.key_to_embed_id = key_to_embed_id
        self.embeddings = embeddings
        self.sigma = sigma
        self.alpha = alpha


class ScoreModelWeight:

    def __init__(self, user_embedding: EmbeddingData = EmbeddingData(),
                 beatmap_embedding: EmbeddingData = EmbeddingData(),
                 mod_embedding: EmbeddingData = EmbeddingData(),
                 feature_weights=None):
        if feature_weights is None:
            feature_weights = {}
        self.user_embedding = user_embedding
        self.beatmap_embedding = beatmap_embedding
        self.mod_embedding = mod_embedding
        self.feature_weights = feature_weights


class BestPerformance:

    def __init__(self, max_length=100):
        self.data = {}  # id -> speed, score, pp, star, scoreid, cs
        self.pp_order_list = []
        self.id_of_pp_order_list = []
        self.max_length = max_length

    def update(self, id, speed, score, pp, star, filter_duplicate=True, score_id=None, cs=None):
        if filter_duplicate and id in self.data:
            remove_index = self.id_of_pp_order_list.index(id)
            del self.id_of_pp_order_list[remove_index]
            del self.pp_order_list[remove_index]

        self.data[id] = (speed, score, pp, star, score_id, cs)
        index = bisect.bisect_left(self.pp_order_list, pp)
        self.pp_order_list.insert(index, pp)
        self.id_of_pp_order_list.insert(index, id)

        if self.max_length is not None and len(self.pp_order_list) > self.max_length:
            remove_id = self.id_of_pp_order_list[0]
            del self.id_of_pp_order_list[0]
            del self.pp_order_list[0]
            del self.data[remove_id]

    def get_pp(self):
        pp_len = len(self.pp_order_list)
        return np.sum(np.asarray(self.pp_order_list) * np.power(0.95, np.arange(0, pp_len))[::-1])

    def get_score_pp(self, bid):
        if bid in self.data:
            return self.data[bid][0], self.data[bid][1], self.data[bid][2], self.data[bid][3], \
                   self.data[bid][4]
        return None, None, None, None, None

    def copy(self):
        new = BestPerformance(self.max_length)
        new.data = self.data.copy()
        new.id_of_pp_order_list = self.id_of_pp_order_list.copy()
        new.pp_order_list = self.pp_order_list.copy()
        return new


def get_pass_model_path(speed, result_path="result", is_training=False):
    path = os.path.join(get_pass_model_dir(speed, result_path), f"model")
    if is_training:
        path += "_train"
    return path


def get_pass_model_dir(speed, result_path="result"):
    dir_name = os.path.join(result_path, f"pass_xgboost_{speed}")
    os.makedirs(dir_name, exist_ok=True)
    return dir_name
