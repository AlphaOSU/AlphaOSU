from posixpath import dirname
from data import repository
import sqlite3
import time
import os
import pandas as pd
import numpy as np
from dateutil import parser


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

    MOD = 'mod'
    SPEED = 'speed'
    IS_ACC = "is_acc"

    EMBEDDING = "mod_embedding"

    @staticmethod
    def create(conn: sqlite3.Connection):
        repository.create_table(conn, table_name=ModEmbedding.TABLE_NAME, columns={
            ModEmbedding.MOD: "TEXT NOT NULL",
            ModEmbedding.SPEED: "INTEGER NOT NULL",
            ModEmbedding.IS_ACC: "BOOLEAN NOT NULL",
        }, primary_keys=ModEmbedding.PRIMARY_KEYS)

    @staticmethod
    def construct_where(mod):
        return {
            ModEmbedding.MOD: mod
        }


class Beatmap:
    TABLE_NAME = "Beatmap"
    PRIMARY_KEYS = ["id"]

    ID = "id"
    SET_ID = "set_id"
    NAME = "name"
    VERSION = "version"
    GAME_MODE = "game_mode"
    CREATOR = "creator"

    LENGTH = "length"
    BPM = "bpm"
    CS = "cs"
    HP = "hp"
    OD = "od"
    AR = "ar"
    STAR = "star"
    DT_STAR = "dt_star"
    HT_STAR = "ht_star"
    SUM_SCORES = "sum_scores"

    PASS_COUNT = "pass_count"
    PLAY_COUNT = "play_count"
    COUNT_CIRCLES = "count_circles"
    COUNT_SLIDERS = "count_sliders"
    COUNT_SPINNERS = "count_spinners"

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
            Beatmap.PASS_COUNT: "INTEGER",
            Beatmap.PLAY_COUNT: "INTEGER",
            Beatmap.COUNT_CIRCLES: "INTEGER",
            Beatmap.COUNT_SLIDERS: "INTEGER",
            Beatmap.COUNT_SPINNERS: "INTEGER",

        }, primary_keys=Beatmap.PRIMARY_KEYS)

    @staticmethod
    def construct_where(bid):
        return {
            Beatmap.TABLE_NAME + "." + Beatmap.ID: bid
        }


BEATMAP_FEATURES = [Beatmap.CS, Beatmap.OD,  # Beatmap.AR, Beatmap.HP,
                    Beatmap.STAR, Beatmap.DT_STAR, Beatmap.HT_STAR,
                    # Beatmap.PASS_COUNT, Beatmap.PLAY_COUNT,
                    Beatmap.BPM, Beatmap.LENGTH,
                    Beatmap.COUNT_CIRCLES, Beatmap.COUNT_SLIDERS]  # , Beatmap.COUNT_SPINNERS]


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


class CannotPass:
    TABLE_NAME = "CannotPass"

    USER_ID = "user_id"
    USER_VARIANT = "variant"
    USER_GAME_MODE = "game_mode"

    BEATMAP_ID = "beatmap_id"

    SPEED = "speed"

    SCORE = "score"
    PP = "pp"
    PP_RANK = "pp_rank"
    PASS = "pass"

    @staticmethod
    def create(conn: sqlite3.Connection, table_name=TABLE_NAME):
        repository.create_table(conn, table_name=table_name, columns={
            CannotPass.USER_ID: "INTEGER NOT NULL",
            CannotPass.USER_VARIANT: "TEXT NOT NULL",
            CannotPass.USER_GAME_MODE: "TEXT NOT NULL",
            CannotPass.BEATMAP_ID: "INTEGER NOT NULL",
            CannotPass.SPEED: "INTEGER NOT NULL",
            CannotPass.SCORE: "INTEGER",
            CannotPass.PP: "INTEGER",
            CannotPass.PP_RANK: "INTEGER",
            CannotPass.PASS: "BOOL",
        }, primary_keys=[CannotPass.USER_ID, CannotPass.USER_VARIANT, CannotPass.USER_GAME_MODE,
                         CannotPass.BEATMAP_ID, CannotPass.SPEED])


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


def parse_beatmap_data(beatmap_data, beatmapset_data, conn: sqlite3.Connection):
    if beatmap_data['status'] != 'ranked':
        return None
    if beatmap_data['convert']:
        return None
    old_sum = repository.select_first(conn, Beatmap.TABLE_NAME, project=[Beatmap.SUM_SCORES],
                                      where={Beatmap.ID: beatmap_data['id']})
    if old_sum is None:
        old_sum = 0
    else:
        old_sum = old_sum[0]
    result = {
        Beatmap.ID: beatmap_data['id'],
        Beatmap.SET_ID: beatmap_data['beatmapset_id'],
        Beatmap.NAME: beatmapset_data['title'],
        Beatmap.VERSION: beatmap_data['version'],
        Beatmap.GAME_MODE: beatmap_data['mode'],
        Beatmap.CREATOR: beatmapset_data['creator'],
        Beatmap.SUM_SCORES: old_sum,

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
        Beatmap.PLAY_COUNT: beatmap_data['playcount']
    }
    return result


def parse_score_data(scores, beatmap_id, is_in_top_scores):
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


def measure_time(fun):
    def decorate(*c, **d):
        # start_time = time.time()
        result = fun(*c, **d)
        # end_time = time.time()
        # print(fun.__name__ + ":", end_time - start_time, "s")
        return result

    return decorate


class NetworkConfig:

    def __init__(self, data_dict=None):
        if data_dict is None:
            data_dict = {
                "game_mode": "mania",
                "embedding_size": 15,
                "embedding_bias": True,
                "epoch": 200,
                "early_stop_patient": 5,
                "dense_units": [256, 128, 64],
                "dropout_rate": 0.0,
                "batch_size": 2048,
                "loss_type": "pp_mse",
                "optimizier": "adam",
                "lr": 1e-3,
                "test_ratio": 0.98,
                "l2": 0,
                "pretrain_epoch": 100,
                "save_results": True,
                "pp_weight_clip": 10,
                "max_recall": 500
            }
        self.embedding_size: int = data_dict['embedding_size']
        self.embedding_size_beyas = (self.embedding_size - 1) ** 2 + 1

        self.embedding_bias = data_dict['embedding_bias']
        self.game_mode = data_dict['game_mode']
        self.dense_units = data_dict['dense_units']
        self.dropout_rate = data_dict['dropout_rate']
        self.batch_size = data_dict['batch_size']
        self.loss_type = data_dict['loss_type']
        self.optimizier = data_dict['optimizier']
        self.lr = data_dict['lr']
        self.epoch = data_dict['epoch']
        self.test_ratio = data_dict['test_ratio']
        self.l2 = data_dict['l2']
        self.pretrain_epoch = data_dict['pretrain_epoch']
        self.load_beyas = True
        self.save_results = data_dict['save_results']
        self.early_stop_patient = data_dict['early_stop_patient']
        self.pp_weight_clip = data_dict['pp_weight_clip']
        self.max_recall = data_dict['max_recall']

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
        :param key_to_embed_id: {key -> i}, i \in [N]
        :param embeddings: (1, N, E)
        :param sigma: (N, (E-1)^2)
        :param alpha: (N, )
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


import bisect


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
            return self.data[bid][0], self.data[bid][1], self.data[bid][2], self.data[bid][4]
        return None, None, None, None

    def copy(self):
        new = BestPerformance(self.max_length)
        new.data = self.data.copy()
        new.id_of_pp_order_list = self.id_of_pp_order_list.copy()
        new.pp_order_list = self.pp_order_list.copy()
        return new

def get_pass_model_path(speed, result_path="result"):
    return os.path.join(get_pass_model_dir(speed, result_path), f"model")

def get_pass_model_dir(speed, result_path="result"):
    dir_name = os.path.join(result_path, f"pass_xgboost_{speed}")
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

if __name__ == "__main__":
    print(', '.join(NetworkConfig().get_pass_features()))
