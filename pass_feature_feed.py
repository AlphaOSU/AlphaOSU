from data.model import *
import math

class PassFeatureFeed:

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection):
        self.config = config
        self.cache_user = {}
        self.cache_beatmap = {}
        self.cache_beatmap_embedding = {}
        self.cache_mod = {}
        self.cache_norm = {}
        self.connection = connection
        self.connection.create_function("log", 1, math.log1p)
        self.min_pc = math.log1p(1_000)

    def fetch_feature(self, table_name, projections, where):
        cursor = repository.select(self.connection, table_name,
                                   project=projections,
                                   where=where)
        names = repository.get_columns_by_cursor(cursor)
        features = np.asarray(cursor.fetchone(), dtype=np.float32)

        return features, names

    def get_user_embedding_features(self, uid, variant):
        if (uid, variant) in self.cache_user:
            return self.cache_user[(uid, variant)]
        # user features
        projections = self.config.get_embedding_names(UserEmbedding.EMBEDDING)
        projections.append(self.config.get_embedding_names(UserEmbedding.EMBEDDING, is_alpha=True))
        result = self.fetch_feature(UserEmbedding.TABLE_NAME,
                                    projections,
                                    UserEmbedding.construct_where(uid=uid,
                                                                  game_mode=self.config.game_mode,
                                                                  variant=variant))
        self.cache_user[(uid, variant)] = result
        return result

    def get_beatmap_features(self, bid):
        if bid in self.cache_beatmap:
            return self.cache_beatmap[bid]
        projections = ["Beatmap.CS", "Beatmap.HT_STAR", "Beatmap.STAR", "Beatmap.PASS_COUNT / (Beatmap.PLAY_COUNT + 1.0)",
                       "Beatmap.DT_STAR", "log(Beatmap.PLAY_COUNT)",
                       "log(Beatmap.LENGTH)", "log(Beatmap.COUNT_CIRCLES)", "log(Beatmap.COUNT_SLIDERS)"]
        where = Beatmap.construct_where(bid)
        result = self.fetch_feature(Beatmap.TABLE_NAME,
                                    projections, where)
        self.cache_beatmap[bid] = result
        return result

    def get_beatmap_embedding_features(self, bid):
        if bid in self.cache_beatmap_embedding:
            return self.cache_beatmap_embedding[bid]
        projections = list(map(lambda x: "BeatmapEmbedding." + x,
                              self.config.get_embedding_names(BeatmapEmbedding.ITEM_EMBEDDING)))
        projections.append(f"BeatmapEmbedding."
                           f"{self.config.get_embedding_names(BeatmapEmbedding.ITEM_EMBEDDING, is_alpha=True)}")
        where = BeatmapEmbedding.construct_where(bid)
        result = self.fetch_feature(BeatmapEmbedding.TABLE_NAME,
                                    projections, where)
        self.cache_beatmap_embedding[bid] = result
        return result

    def get_pass_features(self, uid, variant, bid, return_feature_name=False, is_predicting=False):
        features = []
        names = []

        f_u, n_u = self.get_user_embedding_features(uid, variant)
        features.extend(f_u)
        names.extend(n_u)

        f, n = self.get_beatmap_features(bid)
        if is_predicting:
            pc_index = n.index("log(Beatmap.PLAY_COUNT)")
            if f[pc_index] < self.min_pc:
                f[pc_index] = self.min_pc
        features.extend(f)
        names.extend(n)

        f_b, n_b = self.get_beatmap_embedding_features(bid)
        features.extend(f_b)
        names.extend(n_b)

        combine_embedding_features = f_u[:-1] * f_b[:-1]
        combine_embedding_names = [f"combine_{i}" for i in range(len(combine_embedding_features))]
        features.extend(combine_embedding_features)
        features.append(np.sum(combine_embedding_features))
        names.extend(combine_embedding_names)
        names.append("score")

        features = np.asarray(features, dtype=np.float32)
        if return_feature_name:
            return features, names
        else:
            return features, None
