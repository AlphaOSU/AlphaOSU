from data.model import *
import math

class PassFeature:

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection):
        self.config = config
        self.cache_user = {}
        self.cache_beatmap = {}
        self.cache_beatmap_embedding = {}
        self.cache_mod = {}
        self.cache_norm = {}
        self.connection = connection
        self.connection.create_function("log", 1, math.log1p)

    def fetch_feature(self, table_name, projections, where):
        cursor = repository.select(self.connection, table_name,
                                   project=projections,
                                   where=where)
        names = repository.get_columns_by_cursor(cursor)
        features = np.asarray(cursor.fetchone(), dtype=np.float32)

        if table_name in self.cache_norm:
            avg, std = self.cache_norm[table_name]
        else:
            # get avg and std
            avg_project = [f"AVG({x})" for x in projections]
            avg = repository.select_first(self.connection, table_name, avg_project)
            avg = np.asarray(avg, dtype=np.float32)
            var_project = [f"AVG(({x} - {avg[i]}) * ({x} - {avg[i]}))" for i, x in
                           enumerate(projections)]
            var = repository.select_first(self.connection, table_name, var_project)
            std = np.sqrt(np.asarray(var, dtype=np.float32))
            self.cache_norm[table_name] = (avg, std)

        features_norm = (features - avg) / std
        return features_norm, names

    def get_user_features(self, uid, variant):
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
        projections = ["Beatmap.CS", "Beatmap.OD", "Beatmap.HP", "Beatmap.HT_STAR", "Beatmap.STAR",
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


    def get_mod_features(self, mod):
        if mod in self.cache_mod:
            return self.cache_mod[mod]
        projections = self.config.get_embedding_names(ModEmbedding.EMBEDDING)
        projections.append(self.config.get_embedding_names(ModEmbedding.EMBEDDING, is_alpha=True))
        result = self.fetch_feature(ModEmbedding.TABLE_NAME, projections,
                                    ModEmbedding.construct_where(mod=mod))
        self.cache_mod[mod] = result
        return result

    def get_pass_features(self, uid, variant, bid, mod, return_feature_name=False, is_predicting=False):
        features = []
        names = []
        f, n = self.get_user_features(uid, variant)
        features.extend(f)
        names.extend(n)
        f, n = self.get_beatmap_features(bid)
        if is_predicting:
            f[n.index("log(Beatmap.PLAY_COUNT)")] = 1000
        features.extend(f)
        names.extend(n)
        f, n = self.get_beatmap_embedding_features(bid)
        features.extend(f)
        names.extend(n)
        f, n = self.get_mod_features(mod)
        features.extend(f)
        names.extend(n)

        features = np.asarray(features, dtype=np.float32)
        if return_feature_name:
            return features, names
        else:
            return features, None
