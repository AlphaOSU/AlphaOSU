from data.model import *


class PPRuleSet():

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection):
        self.config = config
        self.connection = connection

    def map_beatmap_name(self, name, version):
        return f"{name} - {version}"

    def generate_recall_table(self, uid, key_count, beatmap_ids,
                              max_star=None,
                              min_star=0, required_mods=None, min_pp=None): raise NotImplemented

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance): raise NotImplemented

    def map_mod(self, mod):
        return mod

    def user_bp(self, uid) -> BestPerformance: raise NotImplemented
