from data.model import *
import time
from collections import OrderedDict

class Timer:
    def __init__(self, profile_dict, name):
        self.profile_dict = profile_dict
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.profile_dict[self.name] = duration

class PPRuleSet():

    def __init__(self, config: NetworkConfig, connection: sqlite3.Connection):
        self.config = config
        self.connection = connection
        self.profile = OrderedDict()

    def map_beatmap_name(self, name, version):
        return f"{name} - {version}"

    def generate_recall_table(self, uid, key_count, beatmap_ids,
                              max_star=None,
                              min_star=0, required_mods=None, min_pp=None): raise NotImplemented

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance): raise NotImplemented

    def map_mod(self, mod):
        return mod

    def user_bp(self, uid) -> BestPerformance: raise NotImplemented

    def timing(self, name):
        return Timer(self.profile, name)

    def export_profile(self):
        content = "====================\n"
        for name, duration in self.profile.items():
            content += (f"{name}: {duration:.3f} s\n")
        content += "===================="
        return content