from abc import abstractmethod, ABCMeta
from collections import defaultdict

from scipy import stats, integrate

import osu_utils
import time
import train_pass_kernel
import json
from data.model import *
from recommend.pp_ruleset import PPRuleSet

class ManiaPP(PPRuleSet):

    def post_process_query_param(self, query_params: dict, required_mods: list): raise NotImplemented

    def train_to_real(self, x): raise NotImplemented

    def real_to_train(self, x): raise NotImplemented

    def pp(self, x, star, od, count): raise NotImplemented

    def db_column(self): raise NotImplemented

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance):
        user_true_pp = user_bp.get_pp()

        def score_with_prob(x, pred_score, pred_score_std):
            # print(x)
            prob = np.exp(-(x - pred_score) ** 2 / (2 * pred_score_std ** 2)) / (
                    pred_score_std * np.sqrt(2 * np.pi))
            x_real = self.train_to_real(x)
            return prob * x_real

        # stage 2: rank precisely with pass rate, break prob, pp recom, etc.
        probable_scores, probable_pps, break_probs, \
        pp_gains, true_scores, true_pps, pass_probs, true_score_ids = [], [], [], [], [], [], [], []
        true_speeds = []
        pass_feature_list = defaultdict(lambda: [])
        pass_feature_list_index = defaultdict(lambda: [])

        # estimate pp incre
        for i, row in enumerate(data.itertuples(name=None)):
            (bid, mod), star, pred_score, _, _, od, count, speed, variant, _, _, _ = row
            pass_feature_list[f"{speed}/{variant}"].append(bid)
            pass_feature_list_index[f"{speed}/{variant}"].append(i)
            # break prob
            true_speed, true_score, true_pp, true_star, true_score_id = user_bp.get_score_pp(bid)
            if true_score is not None:
                true_pp = self.pp(true_score, true_star, od, count)
            if true_score is None:
                break_prob = 1.0
                probable_score = pred_score
            else:
                std_result = osu_utils.predict_score_std(self.connection, uid,
                                                         variant, self.config, bid,
                                                         self.map_mod(mod))
                if std_result is None or true_speed != speed:
                    probable_score = pred_score
                    break_prob = 1.0
                else:
                    score_train, score_std_train = std_result
                    true_score_train = self.real_to_train(true_score)
                    break_prob = 1 - stats.norm.cdf(true_score_train, loc=score_train,
                                                    scale=score_std_train)
                    probable_score = \
                        integrate.quad(score_with_prob, a=true_score_train, b=np.inf,
                                       args=(score_train, score_std_train),
                                       epsabs=1.0)[0] / break_prob
            true_scores.append(true_score)
            true_pps.append(true_pp)
            true_speeds.append(true_speed)
            probable_scores.append(probable_score)
            break_probs.append(round(break_prob, 6))
            true_score_ids.append(int(true_score_id) if true_score_id is not None else None)
            # pp gain
            if probable_score is not None:
                pp_gain_beatmap = self.pp(probable_score, star, od, count)
                pp_gain_beatmap = round(pp_gain_beatmap, 3)
                probable_pps.append(pp_gain_beatmap)
                user_bp2 = user_bp.copy()
                user_bp2.update(bid, 0, probable_score, pp_gain_beatmap, 0)
                pp_gain = user_bp2.get_pp() - user_true_pp
                pp_gain = max(0, pp_gain)
                pp_gains.append(pp_gain)
            else:
                probable_pps.append(None)
                pp_gains.append(None)

        # calculate pass prob
        pass_probs = np.ones(len(probable_scores))
        for key in pass_feature_list:
            speed, variant = key.split("/")
            probs = train_pass_kernel.estimate_pass_probability(uid, variant,
                                                                pass_feature_list[key], speed,
                                                                self.config, self.connection)
            pass_probs[pass_feature_list_index[key]] = probs

        data['pred_score (breaking)'] = probable_scores
        data['pred_pp (breaking)'] = probable_pps
        data['break_prob'] = break_probs
        data["pass_prob"] = pass_probs
        data["pp_gain (breaking)"] = pp_gains
        data['true_score'] = true_scores
        data['true_pp'] = true_pps
        data['true_speed'] = true_speeds
        data['true_score_id'] = true_score_ids
        data['pp_gain_expect'] = data['pp_gain (breaking)'] * data['break_prob'] * data['pass_prob']

        data = pd.DataFrame(data,
                            columns=['name', 'star',
                                     'true_score', 'true_pp',
                                     'pred_score', 'break_prob',
                                     'pred_score (breaking)', 'pred_pp (breaking)',
                                     'pp_gain (breaking)', 'pass_prob', 'cs', 'set_id',
                                     'valid_count', 'true_speed',
                                     'pp_gain_expect', 'pred_pp', 'true_score_id'])
        data.sort_values(by="pp_gain_expect", ascending=False, inplace=True)
        return data


    def generate_recall_table(self, uid, key_count, beatmap_ids,
                              max_star=None,
                              min_star=0, required_mods=None, min_pp=None):
        base_projection = [
            BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.BEATMAP_ID,
            ModEmbedding.SPEED, UserEmbedding.VARIANT,
            # for pp
            Beatmap.OD,
            Beatmap.COUNT_SLIDERS,
            Beatmap.COUNT_CIRCLES,
            Beatmap.HT_STAR, Beatmap.STAR, Beatmap.DT_STAR,
            Beatmap.CS,
            # for name
            Beatmap.VERSION, Beatmap.NAME, Beatmap.SET_ID,
            # for validation
            BeatmapEmbedding.COUNT_HT, BeatmapEmbedding.COUNT_NM, BeatmapEmbedding.COUNT_DT
        ]
        data_list = []
        cursor = []
        query_params = {
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: ('=', uid),
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: ('=', "mania"),
        }
        self.post_process_query_param(query_params, required_mods)
        if 4 in key_count:
            query_params[UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT] = ('=', '4k')
            query_params[Beatmap.CS] = ('=', 4)
            cursor += list(
                osu_utils.predict_score(self.connection, query_params, self.config.embedding_size,
                                        projection=base_projection))
        if 7 in key_count:
            query_params[UserEmbedding.TABLE_NAME + "." + UserEmbedding.VARIANT] = ('=', '7k')
            query_params[Beatmap.CS] = ('=', 7)
            cursor += list(
                osu_utils.predict_score(self.connection, query_params, self.config.embedding_size,
                                        projection=base_projection))
        for x in cursor:
            score, bid, speed, variant, od, count1, count2, \
            star_ht, star_nm, star_dt, cs, version, name, set_id, \
            count_ht, count_nm, count_dt, mod_max_pp, mod_star = x[:len(base_projection) + 1]
            # print(int(bid))
            if len(beatmap_ids) != 0 and int(bid) not in beatmap_ids:
                continue
            if speed == 1:
                star = star_dt
                mod = 'DT'
                count = count_dt
            elif speed == 0:
                star = star_nm
                mod = 'NM'
                count = count_nm
            else:
                star = star_ht
                mod = 'HT'
                count = count_ht
            if max_star is not None:
                if star > max_star:
                    continue
            if star < min_star:
                continue
            data_list.append([bid, mod, star, score, 0, self.map,
                              od, count1 + count2, speed, variant, cs, set_id, count])
        data = pd.DataFrame(data_list,
                            columns=['id', "mod", "star", "pred_score", "pred_pp", "name",
                                     'od', 'count', 'speed', 'variant', 'cs', 'set_id',
                                     'valid_count'])
        data['pred_score'] = self.train_to_real(data['pred_score'].to_numpy())
        data['pred_pp'] = self.pp(data['pred_score'].to_numpy(),
                                  data['star'].to_numpy(),
                                  data['od'].to_numpy(),
                                  data['count'].to_numpy())
        return data



class ManiaPPV3(ManiaPP):

    def post_process_query_param(self, query_params: dict, required_mods: list):
        query_params[ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC] = ('=', '0')

    def train_to_real(self, x):
        return np.round(
            osu_utils.map_osu_score(x, real_to_train=False)
        ).astype(int)

    def real_to_train(self, x):
        return osu_utils.map_osu_score(x, real_to_train=True)

    def pp(self, x, star, od, count):
        return osu_utils.mania_pp(x, od, star, count)

    def user_bp(self, uid):
        return osu_utils.get_user_bp(self.connection, uid, self.config, is_acc=False)

    def db_column(self):
        return Score.SCORE


class ManiaPPV4(ManiaPP):

    def post_process_query_param(self, query_params: dict, required_mods: list):
        query_params[ModEmbedding.TABLE_NAME + "." + ModEmbedding.IS_ACC] = ('=', '1')

    def train_to_real(self, x):
        return osu_utils.map_osu_acc(x, real_to_train=False)

    def real_to_train(self, x):
        return osu_utils.map_osu_acc(x, real_to_train=True)

    def pp(self, x, star, od, count):
        return osu_utils.mania_pp_v4(x, star, count)

    def user_bp(self, uid):
        return osu_utils.get_user_bp(self.connection, uid, self.config, is_acc=True)

    def map_mod(self, mod):
        return mod + "-ACC"

    def db_column(self):
        return Score.CUSTOM_ACCURACY
