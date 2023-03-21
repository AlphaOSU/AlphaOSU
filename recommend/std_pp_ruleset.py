from collections import defaultdict

import json
from collections import defaultdict

from scipy import stats, integrate

import osu_utils
import train_pass_kernel
from data.model import *
from recommend.pp_ruleset import PPRuleSet


class STDPP(PPRuleSet):

    def user_bp(self, uid):
        return osu_utils.get_user_bp(self.connection, uid, self.config)

    def rank(self, uid, data: pd.DataFrame, user_bp: BestPerformance):
        user_true_pp = user_bp.get_pp()

        def pp_with_prob(x, pred_pp, pred_pp_std, max_pp):
            # print(x)
            prob = np.exp(-(x - pred_pp) ** 2 / (2 * pred_pp_std ** 2)) / (
                    pred_pp_std * np.sqrt(2 * np.pi))
            x_real =  osu_utils.map_osu_pp(x, real_to_train=False, max_pp=max_pp)
            return prob * x_real

        probable_pps, break_probs, \
        pp_gains, true_pps, pass_probs, true_score_ids = [], [], [], [], [], []
        true_mods = []
        true_scores = []
        stars = []
        pass_feature_list = defaultdict(lambda: [])
        pass_feature_list_index = defaultdict(lambda: [])
        playable_stars = []
        dt_star = osu_utils.MOD_INT_MAPPING["DT"]
        hr_star = osu_utils.MOD_INT_MAPPING["HR"]

        # estimate pp incre
        for i, row in enumerate(data.itertuples(name=None)):
            (bid, mod), star, max_pp, _, pred_pp, _, speed, _, _ = row
            is_hr = int(mod) & hr_star != 0
            pass_feature_key = f"{speed}/{int(is_hr)}"
            pass_feature_list[pass_feature_key].append(bid)
            pass_feature_list_index[pass_feature_key].append(i)
            # break prob
            true_mod, true_score, true_pp, true_star, true_score_id = user_bp.get_score_pp(bid)
            if true_pp is None:
                break_prob = 1.0
                probable_pp = pred_pp
            else:
                if (int(mod) & dt_star == int(true_mod) & dt_star) and (int(mod) & hr_star == int(true_mod) & hr_star):
                    playable_stars.append(true_star)
                std_result = osu_utils.predict_score_std(self.connection, uid,
                                                         "", self.config, bid,
                                                         self.map_mod(mod))
                if std_result is None:
                    probable_pp = pred_pp
                    break_prob = 1.0
                else:
                    pp_train, pp_std_train = std_result
                    pp_std_train = pp_std_train * 3
                    true_pp_train = osu_utils.map_osu_pp(true_pp, real_to_train=True, max_pp=max_pp)
                    break_prob = 1 - stats.norm.cdf(true_pp_train, loc=pp_train,
                                                    scale=pp_std_train)
                    probable_pp = \
                        integrate.quad(pp_with_prob, a=true_pp_train, b=np.inf,
                                       args=(pp_train, pp_std_train, max_pp),
                                       epsabs=1.0)[0] / break_prob
                    probable_pp = probable_pp * 0.5 + pred_pp * 0.5
            true_pps.append(true_pp)
            probable_pps.append(probable_pp)
            break_probs.append(round(break_prob, 6))
            stars.append(star)
            true_mods.append(true_mod)
            true_scores.append(true_score)
            true_score_ids.append(int(true_score_id) if true_score_id is not None else None)
            # pp gain
            if probable_pp is not None:
                user_bp2 = user_bp.copy()
                user_bp2.update(bid, 0, probable_pp, probable_pp, 0)
                pp_gain = user_bp2.get_pp() - user_true_pp
                pp_gain = max(0, pp_gain)
                pp_gains.append(pp_gain)
            else:
                probable_pps.append(None)
                pp_gains.append(None)

        # calculate pass prob
        pass_probs = np.ones(len(probable_pps))
        for key in pass_feature_list:
            speed, is_hr = key.split("/")
            speed = int(speed)
            is_hr = int(is_hr) == 1
            probs = train_pass_kernel.estimate_pass_probability(int(uid), "",
                                                                pass_feature_list[key], speed,
                                                                self.config, self.connection,
                                                                is_hr=is_hr)
            pass_probs[pass_feature_list_index[key]] = probs

        if len(playable_stars) > 0:
            stars = np.asarray(stars)
            max_playable_stars = np.percentile(playable_stars, q=90)
            print(max_playable_stars)
            # if star < max_playable_star, add 20% pass prob.
            # if max_playable_star < star < max_playable_star + 0.2, add a decaying pass prob.
            pass_complement = 1 - np.clip((stars - max_playable_stars) / 0.2, 0, 1)
            pass_probs = pass_complement * 0.05 + pass_probs * 0.95

        data['pred_pp (breaking)'] = probable_pps
        data['break_prob'] = break_probs
        data["pass_prob"] = pass_probs
        data["pp_gain (breaking)"] = pp_gains
        data['true_pp'] = true_pps
        data['true_score'] = true_scores
        data['true_score_id'] = true_score_ids
        data['true_mod'] = true_mods
        data['pp_gain_expect'] = data['pp_gain (breaking)'] * data['break_prob'] * data['pass_prob']

        data = pd.DataFrame(data,
                            columns=['name', 'star', "max_pp",
                                     'true_pp',
                                     'pred_pp',
                                     'pred_pp (breaking)',
                                     'break_prob',
                                     'pp_gain (breaking)', 'pass_prob', 'set_id',
                                     'valid_count',
                                     'pp_gain_expect', 'true_score_id', 'true_mod', 
                                     'true_score'])
        data.sort_values(by="pp_gain_expect", ascending=False, inplace=True)
        return data


    def generate_recall_table(self, uid, key_count, beatmap_ids, max_star=None, min_star=0,
                              required_mods=None, min_pp=None):
        base_projection = [
            BeatmapEmbedding.TABLE_NAME + "." + BeatmapEmbedding.BEATMAP_ID,
            ModEmbedding.SPEED,
            # for pp
            Beatmap.MOD_STAR, Beatmap.MOD_MAX_PP,
            # for name
            Beatmap.VERSION, Beatmap.NAME, Beatmap.SET_ID,
            # for validation
            BeatmapEmbedding.COUNT_HT, BeatmapEmbedding.COUNT_NM, BeatmapEmbedding.COUNT_DT
        ]
        data_list = []
        cursor = []
        query_params = {
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.USER_ID: ('=', uid),
            UserEmbedding.TABLE_NAME + "." + UserEmbedding.GAME_MODE: ('=', "osu"),
            ModEmbedding.TABLE_NAME + "." + ModEmbedding.MOD: (
                '=',
                osu_utils.mods_to_db_key(required_mods)
            ),
            BeatmapEmbedding.TABLE_NAME + ".item_embedding_alpha": ('!=', 0)
        }

        cursor += list(
            osu_utils.predict_score(self.connection, query_params, self.config.embedding_size,
                                    projection=base_projection))
        for x in cursor:
            pp_ratio, bid, speed, mod_star, mod_max_pp, version, name, set_id, \
            count_ht, count_nm, count_dt = x[:len(base_projection) + 1]
            if len(beatmap_ids) != 0 and int(bid) not in beatmap_ids:
                continue
            mod_int = osu_utils.mods_to_db_key(required_mods)
            max_pp = json.loads(mod_max_pp)[mod_int]
            pred_pp = osu_utils.map_osu_pp(pp_ratio, real_to_train=False, max_pp=max_pp)
            star = json.loads(mod_star)[mod_int]
            count = count_dt if speed == 1 else count_nm
            if max_star is not None:
                if star > max_star:
                    continue
            if star < min_star:
                continue
            if count_dt + count_nm < self.config.embedding_size:
                continue
            data_list.append([bid, mod_int, star, max_pp, 0, pred_pp, self.map_beatmap_name(name, version),
                              speed, set_id, count])
        data = pd.DataFrame(data_list,
                            columns=['id', "mod", "star", "max_pp", "real_pp", "pred_pp", "name",
                                     'speed', 'set_id', 'valid_count'])
        return data
