from hashlib import md5
from os import getuid
import sys
sys.path.append(".")

import osu_utils
import math

from flask import Flask, request, g
from recommender import PPRecommender

from data.model import *

app = Flask(__name__)


def get_mock_data():
    import csv, time
    time.sleep(0.5)
    with open(os.path.join("report", "[GS]Myuka - PP report.csv"), "r") as f:
        lines = f.readlines()
        csv_reader = csv.reader(lines[1:])
        results = []
        for i, line in enumerate(csv_reader):
            results.append({
                "id": str(i),
                "mapName": line[2],
                "mapLink": f"https://osu.ppy.sh/beatmaps/{line[0]}",
                "mod": [line[1]],
                "difficulty": float(line[3]),
                "keyCount": 4,
                "currentScore": float(line[4]) if line[4] != "" else None,
                "currentScoreLink": "https://baidu.com",
                "currentPP": float(line[5]) if line[5] != "" else None,
                "predictScore":  float(line[6]),
                "predictPP":  float(line[11]),
                "newRecordPercent": float(line[7]),
                "ppIncrement": float(line[8]),
                "passPercent": float(line[9]),
                "ppIncrementExpect": float(line[10])
            })
        return results

def nan_to_none(x):
    return None if math.isnan(x) else x

def map_recommend_data(record):
    true_score_id = record['true_score_id']
    return {
        'id': md5(f"{record['id']} {record['mod']}".encode()).hexdigest(), 
        'mapName': record['name'],
        'mapLink': f"https://osu.ppy.sh/beatmaps/{record['id']}",
        'mod': [record['mod']],
        'difficulty': record['star'],
        'keyCount': int(record['cs']), 
        'currentScore': nan_to_none(record['true_score']),
        'currentScoreLink': None if math.isnan(record['true_score_id']) else f"https://osu.ppy.sh/scores/mania/{int(true_score_id)}",
        'currentPP': nan_to_none(record['true_pp']),
        'predictScore': record['pred_score'],
        'predictPP': record['pred_pp'],
        'newRecordPercent': record['break_prob'],
        'ppIncrement': record['pp_gain (breaking)'],
        'passPercent': record['pass_prob'],
        'ppIncrementExpect': record['pp_gain_expect']
    }

def get_recommend_data(uid, key_count, beatmap_ids):
    config = NetworkConfig()
    recommender = PPRecommender(config, get_connection())
    df = recommender.predict(uid, key_count, beatmap_ids)
    if df is None:
        raise AppException(1, "invalid uid")
    df.reset_index(inplace=True)
    return list(map(map_recommend_data, df.to_dict("records")))
    # data.map

class AppException(Exception):
    def __init__(self, error_code, error_msg) -> None:
        self.code = error_code
        self.msg = error_msg

def on_success(data):
    return {
        "success": True,
        "code": 0,
        "message": "",
        "data": data
    }

def on_error(code, msg):
    return {
        "success": False,
        "code": code,
        "message": msg,
        "data": None
    }

def on_crash(e):
    if isinstance(e, AppException):
        return on_error(e.code, e.msg)
    import traceback
    exp_str = traceback.format_exc()
    print(exp_str)
    return on_error(-1, exp_str)

@app.route("/test", methods=['GET'])
def test():
    return str(request.args)

def get_connection() -> sqlite3.Connection:
    if not hasattr(g, 'connection') or g.connection is None:
        g.connection = repository.get_connection(deploy=True)
    return g.connection

def check_param(key, expect_type, default=None, list_min_size=0, param_dict=None):
    if param_dict is None:
        param_dict = request.args
    if key not in param_dict:
        if default is not None:
            return default
        raise AppException(-2, f"require param: '{key}'")
    if list_min_size > 0:
        result = list(map(expect_type, param_dict.get(key, "").split(",")))
        if len(result) < list_min_size:
            raise AppException(-2, f"param '{key}' = {param_dict[key]} should be an array and have a length >= {list_min_size}")
        return result
    try:
        return expect_type(param_dict.get(key, default))
    except:
        raise AppException(-2, f"param '{key}' = {param_dict[key]}, but expect type {str(expect_type)}")

def map_sim_data(record):
    return {
        'id': md5(f"{record['id']}".encode()).hexdigest(), 
        'userName': record['name'],
        'userLink': f"https://osu.ppy.sh/users/{record['id']}/mania",
        'pp': record['pp'],
        'similarity': -record['distance']
    }

@app.route("/api/v1/self/users/similarity", methods=['GET'])
def similarity():
    try:
        uid = get_uid()
        key_count = check_param("keyCount", expect_type=int)
        if key_count != 4 and key_count != 7:
            return on_error(-2, f"invalid keyCount: {key_count}")
        config = NetworkConfig()
        recommender = PPRecommender(config, get_connection())
        sim = recommender.similarity(uid, f"{key_count}k")
        if sim is None:
            return on_error(1, "user not found")
        sim = sim.iloc[:min(50, len(sim)), :]
        data = list(map(map_sim_data, sim.to_dict("records")))
        return on_success(data)
        
    except Exception as e:
        return on_crash(e)


@app.route("/api/v1/login", methods=['POST', 'GET'])
def login():
    try:
        connection = get_connection()
        if request.method == 'POST':
            user_name = check_param('username', str, None, param_dict=request.get_json())
        else:
            user_name = check_param('username', str, None)
        uid = repository.execute_sql(connection, 
                                     f"SELECT {User.ID} FROM {User.TABLE_NAME} WHERE {User.NAME} == ? COLLATE NOCASE", 
                                     [user_name]).fetchall()
        if len(uid) > 0:
            uid = uid[0][0]
            user_bp = osu_utils.get_user_bp(connection, uid, NetworkConfig())
            count_4k = count_7k = 0
            for tup in user_bp.data.values():
                if tup[5] == 4:
                    count_4k += 1
                elif tup[5] == 7:
                    count_7k += 1
            return on_success({
                "uid": str(uid),
                "keyCount": 4 if count_4k >= count_7k else 7,
                "gameMode": 3
            })
        else:
            return on_error(1, "user not found")
    # request.headers[]
    except Exception as e:
        return on_crash(e)

def get_uid():
    uid = check_param("uid", expect_type=int, default=-1, param_dict=request.headers)
    if uid == -1:
        uid = check_param("uid", expect_type=int, default=-1, param_dict=request.args)
    if uid < 0:
        raise AppException(2, "invalid uid")
    return uid

@app.route("/api/v1/self/maps/recommend", methods=['GET'])
def recommend():
    try:
        uid = get_uid()
        search = check_param("search", str, "").strip()
        key_count = check_param("keyCount", int, [4, 7], list_min_size=1)
        beatmap_id = []
        if search != "":
            beatmap_id = repository.execute_sql(get_connection(), "SELECT id FROM BeatmapSearch WHERE BeatmapSearch MATCH ?", [search]).fetchall()
            beatmap_id = list(map(lambda x: x[0], beatmap_id))
        results = get_recommend_data(uid, key_count, beatmap_id)
        
        # filter
        new_record_prob_filter = check_param("newRecordPercent", float, default=[0, 1], list_min_size=2)
        pass_prob_filter = check_param("passPercent", float, default=[0, 1], list_min_size=2)
        
        
        def filter_result(x):
            if x['passPercent'] < pass_prob_filter[0] or x['passPercent'] > pass_prob_filter[1]:
                return False
            if x['newRecordPercent'] < new_record_prob_filter[0] or x['newRecordPercent'] > new_record_prob_filter[1]:
                return False
            if x['keyCount'] not in key_count:
                return False
            return True

        results = list(filter(filter_result, results))

        current = check_param("current", int, 1)
        page_size = check_param("pageSize", int, 50)
        def clip(x):
            return max(min(x, len(results)), 0)
        start = clip((current - 1) * page_size)
        end = clip(current * page_size)
        next = current + 1 if (current + 1) * page_size < len(results) else -1
        prev = current - 1 if current > 1 else -1
        return on_success({
            "prev": prev, "next": next,
            "total": len(results),
            "list": results[start:end]
        })
        
    # request.headers[]
    except Exception as e:
        return on_crash(e)

@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'connection') and g.connection is not None:
        g.connection.close()
        g.connection = None