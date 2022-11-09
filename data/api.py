import json
import requests
import os
import time

session = None
recent_request = None
recent_request_time = 0
secret_object = None

DEBUG = False
REQUEST_MIN_INTERVAL = 1

OSU_WEBSITE = 'https://osu.ppy.sh/'

def get_secret_value(key, default=None):
    global secret_object
    if secret_object is None:
        with open(os.path.join("data", "secret.json")) as f:
            secret_object = json.load(f)
    return secret_object.get(key, default)

def request_api(api, method, end_point=None, params=None, header=None, retry_count=0):
    if params is None:
        params = {}
    if header is None:
        header = {}
    if end_point is None:
        end_point = get_secret_value("osu_website", OSU_WEBSITE)
        end_point = end_point + "api/v2/"
    url = end_point + api
    global session, recent_request, recent_request_time, REQUEST_MIN_INTERVAL
    if session is None:
        session = requests.Session()
    recent_request = "{method} {url} params: {params}, headers: {headers}".format(
        method=method, url=url,
            params=params,
        headers=header)
    if DEBUG:
        print(recent_request)

    current_interval = time.time() - recent_request_time
    if current_interval < REQUEST_MIN_INTERVAL:
        time.sleep(REQUEST_MIN_INTERVAL - current_interval)
    recent_request_time = time.time()

    try:
        if method.lower() == 'get':
            response = session.get(url, params=params, timeout=60, headers=header).json()
        else:
            response = session.post(end_point + api, data=params, timeout=60, headers=header).json()
    except Exception as e:
        if retry_count >= 5:
            raise e
        session = None
        print("retry...")
        time.sleep(10 + retry_count * 30)
        return request_api(api, method, end_point, params, header, retry_count + 1)
    recent_request += " -> " + str(response)
    return response


def auth(params, save_name):
    for oauth_key in ['client_id', 'client_secret', 'redirect_uri', 'scope']:
        params[oauth_key] = get_secret_value(oauth_key)
    osu_website = get_secret_value("osu_website", OSU_WEBSITE)
    auth_data = request_api('token', 'post', end_point=f'{osu_website}oauth/',
                            params=params)
    auth_data['expire_time'] = time.time() + auth_data['expires_in'] - 3600
    print("auth success!")
    with open(save_name, 'w') as f:
        json.dump(auth_data, f)
    return auth_data


def get_access_token():
    # get cache token
    auth_data = {}
    auth_cache_name = "auth.json"
    if os.path.exists(auth_cache_name):
        auth_data = json.load(open(auth_cache_name))
    expire_time = auth_data.get('expire_time', 0)
    if time.time() >= expire_time:
        refresh_token = auth_data.get('refresh_token', None)
        if refresh_token is None:
            # auth first
            # webbrowser.open("http://keytoix.vip/mania/api/osu-oauth")
            print("http://keytoix.vip/mania/api/osu-oauth")
            code = input("Please open the above url, and paste the code: ")
            auth_data = auth({'grant_type': 'authorization_code', 'code': code}, auth_cache_name)
        else:
            # refresh token
            auth_data = auth({'grant_type': 'refresh_token', 'refresh_token': refresh_token},
                             auth_cache_name)
    return auth_data['token_type'] + ' ' + auth_data['access_token']


def request_auth_api(api, method, params):
    access_token = get_access_token()
    header = {'Authorization': access_token}
    return request_api(api, method, params=params, header=header)
