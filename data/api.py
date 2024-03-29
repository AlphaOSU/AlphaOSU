import json
import requests
import os
import time
from urllib.parse import urljoin

session = None
recent_request = None
recent_request_time = 0
secret_object = None
auth_file = "auth.json"

DEBUG = False
REQUEST_MIN_INTERVAL = 1

OSU_WEBSITE = 'https://osu.ppy.sh/'


def get_secret_value(key, default=None):
    """
    Get value in data/secret.json. Secrets will be cached in memory.
    @param key: key
    @param default: default
    @return: value
    """
    global secret_object
    if secret_object is None:
        with open(os.path.join("data", "secret.json")) as f:
            secret_object = json.load(f)
    return secret_object.get(key, default)


def request_api(api, method, end_point=None, params=None, header=None, retry_count=0, json=True,
                stream=False):
    """
    Request a network api.
    @param api: api name
    @param method: method name, GET/POST
    @param end_point: if null, use OSU_WEBSITE/api/v2
    @param params: parameters for requests
    @param header: headers for requests
    @param retry_count: maximum allowed retry count.
    @param json: whether to interpret the response in json format. True will call response.json()
    @param stream: streams for requests
    @return: response or json if json=True
    """
    if params is None:
        params = {}
    if header is None:
        header = {}
    if end_point is None:
        end_point = get_secret_value("osu_website", OSU_WEBSITE)
        end_point = urljoin(end_point, "api/v2/")
    url = urljoin(end_point, api)
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
            response = session.get(url, params=params, timeout=60, headers=header, stream=stream)
        else:
            response = session.post(end_point + api, data=params, timeout=60, headers=header,
                                    stream=stream)
        if json:
            response = response.json()
    except Exception as e:
        if retry_count >= 5:
            raise e
        session = None
        print("retry...")
        time.sleep(10 + retry_count * 30)
        return request_api(api, method, end_point, params, header, retry_count + 1, json)
    recent_request += " -> " + str(response)
    return response


def auth(params, save_name):
    """
    Auth using osu api v2.
    @param params: parameters for authentication
    @param save_name: file path to save the auth data
    @return: auth data
    """
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
    """
    Get the access token after authentication
    @return: a string containing access token
    """
    auth_data = {}
    auth_cache_name = auth_file
    if os.path.exists(auth_cache_name):
        auth_data = json.load(open(auth_cache_name))
    expire_time = auth_data.get('expire_time', 0)
    if time.time() >= expire_time:
        refresh_token = auth_data.get('refresh_token', None)
        if refresh_token is None:
            # auth first
            if (oauth_url := get_secret_value('oauth_url')) is None:
                raise ValueError("Please set oauth_url in secret.json")
            print(oauth_url)
            code = input("Please open the above url, and paste the code: ")
            auth_data = auth({'grant_type': 'authorization_code', 'code': code}, auth_cache_name)
        else:
            # refresh token
            auth_data = auth({'grant_type': 'refresh_token', 'refresh_token': refresh_token},
                             auth_cache_name)
    return auth_data['token_type'] + ' ' + auth_data['access_token']


def request_auth_api(api, method, params, enable_retry=True):
    """
    A wrapper for request_api which includes an access token in the head.
    @param api: same to request_api
    @param method: same to request_api
    @param params: same to request_api
    @param enable_retry: if true, retry_count = 5, else retry_count = 0
    @return: same to request_api
    """
    access_token = get_access_token()
    header = {'Authorization': access_token}
    if enable_retry:
        retry_count = 0
    else:
        retry_count = 5
    return request_api(api, method, params=params, header=header, retry_count=retry_count)
