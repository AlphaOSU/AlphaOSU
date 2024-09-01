# AlphaOsu!

[![](https://img.shields.io/endpoint?color=blue&url=https%3A%2F%2Falphaosu.keytoix.vip%2Fapi%2Fstatistics%3Ftype%3Duser)](https://alphaosu.keytoix.vip/)
[![](https://img.shields.io/endpoint?color=orange&url=https%3A%2F%2Falphaosu.keytoix.vip%2Fapi%2Fstatistics%3Ftype%3Drecommend)](https://alphaosu.keytoix.vip/)
[![](https://img.shields.io/endpoint?color=green&url=https%3A%2F%2Falphaosu.keytoix.vip%2Fapi%2Fstatistics%3Ftype%3Dusage)](https://alphaosu.keytoix.vip/)
[![](https://img.shields.io/discord/1021343109398937610?label=discord&logo=discord&style=social)](https://discord.gg/H5VzJxeK4F)

Using advanced machine learning technology to help you farm PP in [osu!](https://osu.ppy.sh/) .


Link: https://alphaosu.keytoix.vip/

**PR is welcome!**


## Setup environment

- First, install the python requirements.

```bash
pip install -r requirements.txt
```

- Second, clone the project [osu-tools](https://github.com/ppy/osu-tools/) and setup the environment of osu-tools. You may need to install .NET 6.0 SDK.

- Finally, write a `data/secret.json` file. The content is like:

```json
{
  "oauth_url": "<oauth_url>",
  "client_id": "<client_id>",
  "client_secret": "<client_secret>",
  "redirect_uri": "<redirect_uri>",
  "scope": "public",
  "osu_website": "https://osu.ppy.sh/",
  "osu_tools_command": [
    "dotnet",
    "run",
    "--project",
    "<path to osu_tools>/PerformanceCalculator"
  ]
}
```

Please setup an osu oauth application to get `client_id`, `client_secret` and `redirect_uri`. You may need a osu! supporter account to fetch DT/HT/Country rankings.

## Run the whole pipeline

```bash
python pipeline.py
```

## Struct

- fetch data: `data_fetcher.py`
- train the score model: `train_score_als_db.py`
- train the pass model: `train_pass_kernel.py`
- inference: `recommender.py`


## About

Please visit https://alphaosu.keytoix.vip/about for more details about the algorithm.
