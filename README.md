# AlphaOsu!

[![](https://img.shields.io/endpoint?color=blue&url=https%3A%2F%2Falphaosu.keytoix.vip%2Fapi%2Fstatistics%3Ftype%3Duser)](https://alphaosu.keytoix.vip/)
[![](https://img.shields.io/endpoint?color=orange&url=https%3A%2F%2Falphaosu.keytoix.vip%2Fapi%2Fstatistics%3Ftype%3Drecommend)](https://alphaosu.keytoix.vip/)
[![](https://img.shields.io/endpoint?color=green&url=https%3A%2F%2Falphaosu.keytoix.vip%2Fapi%2Fstatistics%3Ftype%3Dusage)](https://alphaosu.keytoix.vip/)
[![](https://img.shields.io/discord/1021343109398937610?label=discord&logo=discord&style=social)](https://discord.gg/H5VzJxeK4F)

Using advanced machine learning technology to help you farm PP in [osu!](https://osu.ppy.sh/) .


Link: https://alphaosu.keytoix.vip/

**PR is welcome!**


## Setup environment

```bash
pip install -r requirements.txt
```

## Run the whole pipeline

```bash
python pipeline.py
```

## Struct

- fetch data: `data_fetcher.py`
- train the score model: `train_score_als_db.py`
- train the pass model: `train_pass_xgboost.py`
- inference: `recommender.py`


## About

Please visit https://alphaosu.keytoix.vip/about for more details about the algorithm.
