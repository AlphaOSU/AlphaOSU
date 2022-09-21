# AlphaOsu!

Using advanced machine learning technology to help you farm PP in [osu!](https://osu.ppy.sh/)

Link: https://alphaosu.keytoix.vip/

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
