import numbers
from typing import Any, Dict, List, Tuple

import xgboost
from xgboost import callback
from tqdm import tqdm

import osu_utils
from data import repository
from data.model import *
from pass_feature_feed import PassFeatureFeed
from matplotlib import pyplot as plt


class PassDataIter(xgboost.core.DataIter):

    def __init__(self, connection: sqlite3.Connection, config: NetworkConfig, start_ratio, end_ratio, speed):
        super().__init__()
        self.connection = connection
        self.config = config
        self.pass_feature = PassFeatureFeed(config, connection)
        self.batch_size = 4096 * 16
        self.it = 0
        self.pass_rate, self.count = repository.select(connection, CannotPass.TABLE_NAME, 
                                                       [f'AVG({CannotPass.PASS}), COUNT(1)'], 
                                                       where={CannotPass.SPEED: speed}).fetchone()
        self.data_start = int(self.count * start_ratio)
        self.data_end = int(self.count * end_ratio)
        self.pbar: tqdm = None
        self.reset()
        self.speed = speed

    def reset(self) -> None:
        self.it = 0
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=(self.data_end - self.data_start) // self.batch_size)

    # @measure_time
    def get_data(self, start, end) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = repository.select(self.connection, CannotPass.TABLE_NAME, project=[
            CannotPass.PASS, CannotPass.BEATMAP_ID, CannotPass.USER_ID,
            CannotPass.USER_VARIANT
        ], where={CannotPass.SPEED: self.speed}, limit=end - start, offset=start).fetchall()
        x = []
        name = None
        y = []
        for d in data:
            yi, bid, uid, variant = d
            xi, ni = self.pass_feature.get_pass_features(uid, variant, bid, name is None)
            y.append(yi)
            x.append(xi)
            if name is None:
                name = ni

        x = np.asarray(x)
        y = np.asarray(y)
        return x, y, name

    def next(self, input_data) -> int:
        start = self.it * self.batch_size + self.data_start
        end = min(self.data_end, start + self.batch_size)
        if end <= start:
            self.pbar.close()
            self.pbar = None
            return 0
        x, y, name = self.get_data(start, end)
        if input_data is not None:
            input_data(data=x, label=y)
        self.it += 1
        self.pbar.update()
        return 1

def save(model, feature_names, speed, record=None):
    model.save_model(get_pass_model_path(speed) + "_train")
    old = model.feature_names
    # model.dump_model(get_pass_model_path(speed) + "_dump.txt")
    model.feature_names = feature_names
    with open(os.path.join(get_pass_model_dir(speed), "importance.txt"), "w") as f:
        for k, v in sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True):
            f.write(f"{k}\t\t{v}\n")
    model.feature_names = old
    if record is not None:
        import json
        with open(os.path.join(get_pass_model_dir(speed), "log.json"), "w") as f:
            json.dump(record, f)

class SaveModelCallback(callback.TrainingCallback):

    def __init__(self, feature_names: List[str], speed: int, params: Dict[str, Any]):
        super().__init__()
        self.last_time = time.time()
        self.speed = speed
        self.feature_names = feature_names
        self.params = params

    def _get_key(self, data, metric):
        return f'{data}-{metric}'

    def after_iteration(self, model: xgboost.Booster, epoch, evals_log):
        t = time.time() - self.last_time
        save(model, self.feature_names, self.speed, evals_log)
        self.last_time = time.time()

        cur_auc = evals_log['eval']['auc'][-1]
        last_auc = evals_log['eval']['auc'][-2] if len(evals_log['eval']['auc']) > 1 else -1
        if cur_auc <= last_auc:
            self.params["learning_rate"] = self.params["learning_rate"] * 0.8
            model.set_param("learning_rate", self.params["learning_rate"])
        print(f'time: {t} s, eval auc: {last_auc:.6f} -> {cur_auc:.6f}, lr: {self.params["learning_rate"]}')
        # False to indicate training should not stop.
        return self.params["learning_rate"] <= 0.05

def train(config: NetworkConfig):
    connection = repository.get_connection()

    for speed in [1, 0, -1]:
        train_data = PassDataIter(connection, config, 0.0, 0.95, speed)
        valid_data = PassDataIter(connection, config, 0.95, 1.00, speed)
        print(f"start training speed = {speed}, size = {train_data.count}, pass rate = {train_data.pass_rate}")
        # train_data = PassDataIter(connection, config, 0.0, 0.1)
        # valid_data = PassDataIter(connection, config, 0.1, 0.12)
        x, y, feature_names = train_data.get_data(0, 1)
        monotone_constraints = [0] * len(feature_names)
        monotone_constraints[feature_names.index('log(Beatmap.PLAY_COUNT)')] = 1

        param = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'error'],
            'subsample': 1.0,
            'learning_rate': 1.0,
            'monotone_constraints': tuple(monotone_constraints),
            'tree_method': 'hist',
            'nthread': 1
        }
        print('param:', param)
        print('x[0]:', x)
        print('y[0]:', y)
        print('features:', feature_names)

        round = 200
        d_train = xgboost.DMatrix(train_data, feature_names=feature_names)
        d_val = xgboost.DMatrix(valid_data, feature_names=feature_names)
        bst = xgboost.train(param, d_train, round, evals=[(d_train, 'train'), (d_val, 'eval')],
                            callbacks=[
                                callback.EarlyStopping(
                                    5,
                                    metric_name='auc',
                                    data_name='eval', maximize=True, save_best=True
                                ), 
                                SaveModelCallback(feature_names, speed, param)
                            ])
        save(bst, feature_names, speed, None)

        del d_train
        del d_val
        del train_data
        del valid_data
        del bst

        for file in os.listdir(None):
            if file.startswith("DMatrix"):
                print("Remove: ", file)
                os.remove(file)

if __name__ == "__main__":
    try:
        train(NetworkConfig())
    finally:
        for file in os.listdir(None):
            if file.startswith("DMatrix"):
                print("Remove: ", file)




