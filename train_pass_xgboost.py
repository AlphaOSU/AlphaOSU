import numbers

import xgboost
from xgboost import callback
from tqdm import tqdm

import osu_utils
from data import repository
from data.model import *
from pass_feature import PassFeature
from matplotlib import pyplot as plt


class PassDataIter(xgboost.core.DataIter):

    def __init__(self, connection: sqlite3.Connection, config: NetworkConfig, start_ratio, end_ratio):
        super().__init__()
        self.connection = connection
        self.config = config
        self.pass_feature = PassFeature(config, connection)
        self.batch_size = 4096 * 16
        self.it = 0
        self.count = repository.count(connection, CannotPass.TABLE_NAME)
        self.data_start = int(self.count * start_ratio)
        self.data_end = int(self.count * end_ratio)
        self.pbar: tqdm = None
        self.reset()

    def reset(self) -> None:
        self.it = 0
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=(self.data_end - self.data_start) // self.batch_size)

    # @measure_time
    def get_data(self, start, end):
        data = repository.select(self.connection, CannotPass.TABLE_NAME, project=[
            CannotPass.PASS, CannotPass.SPEED, CannotPass.BEATMAP_ID, CannotPass.USER_ID,
            CannotPass.USER_VARIANT, CannotPass.SCORE
        ], limit=end - start, offset=start).fetchall()

        x = []
        name = None
        y = []
        for d in data:
            yi, speed, bid, uid, variant, score = d
            mod = ["HT", "NM", "DT"][speed + 1]
            xi, ni = self.pass_feature.get_pass_features(uid, variant, bid, mod, name is None)
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

def save(model, record=None):
    path = os.path.join("result", "pass_xgboost")
    os.makedirs(path, exist_ok=True)
    model.save_model(os.path.join(path, "model"))
    model.dump_model(os.path.join(path, "model_dump.txt"))
    if record is not None:
        import json
        with open(os.path.join(path, "log.json"), "w") as f:
            json.dump(record, f)

class Plotting(callback.TrainingCallback):
    '''Plot evaluation result during training.  Only for demonstration purpose as it's quite
    slow to draw.

    '''
    def __init__(self, feature_names):
        super().__init__()
        self.last_time = time.time()
        self.feature_names = feature_names

    def _get_key(self, data, metric):
        return f'{data}-{metric}'

    def after_iteration(self, model, epoch, evals_log):
        old = model.feature_names
        model.feature_names = self.feature_names
        i = 0
        for k, v in sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True):
            print(k, v, end="\t\t")
            if i >= 10:
                break
            i += 1
        print("time:", time.time() - self.last_time)
        self.last_time = time.time()
        save(model, evals_log)
        model.feature_names = old
        # False to indicate training should not stop.
        return False

def train(config: NetworkConfig):
    connection = repository.get_connection()
    train_data = PassDataIter(connection, config, 0.0, 0.8)
    valid_data = PassDataIter(connection, config, 0.8, 1.0)
    # train_data = PassDataIter(connection, config, 0.0, 0.1)
    # valid_data = PassDataIter(connection, config, 0.1, 0.12)
    _, _, feature_names = train_data.get_data(0, 1)
    monotone_constraints = [0] * len(feature_names)
    monotone_constraints[feature_names.index('log(Beatmap.PLAY_COUNT)')] = 1
    monotone_constraints[feature_names.index(Beatmap.HP)] = -1

    d_train = xgboost.DMatrix(train_data, feature_names=feature_names)
    d_val = xgboost.DMatrix(valid_data, feature_names=feature_names)


    param = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'error'],
        'subsample': 1.0,
        'eta': 1.0,
        'monotone_constraints': tuple(monotone_constraints)
        # 'verbosity': 3
    }

    round = 200
    bst = xgboost.train(param, d_train, round, evals=[(d_train, 'train'), (d_val, 'eval')],
                        callbacks=[callback.EarlyStopping(
                            5,
                            metric_name='auc',
                            data_name='eval', maximize=True, save_best=True
                        ), Plotting(feature_names),
                        callback.LearningRateScheduler(lambda epoch: 1 - epoch / round)])
    bst.feature_names = feature_names
    for k, v in sorted(bst.get_fscore().items(), key=lambda x: x[1], reverse=True):
        print(k, v)
    save(bst, None)

    # for batch_id in tqdm(range(size // seq.batch_size)):
    #     start = batch_id * seq.batch_size
    #     end = min(size, start + seq.batch_size)
    #     x = seq[start:end]



if __name__ == "__main__":
    try:
        train(NetworkConfig())
    finally:
        for file in os.listdir(None):
            if file.startswith("DMatrix"):
                print("Remove: ", file)




