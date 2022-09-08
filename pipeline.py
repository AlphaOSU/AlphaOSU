import data_fetcher
import train_score_als_db
import prepare_pass_data
import train_pass_xgboost
from data.model import *
import sys


class RedirectLogger:

    def __init__(self, name, std_out):
        log_dir = os.path.join("result", "log")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{name}.log")
        if not os.path.isfile(path):
            open(path, mode="w").close()
        self.path = path
        self.stdout = std_out
        self.name = name

    def write(self, string):
        string = f"[{time.ctime()}] ({self.name}) {string}"
        self.stdout.write(string)
        with open(self.path, 'a') as f:
            f.write(string)

    def flush(self):
        self.stdout.flush()


def register_log_output(name):
    if type(sys.stdout) == RedirectLogger:
        sys.stdout = sys.stdout.stdout
        sys.stderr = sys.stderr.stdout


    sys.stdout = RedirectLogger(name, sys.stdout)
    sys.stderr = RedirectLogger(name, sys.stderr)


if __name__ == "__main__":
    config = NetworkConfig()

    try:
        register_log_output("data_fetch")
        data_fetcher.fetch()

        register_log_output("train_score")
        train_score_als_db.train_score_by_als(config)

        register_log_output("prepare_not_passed_candidates")
        prepare_pass_data.prepare_not_passed_candidates(config)
        prepare_pass_data.shuffle_cannot_pass()

        register_log_output("train pass")
        train_pass_xgboost.train(config)

        # clean
        for file in os.listdir(None):
            if file.startswith("DMatrix"):
                print("Remove: ", file)
                os.remove(file)
        with repository.get_connection() as conn:
            print("VACCUM")
            conn.execute("VACUUM")
    except:
        import traceback
        traceback.print_exc()
