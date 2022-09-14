import data_fetcher
import train_score_als_db
import prepare_pass_data
import train_pass_xgboost
from data.model import *
import sys
import shutil
from recommender import PPRecommender


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
        self.line_start = True

    def write(self, string):
        print_string = log_string = string
        if self.line_start:
            prefix = f"[{time.ctime()}] "
            print_string = f"\033[32m{prefix}\033[0m{string}"
            log_string = f"{prefix}{string}"
            self.line_start = False
        if string.endswith("\n"):
            self.line_start = True
        self.stdout.write(print_string)
        self.stdout.flush()
        with open(self.path, 'a') as f:
            f.write(log_string)

    def flush(self):
        self.stdout.flush()


def register_log_output(name):
    if type(sys.stdout) == RedirectLogger:
        sys.stdout = sys.stdout.stdout
        # sys.stderr = sys.stderr.stdout

    sys.stdout = RedirectLogger(name, sys.stdout)
    print(name)
    # sys.stderr = RedirectLogger(name, sys.stderr)


if __name__ == "__main__":
    config = NetworkConfig()

    try:
        shutil.rmtree(os.path.join("result", "log"), ignore_errors=True)
        register_log_output("data_fetch")
        data_fetcher.fetch()

        register_log_output("train_score")
        train_score_als_db.train_score_by_als(config)

        register_log_output("prepare_not_passed_candidates")
        prepare_pass_data.prepare_not_passed_candidates(config)
        prepare_pass_data.shuffle_cannot_pass()

        register_log_output("train_pass")
        train_pass_xgboost.train(config)

        register_log_output("test")
        connection = repository.get_connection()
        recommender = PPRecommender(config, connection)
        
        for uid in [10500832, 7304075]:
            name, data = recommender.predict(uid)
            print("\n\n")
            print(f"name: [{uid}] {name}")
            print(data.to_string())

        # move db
        with repository.get_connection() as conn:
            print("Drop CannotPass")
            repository.execute_sql(conn, f"DROP TABLE IF EXISTS {CannotPass.TABLE_NAME}")

            print("Index Beatmap")
            repository.execute_sql(conn, "DROP TABLE IF EXISTS BeatmapSearch")
            repository.execute_sql(conn, "CREATE VIRTUAL TABLE IF NOT EXISTS BeatmapSearch USING fts4(id, set_id, name, version, creator)")
            repository.execute_sql(conn, "INSERT INTO BeatmapSearch SELECT id, set_id, name, version, creator FROM Beatmap")
            conn.commit()

            print("VACCUM")
            conn.execute("VACUUM")

            print("Move DB")
            shutil.copyfile(os.path.join("result", "data.db"), os.path.join("result", "data_deploy.db"))

    except:
        import traceback
        traceback.print_exc(file=sys.stdout)
