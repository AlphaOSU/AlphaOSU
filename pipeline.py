from genericpath import isfile
import data_fetcher
import train_score_als_db
import prepare_pass_data
import train_pass_xgboost
from data.model import *
import sys
import shutil
import gzip


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
        with repository.get_connection() as conn:
            train_score_als_db.update_score_count(conn)

        register_log_output("prepare_not_passed_candidates")
        prepare_pass_data.prepare_not_passed_candidates(config)
        prepare_pass_data.shuffle_cannot_pass()

        register_log_output("train_pass")
        train_pass_xgboost.train(config)

        # deploy
        register_log_output("deploy")
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

            deploy_db_file = os.path.join("result", "data_deploy.db")
            backup_db_file = os.path.join("result", "data_deploy_backup.tar.gz")
            training_db_file = os.path.join("result", "data.db")

            print("Backup")
            if os.isfile(deploy_db_file):
                with open(deploy_db_file, "rb") as fin, gzip.open(backup_db_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            for speed in [-1, 0, 1]:
                path = get_pass_model_path(speed)
                if os.isfile(path):
                    shutil.copyfile(path, get_pass_model_path(speed) + "_backup")

            print("Deploy")
            # fast deploy by moving
            shutil.move(training_db_file, deploy_db_file)
            for speed in [-1, 0, 1]:
                shutil.move(get_pass_model_path(speed, is_training=True), get_pass_model_path(speed))
            
            print("Copy back")
            # copy back database for the next training
            shutil.copyfile(os.path.join("result", "data_deploy.db"), os.path.join("result", "data.db"))

            try:
                from post_process import finish
            except ImportError:
                def finish():
                    pass
            finish()

            print("Finish!")

    except:
        import traceback
        traceback.print_exc(file=sys.stdout)

        try:
            from post_process import fail
        except ImportError:
            def fail():
                pass
            fail()
