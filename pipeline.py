import gzip
import shutil
import sys

import data_fetcher
import train_pass_kernel
import train_score_als_db
from data.model import *
import time


class RedirectLogger:
    """
    Redirect the logs to both stdout and file.
    """

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

    sys.stdout = RedirectLogger(name, sys.stdout)
    print(name)


if __name__ == "__main__":
    config_path = sys.argv[-1]
    config = NetworkConfig.from_config(config_path)
    deploy_db_file = os.path.join("result", "data_deploy.db")
    backup_db_file = os.path.join("result", "data.db.gz")
    training_db_file = os.path.join("result", "data.db")

    try:
        # First, remove the log dir
        shutil.rmtree(os.path.join("result", "log"), ignore_errors=True)

        # Backup old database to avoid accidents.
        print("Backup")
        if os.path.isfile(training_db_file):
            with open(training_db_file, "rb") as fin, gzip.open(backup_db_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)

        # Second, fetch data
        register_log_output("data_fetch")
        if config.game_mode == 'osu':
            data_fetcher.fetch_std()
        elif config.game_mode == 'mania':
            data_fetcher.fetch_mania()

        # Third, train the score prediction model.
        register_log_output("train_score")
        with repository.get_connection() as conn:
            train_score_als_db.train_score_by_als(config, conn)
            train_score_als_db.update_score_count(config, conn)

        # Fourth, train the pass model.
        register_log_output("train_pass")
        train_pass_kernel.construct_nearest_neighbor(config)

        # Finally, deploy it!!
        register_log_output("deploy")
        with repository.get_connection() as conn:

            # Create beatmap index for fast beatmap searching.
            # See https://www.sqlite.org/fts3.html for more details.
            print("Index Beatmap")
            repository.execute_sql(conn, "DROP TABLE IF EXISTS BeatmapSearch")
            repository.execute_sql(conn, "CREATE VIRTUAL TABLE IF NOT EXISTS BeatmapSearch USING fts4(id, set_id, name, version, creator)")
            repository.execute_sql(conn, "INSERT INTO BeatmapSearch SELECT id, set_id, name, version, creator FROM Beatmap")
            conn.commit()

            # VACCUM database for saving server disk space
            print("VACCUM")
            conn.execute("VACUUM")

            print("Copy back")
            # copy back database for the next training
            shutil.copyfile(training_db_file, training_db_file + "_tmp")

            print("Deploy")
            # Fast deploy by directly moving
            shutil.move(training_db_file, deploy_db_file)
            shutil.move(training_db_file + "_tmp", training_db_file)

            try:
                from post_process import finish
            except ImportError:
                def finish():
                    pass
            finish()

            print("Finish!")

    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)

        try:
            from post_process import fail
        except ImportError:
            def fail():
                pass
        fail()
