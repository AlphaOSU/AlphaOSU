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

        # move db
        with repository.get_connection() as conn:
            print("Drop CannotPass")
            repository.execute_sql(conn, f"DROP TABLE IF EXISTS {CannotPass.TABLE_NAME}")

            print("Index Beatmap")
            repository.execute_sql(conn, "DROP TABLE IF EXISTS BeatmapSearch")
            repository.execute_sql(conn, "CREATE VIRTUAL TABLE IF NOT EXISTS BeatmapSearch USING fts4(id, set_id, name, version, creator)")
            repository.execute_sql(conn, "INSERT INTO BeatmapSearch SELECT id, set_id, name, version, creator FROM Beatmap")
            conn.commit()

            print("Update score count")
            repository.ensure_column(conn, UserEmbedding.TABLE_NAME, [("count", "integer", 0)])
            repository.ensure_column(conn, BeatmapEmbedding.TABLE_NAME, [("count_HT", "integer", 0)])
            repository.ensure_column(conn, BeatmapEmbedding.TABLE_NAME, [("count_NM", "integer", 0)])
            repository.ensure_column(conn, BeatmapEmbedding.TABLE_NAME, [("count_DT", "integer", 0)])
            for speed in [-1, 0, 1]:
                mod = ['HT', 'NM', 'DT'][speed + 1]
                repository.execute_sql(conn, f"UPDATE BeatmapEmbedding SET count_{mod} = (SELECT COUNT(1) FROM Score WHERE Score.beatmap_id == BeatmapEmbedding.id AND Score.speed == {speed})")
            repository.execute_sql(conn, "UPDATE UserEmbedding SET count = (SELECT COUNT(1) FROM Score WHERE Score.user_id == UserEmbedding.id AND Score.game_mode == UserEmbedding.game_mode AND (Score.cs || 'k') == UserEmbedding.variant)")
            conn.commit()

            print("VACCUM")
            conn.execute("VACUUM")

            print("Backup")
            shutil.copyfile(os.path.join("result", "data_deploy.db"), os.path.join("result", "data_deploy_backup.db"))
            for speed in [-1, 0, 1]:
                shutil.copyfile(get_pass_model_path(speed), get_pass_model_path(speed) + "_backup")

            print("Deploy")
            shutil.copyfile(os.path.join("result", "data.db"), os.path.join("result", "data_deploy.db"))
            for speed in [-1, 0, 1]:
                shutil.copyfile(get_pass_model_path(speed) + "_train", get_pass_model_path(speed))

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
