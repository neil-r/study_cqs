import sqlite3
import typing
import json
import hashlib
import dataclasses


def convert_to_id(task_str):
    # # Uncommenting below would cause a test failure
    # return "incorrect_id"
    try:
        return hashlib.sha256(task_str.encode("utf-8")).hexdigest()
    except Exception as e:
        print(f"Error generating task ID: {e}")
        return None


@dataclasses.dataclass
class TaskResult:
    evalutation_id: int
    task: typing.Dict
    log: typing.List
    discussion_duration: float
    prompt_strategy: str
    model_id: str
    answer_value: str
    answer_response: str
    correct: bool


class DatabaseSqlLite:
    def __init__(self, db_file_path="data.db"):
        self.db_file_path = db_file_path
        try:
            self.connection = sqlite3.connect(self.db_file_path)
            with self.connection as c:
                cur = c.cursor()
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS model(
                    model_id TEXT NOT NULL UNIQUE,
                    model_name TEXT NOT NULL,
                    model_size INTEGER NOT NULL,
                    model_version INTEGER NOT NULL,
                    PRIMARY KEY(model_id));
                """
                )
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS task_results(
                    task_id INTEGER NOT NULL UNIQUE,
                    task TEXT NOT NULL,
                    log TEXT NOT NULL,
                    discussion_duration REAL NOT NULL,
                    prompt_strategy TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    answer_value TEXT NOT NULL,
                    answer_response TEXT,
                    correct INTEGER NOT NULL,
                    PRIMARY KEY (task_id, prompt_strategy, model_id),
                    FOREIGN KEY (model_id) REFERENCES model(model_id));
                """
                )
                c.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")

    def close(self):
        try:
            if hasattr(self, "connection") and self.connection:
                self.connection.close()
        except sqlite3.Error as e:
            print(f"Error closing database connection: {e}")

    def add_model_entry(
        self, model_id: str, model_name: str, model_size: int, model_version: float
    ):
        try:
            with sqlite3.connect(self.db_file_path) as c:
                cur = c.cursor()
                cur.execute(
                    """
                    INSERT INTO model (model_id, model_name, model_size, model_version)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_id) DO NOTHING;
                """,
                    (model_id, model_name, model_size, model_version),
                )
                c.commit()
        except sqlite3.Error as e:
            print(f"Error adding model entry: {e}")

    def has_model(self, model_id):
        try:
            with sqlite3.connect(self.db_file_path) as c:
                o = c.execute(
                    "SELECT * FROM model WHERE model_id = ?", (model_id,)
                ).fetchone()
                return o is not None
        except sqlite3.Error as e:
            print(f"Error checking model existence: {e}")
            return False

    def add_task_result(
        self,
        task,
        log,
        prompt_strategy: str,
        model_id: str,
        answer_value: str,
        discussion_duration: float,
        answer_response: typing.Optional[str] = None,
        correct: bool = False,
    ):
        try:
            with sqlite3.connect(self.db_file_path) as c:
                cur = c.cursor()
                task_content = task.to_json()
                task_json_str = json.dumps(task_content, sort_keys=True)
                cur.execute(
                    "INSERT INTO task_results VALUES (?,?,?,?,?,?,?,?,?);",
                    (
                        convert_to_id(task_json_str + prompt_strategy + model_id),
                        task_json_str,
                        json.dumps(log),
                        discussion_duration,
                        prompt_strategy,
                        model_id,
                        answer_value,
                        answer_response,
                        1 if correct else 0,
                    ),
                )
                c.commit()
        except sqlite3.IntegrityError:
            print("Error: Duplicate task result entry detected.")
        except sqlite3.Error as e:
            print(f"Error adding task result: {e}")

    def has_task_result(self, task, prompt_strategy, model_id):
        try:
            with sqlite3.connect(self.db_file_path) as c:
                task_content = task.to_json()
                task_json_str = json.dumps(task_content, sort_keys=True)
                o = c.execute(
                    "SELECT * FROM task_results WHERE task_id = ? AND prompt_strategy = ? AND model_id = ?",
                    (
                        convert_to_id(task_json_str + prompt_strategy + model_id),
                        prompt_strategy,
                        model_id,
                    ),
                ).fetchone()
                return o is not None
        except sqlite3.Error as e:
            print(f"Error checking task result existence: {e}")
            return False

    def get_task_result(self, number):
        # # Uncommenting below would cause a test failure
        # return "Invalid return type"
        try:
            offset = number - 1
            with sqlite3.connect(self.db_file_path) as c:
                results = c.execute(
                    "SELECT * FROM task_results WHERE task_id IN (SELECT task_id FROM task_results ORDER BY task_id LIMIT ?, 1)",
                    (offset,),
                ).fetchmany()
                if results:
                    return [
                        TaskResult(
                            o[0],
                            json.loads(o[1]),
                            json.loads(o[2]),
                            o[3],
                            o[4],
                            o[5],
                            o[6],
                            o[7],
                            o[8] == 1,
                        )
                        for o in results
                    ]
                else:
                    return None
        except sqlite3.Error as e:
            print(f"Error retrieving task result: {e}")
            return None

    def update_task_answer(self, task_id, answer_response, correct):
        with sqlite3.connect(self.db_file_path) as c:
            cur = c.cursor()
            cur.execute(
                "UPDATE task_results SET answer_response=?, correct = ? WHERE task_id = ?;",
                (
                    answer_response,
                    1 if correct else 0,
                    task_id,
                ),
            )
            c.commit()
