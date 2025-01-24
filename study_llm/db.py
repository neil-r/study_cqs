import sqlite3
import typing
import json
import hashlib
import dataclasses


def convert_to_id(task_str):
    id =  hashlib.sha256(task_str.encode("utf-8")).hexdigest()
    return id


@dataclasses.dataclass
class TaskResult:
    evalutation_id:int
    task: typing.Dict
    log: typing.List
    discussion_duration: float
    prompt_strategy: str
    model_id: str
    answer_value: str
    answer_response: str
    correct: bool

class DatabaseSqlLite:


    def __init__(self, db_file_path = "data.db"):
        self.db_file_path = db_file_path

        with sqlite3.connect(self.db_file_path) as c:
            cur = c.cursor()

            cur.execute("""CREATE TABLE IF NOT EXISTS task_results(
                task_id INTEGER NOT NULL,
                task TEXT NOT NULL,
                log TEXT NOT NULL,
                discussion_duration REAL NOT NULL,
                prompt_strategy TEXT NOT NULL,
                model_id TEXT NOT NULL,
                answer_value TEXT NOT NULL,
                answer_response TEXT,
                correct INTEGER NOT NULL,
                PRIMARY KEY (task_id, prompt_strategy, model_id));
            """)

            c.commit()
    
    def add_task_result(
        self,
        task,
        log,
        prompt_strategy:str,
        model_id:str,
        answer_value:str,
        discussion_duration:float,
        answer_response:typing.Optional[str]=None,
        correct:bool=False,
    ):
        with sqlite3.connect(self.db_file_path) as c:
            cur = c.cursor()

            task_content = task.to_json()
            task_json_str = json.dumps(task_content, sort_keys=True)

            cur.execute("INSERT INTO task_results VALUES (?,?,?,?,?,?,?,?,?);",(
                convert_to_id(task_json_str),
                task_json_str,
                json.dumps(log),
                discussion_duration,
                prompt_strategy,
                model_id,
                answer_value,
                answer_response,
                1 if correct else 0
            ))

            c.commit()

    def has_task_result(self,
        task,
        prompt_strategy,
        model_id
    ):
        with sqlite3.connect(self.db_file_path) as c:

            o = c.execute(
                "SELECT * FROM task_results WHERE task_id = ? AND prompt_strategy = ? AND model_id = ?", (
                convert_to_id(json.dumps(task.to_json(), sort_keys=True)),
                prompt_strategy,
                model_id
            )).fetchone()

            return o is not None
    
    def get_task_result(self, number):
        offset = number - 1
        print(offset)
        with sqlite3.connect(self.db_file_path) as c:
            results = c.execute(
                "SELECT * FROM task_results WHERE task_id IN (SELECT task_id FROM task_results ORDER BY task_id LIMIT ?, 1)", (
                offset,
            )).fetchmany()

            if results is not None:
                return list(TaskResult(o[0],json.loads(o[1]),json.loads(o[2]),o[3],o[4],o[5],o[6],o[7],o[8] == 1) for o in results)
            else:
                return None
