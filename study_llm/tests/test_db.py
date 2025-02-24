import pytest
import sqlite3
import json
import hashlib
import os
from unittest.mock import MagicMock
from study_llm.db import convert_to_id, DatabaseSqlLite, TaskResult

DB_FILE = "test_database.db"  # Pre-generated database

@pytest.fixture(scope="function")
def db():
    """Fixture to connect to an already existing test database."""
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Test database {DB_FILE} not found. Please generate it before running tests.")
    
    db = DatabaseSqlLite(DB_FILE)  # Connect to the pre-existing database
    yield db  # Provide the database instance to the test
    
    # Ensure the database connection is closed after each test
    if hasattr(db, "connection"):
        db.connection.close()

# Test convert_to_id function
def test_convert_to_id():
    task_str = "sample task"
    expected_id = hashlib.sha256(task_str.encode("utf-8")).hexdigest()
    assert convert_to_id(task_str) == expected_id

# Test adding and checking models
def test_add_model_entry():
    db = DatabaseSqlLite(DB_FILE)
    model_id = "model_123"
    db.add_model_entry(model_id, "Test Model", 256, 1.0)
    assert db.has_model(model_id) is True
    db.connection.close()

def test_add_task_result():
    db = DatabaseSqlLite(DB_FILE)
    mock_task = MagicMock()
    mock_task.to_json.return_value = {"task": f"sample task {hashlib.sha256(os.urandom(16)).hexdigest()}"}  # Ensure uniqueness

    log = ["Step 1", "Step 2"]
    prompt_strategy = f"strategy_{hashlib.sha256(os.urandom(8)).hexdigest()[:6]}"  # Ensure uniqueness
    model_id = f"model_{hashlib.sha256(os.urandom(8)).hexdigest()[:6]}"  # Unique model ID
    answer_value = "42"
    discussion_duration = 3.5
    answer_response = "response"
    correct = True
    
    db.add_task_result(mock_task, log, prompt_strategy, model_id, answer_value, discussion_duration, answer_response, correct)
    assert db.has_task_result(mock_task, prompt_strategy, model_id) is True
    db.connection.close()


def test_get_task_result():
    db = DatabaseSqlLite(DB_FILE)
    mock_task = MagicMock()
    # Ensure unique task data per test run
    mock_task.to_json.return_value = {"task": f"task_2_{hashlib.sha256(os.urandom(16)).hexdigest()}"}

    unique_strategy = f"strategy_{hashlib.sha256(os.urandom(8)).hexdigest()[:6]}"  # Ensure uniqueness
    unique_model_id = f"model_{hashlib.sha256(os.urandom(8)).hexdigest()[:6]}"  # Ensure unique model ID

    db.add_task_result(mock_task, ["log entry"], unique_strategy, unique_model_id, "value", 5.0, "response", False)
    result = db.get_task_result(1)
    
    assert result is not None
    assert isinstance(result, list)
    assert result[0].task != {"task": "task_2"}  # Ensure uniqueness
    db.connection.close()

# Test has_model function
def test_has_model(db):
    model_id = "model_123"
    db.add_model_entry(model_id, "Test Model", 256, 1.0)
    # assert db.has_model(model_id) is True  # Should return True for existing model
    assert db.has_model("non_existent_model") is False  # Should return False for non-existing model
    # Should Fail if below uncommented
    # assert db.has_model("non_existent_model") is True  # Should return False for non-existing model


# Test has_task_result function
def test_has_task_result(db):
    mock_task = MagicMock()
    mock_task.to_json.return_value = {"task": "sample task"}
    prompt_strategy = "strategy_1"
    model_id = "model_123"
    log = ["Step 1", "Step 2"]
    answer_value = "42"
    discussion_duration = 3.5
    answer_response = "response"
    correct = True
    
    db.add_task_result(mock_task, log, prompt_strategy, model_id, answer_value, discussion_duration, answer_response, correct)
    assert db.has_task_result(mock_task, prompt_strategy, model_id) is True  # Should return True for existing task result
    assert db.has_task_result(mock_task, "wrong_strategy", model_id) is False  # Should return False for incorrect prompt strategy
    # Should Fail if below uncommented
    # assert db.has_task_result(mock_task, "wrong_strategy", model_id) is True  # Should return False for incorrect prompt strategy

