'''
    This script generates the confusion matrix for each model. It generates
    three confusion matrixes: 1 for the classification task, 1 for the 
    extraction task, and 1 for the task sequence.
'''
import sqlite3
import pandas as pd

db_file_path = 'data.db'

with sqlite3.connect(db_file_path) as conn:
    df_results = pd.read_sql('''
        SELECT model_id, classification_correct_result, extraction_correct_result, answer_value as has_events, COUNT(*) as instance_count FROM (
        SELECT model_id, answer_value,
        MAX(CASE WHEN (prompt_strategy == "DefaultEventExistsClassificationTaskPrompt")  THEN correct END) as classification_correct_result,
        MAX(CASE WHEN (prompt_strategy == "DefaultExtractEventsPrompt") THEN correct END) as extraction_correct_result,
        COUNT(*) as pt_count
        FROM task_results GROUP BY model_id, task) GROUP BY model_id, classification_correct_result, extraction_correct_result, answer_value
                ''',conn)

print(df_results.head())

model_ids = df_results["model_id"].unique()

for model_id in model_ids:
    df_f = df_results[df_results["model_id"] == model_id]

    print("\n\n")
    print(f"Results for model: {model_id}")
    df_true = df_f[df_f['has_events'] == "1"]
    df_false = df_f[df_f['has_events'] == "0"]
    #print(df_true.head())
    print("Classification Confusion Matrix")
    print(f"truth | {df_true[df_true['classification_correct_result'] == 1]['instance_count'].sum()} | {df_true[df_true['classification_correct_result'] == 0]['instance_count'].sum()}")
    print(f"false | {df_false[df_false['classification_correct_result'] == 0]['instance_count'].sum()} | {df_false[df_false['classification_correct_result'] == 1]['instance_count'].sum()}")


    print("Extraction Confusion Matrix")
    print(f"truth | {df_true[df_true['extraction_correct_result'] == 1]['instance_count'].sum()} | {df_true[df_true['extraction_correct_result'] == 0]['instance_count'].sum()}")
    print(f"false | {df_false[df_false['extraction_correct_result'] == 0]['instance_count'].sum()} | {df_false[df_false['extraction_correct_result'] == 1]['instance_count'].sum()}")

    print("Combined Confusion Matrix")
    print(f"truth | {df_true[(df_true['classification_correct_result'] == 1) & (df_true['extraction_correct_result'] == 1)]['instance_count'].sum()} | {df_true[(df_true['extraction_correct_result'] == 0) | (df_true['classification_correct_result'] == 0)]['instance_count'].sum()}")
    print(f"false | {df_false[(df_false['extraction_correct_result'] == 0) & (df_false['classification_correct_result'] == 0)]['instance_count'].sum()} | {df_false[(df_false['extraction_correct_result'] == 1) | (df_false['classification_correct_result'] == 1)]['instance_count'].sum()}")
