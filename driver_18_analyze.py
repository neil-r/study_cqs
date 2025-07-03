"""
This script generates the confusion matrix for each model. It generates
three confusion matrixes: 1 for the classification task, 1 for the
extraction task, and 1 for the task sequence.
"""

import sqlite3
import pandas as pd

db_file_path = "data.db"

with sqlite3.connect(db_file_path) as conn:
    df_results = pd.read_sql(
        """
        SELECT model_id, answer_value as has_events, text_filter_correct_result, classification_correct_result, extraction_correct_result, COUNT(*) as instance_count FROM (
            SELECT tr.model_id as model_id, tr.answer_value as answer_value,
                    ftr.correct as text_filter_correct_result,
                    MAX(CASE WHEN (tr.prompt_strategy == "DefaultEventExistsClassificationTaskPrompt")  THEN tr.correct END) as classification_correct_result,
                    COUNT(CASE WHEN (tr.prompt_strategy == "DefaultEventExistsClassificationTaskPrompt")  THEN 1 END) as classification_result_count,
                    MAX(CASE WHEN (tr.prompt_strategy == "DefaultExtractEventsPrompt") THEN tr.correct END) as extraction_correct_result,
                    COUNT(CASE WHEN (tr.prompt_strategy == "DefaultExtractEventsPrompt") THEN 1 END) as extraction_result_count,
                    COUNT(*) as pt_count
            FROM task_results as tr INNER JOIN (
                SELECT *, json_extract(task, '$.raw_json.doc_key') as doc_key FROM task_results WHERE prompt_strategy == "full_text_search"
            ) as ftr ON json_extract(tr.task, '$.raw_json.doc_key') = ftr.doc_key
            WHERE tr.prompt_strategy != "full_text_search" 
            GROUP BY tr.model_id, tr.task
        )
        GROUP BY model_id, answer_value, text_filter_correct_result, classification_correct_result, extraction_correct_result
                """,
        conn,
    )

    df_search_filter_counts = pd.read_sql(
        """
        SELECT 
            answer_value as has_events,
            COUNT(CASE WHEN answer_response == "1" THEN answer_value == "1" END) as predicted_true,
            COUNT(CASE WHEN answer_response == "0" THEN answer_value == "0" END) as predicted_false
        FROM task_results WHERE prompt_strategy == "full_text_search" GROUP BY answer_value
    """,
        conn,
    )

print("Search Filter results:")
print(df_search_filter_counts)

# print(df_results.head())

model_ids = df_results["model_id"].unique()

for model_id in model_ids:
    df_f = df_results[df_results["model_id"] == model_id]

    print("\n\n")
    print(f"Results for model: {model_id}")
    df_true = df_f[df_f["has_events"] == "1"]
    df_false = df_f[df_f["has_events"] == "0"]
    # print(df_true.head())
    print("LLM Classification Confusion Matrix")
    print(
        f"truth | {df_true[df_true['classification_correct_result'] == 1]['instance_count'].sum()} | {df_true[df_true['classification_correct_result'] == 0]['instance_count'].sum()}"
    )
    print(
        f"false | {df_false[df_false['classification_correct_result'] == 0]['instance_count'].sum()} | {df_false[df_false['classification_correct_result'] == 1]['instance_count'].sum()}"
    )

    print("Extraction Confusion Matrix")
    print(
        f"truth | {df_true[df_true['extraction_correct_result'] == 1]['instance_count'].sum()} | {df_true[df_true['extraction_correct_result'] == 0]['instance_count'].sum()}"
    )
    print(
        f"false | {df_false[df_false['extraction_correct_result'] == 0]['instance_count'].sum()} | {df_false[df_false['extraction_correct_result'] == 1]['instance_count'].sum()}"
    )

    print("C+E Confusion Matrix")
    print(
        f"truth | {df_true[(df_true['classification_correct_result'] == 1) & (df_true['extraction_correct_result'] == 1)]['instance_count'].sum()} | {df_true[(df_true['extraction_correct_result'] == 0) | (df_true['classification_correct_result'] == 0)]['instance_count'].sum()}"
    )
    print(
        f"false | {df_false[(df_false['extraction_correct_result'] == 0) & (df_false['classification_correct_result'] == 0)]['instance_count'].sum()} | {df_false[(df_false['extraction_correct_result'] == 1) | (df_false['classification_correct_result'] == 1)]['instance_count'].sum()}"
    )

    print("T+E Confusion Matrix")
    print(
        f"truth | {df_true[(df_true['text_filter_correct_result'] == 1) & (df_true['extraction_correct_result'] == 1)]['instance_count'].sum()} | {df_true[(df_true['extraction_correct_result'] == 0) | (df_true['text_filter_correct_result'] == 0)]['instance_count'].sum()}"
    )
    print(
        f"false | {df_false[(df_false['extraction_correct_result'] == 0) & (df_false['text_filter_correct_result'] == 0)]['instance_count'].sum()} | {df_false[(df_false['extraction_correct_result'] == 1) | (df_false['text_filter_correct_result'] == 1)]['instance_count'].sum()}"
    )

    print("C+T Confusion Matrix")
    print(
        f"truth | {df_true[(df_true['classification_correct_result'] == 1) & (df_true['text_filter_correct_result'] == 1)]['instance_count'].sum()} | {df_true[(df_true['text_filter_correct_result'] == 0) | (df_true['classification_correct_result'] == 0)]['instance_count'].sum()}"
    )
    print(
        f"false | {df_false[(df_false['text_filter_correct_result'] == 0) & (df_false['classification_correct_result'] == 0)]['instance_count'].sum()} | {df_false[(df_false['text_filter_correct_result'] == 1) | (df_false['classification_correct_result'] == 1)]['instance_count'].sum()}"
    )

    print("All Confusion Matrix")
    print(
        f"truth | {df_true[(df_true['text_filter_correct_result'] == 1) & (df_true['classification_correct_result'] == 1) & (df_true['extraction_correct_result'] == 1)]['instance_count'].sum()} | {df_true[(df_true['text_filter_correct_result'] == 0) | (df_true['extraction_correct_result'] == 0) | (df_true['classification_correct_result'] == 0)]['instance_count'].sum()}"
    )
    print(
        f"false | {df_false[(df_false['text_filter_correct_result'] == 0) & (df_false['extraction_correct_result'] == 0) & (df_false['classification_correct_result'] == 0)]['instance_count'].sum()} | {df_false[(df_false['text_filter_correct_result'] == 1) | (df_false['extraction_correct_result'] == 1) | (df_false['classification_correct_result'] == 1)]['instance_count'].sum()}"
    )
