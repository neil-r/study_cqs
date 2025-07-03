"""
Some of the answer responses are not parsed given a token limit being
reached. This script fixes this by seeing if some terms used in the
JSON objects are detected in the partial JSON result. Note that
to analyze the content of the JSON objects beyond just empty
list check, these results should be re-executed (not applicable
for the study's focus).
"""

from driver_11_prep_db import database


t_index = 0
while True:
    tr = database.get_task_result(t_index)
    if tr is None or len(tr) == 0:
        break
    tr_e = tr[0]
    if tr_e.answer_response is None:  # or tr_e.answer_response == "JSON List":
        et_index = tr_e.log[1]["content"].find("event_type")
        if et_index < 0:
            et_index = tr_e.log[1]["content"].find('type"')
        if et_index > 0:
            database.update_task_answer(
                tr_e.evalutation_id,
                answer_response="JSON List",
                correct=1 if tr_e.answer_value == "1" else 0,
            )
            print(et_index)
    t_index += 1
