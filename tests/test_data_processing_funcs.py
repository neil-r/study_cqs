from study_llm.event_extraction_task.prompts import extract_json_list


def test_extract_json_list():
    l = extract_json_list('This is is text, ["one","two"] three.')

    assert isinstance(l, list)
    assert l[0] == "one"
    assert l[1] == "two"
    assert len(l) == 2

    l = extract_json_list('This is is text, [   ] three.')
    assert isinstance(l, list)
    assert len(l) == 0

    l = extract_json_list('This is is text, one two three.')
    assert l is None
