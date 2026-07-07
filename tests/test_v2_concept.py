from src.md_manifest import load_md_manifest
from src.v2_concept import EXPLICIT_TEMPLATES, _cue_prompt_records


def test_explicit_templates_are_distinct_and_leave_a_completion_slot():
    rendered = {name: function("A hidden clue") for name, function in EXPLICIT_TEMPLATES.items()}
    assert len(set(rendered.values())) == len(rendered)
    assert all("A hidden clue" in prompt for prompt in rendered.values())
    assert all(prompt.endswith(":") for prompt in rendered.values())


def test_question_template_does_not_embed_the_answer():
    prompt = EXPLICIT_TEMPLATES["question_one_word"]("A ringed gas giant")
    assert "Saturn" not in prompt
    assert "Answer (one word):" in prompt


def test_cue_records_exclude_common_anchor_from_span():
    payload = load_md_manifest()
    records = _cue_prompt_records(payload, "train")
    first = records[sorted(records)[0]][0]
    cue = first["prompt"][first["cue_char_start"] : first["cue_char_end"]]
    assert cue
    assert payload["anchor_suffix"] not in cue
    assert first["prompt"].endswith(payload["anchor_suffix"])
