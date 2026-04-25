import json
import re
from pathlib import Path

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mathverse.mathverse_evals import MathVerseEvaluator

with open(Path(__file__).parent / "mathverse.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

mathverse_evaluator = MathVerseEvaluator()


def mathverse_doc_to_visual(doc):
    if str(doc["image"]).strip() == "":
        return []
    return [doc["image"].convert("RGB")]


def mathverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    problem = {
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_type": doc["question_type"],
        "problem_version": doc["problem_version"],
    }
    query_prompt = mathverse_evaluator.create_one_query(
        problem, examples=None, shot_num=0, shot_type=lmms_eval_specific_kwargs["shot_type"], hint=lmms_eval_specific_kwargs.get("hint", None), query_type=lmms_eval_specific_kwargs["query_type"]
    )
    return query_prompt


def mathverse_process_results(doc, results):
    prediction = results[0].strip()
    question = doc["question_for_eval"]
    answer = doc["answer"] if "answer" in doc else None

    judge_result = 0
    if answer is not None:
        # Step 1: Extract from <answer></answer> tag
        answer_tag_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", prediction)
        extracted = answer_tag_match.group(1).strip() if answer_tag_match else None

        # Step 2: Rule-based matching with extracted answer
        if extracted is not None:
            extracted_norm = extracted.strip().lower().rstrip(".")
            answer_norm = str(answer).strip().lower().rstrip(".")
            if extracted_norm == answer_norm:
                judge_result = 1
            elif doc["question_type"] == "multi-choice":
                extracted_letter = extracted.strip().upper()
                answer_letter = str(answer).strip().upper()
                if len(extracted_letter) == 1 and extracted_letter == answer_letter:
                    judge_result = 1

        # Step 3: LLM judge fallback
        if judge_result == 0:
            judge_result = 1 if mathverse_evaluator.score_answer(question, answer, prediction, quick_match=False) else 0

    result = {
        "sample_index": doc["sample_index"],
        "problem_index": doc["problem_index"],
        "problem_version": doc["problem_version"],
        "question": doc["question"],
        "answer": doc["answer"] if "answer" in doc else None,
        "prediction": prediction,
        "question_type": doc["question_type"],
        "metadata": doc["metadata"],
        "query_wo": doc["query_wo"],
        "query_cot": doc["query_cot"],
        "question_for_eval": doc["question_for_eval"],
        "true_false": judge_result == 1,
    }

    return {
        "gpt_eval_score": judge_result,
        "submission": result,
    }


def mathverse_aggregate_results_submission(results, args, *, calculate_gain=False, random_scores=None):
    # Don't know why but this sometimes yields error so I hardcode it
    try:
        split_flag = results[0]["metadata"]["split"]
    except:
        split_flag = "testmini"
    path = generate_submission_file(f"mathverse_{split_flag}_results.json", args)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Saved results to {path}")


def mathverse_aggregate_results_eval(results, args, *, calculate_gain=False, random_scores=None):
    scores = sum(results) / len(results)
    return scores
