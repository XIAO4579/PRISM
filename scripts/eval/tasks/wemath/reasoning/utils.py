import os

import pandas as pd
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.wemath.wemath_utils import (
    calculate_metrics,
    compute_final_scores,
    process_steps_data,
    update_main_results_df,
)

API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

_CLEAN_PROMPT = os.getenv("USE_CLEAN_PROMPT", "").lower() in ("true", "1", "yes")

server_config = ServerConfig(model_name=GPT_MODEL)
server = get_server(server_name=API_TYPE, config=server_config)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + "\n" + doc["option"]


def wemath_doc_to_visual(doc):
    return [doc["image_path"].convert("RGB")]


def wemath_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = wemath_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    if not _CLEAN_PROMPT:
        system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        messages = system_messages + messages
    return messages


def wemath_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = wemath_doc_to_text_cot(doc, None)
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="wemath", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
        single_acc = score_dict["acc_score"]

        # LLM judge fallback when rule-based scoring says wrong
        if single_acc == 0:
            try:
                judge_result = server.evaluate_binary(
                    question=question,
                    answer=str(doc["answer"]),
                    prediction=pred.strip(),
                    output_format="0/1",
                )
                if judge_result["success"] and judge_result["result"] == 1:
                    single_acc = 1.0
            except Exception as e:
                eval_logger.error(f"LLM judge fallback error: {e}")

        acc_score += single_acc
        format_score += score_dict.get("format_reward_score", 0.0)

    data_dict = {
        "ID": doc["ID"],
        "split": doc["split"],
        "knowledge concept": doc["knowledge concept"],
        "question": doc["question"],
        "option": doc["option"],
        "answer": doc["answer"],
        # "image_path": doc['image_path'],
        "key": doc["key"],
        "question number": doc["question number"],
        "knowledge concept description": doc["knowledge concept description"],
        "acc_score": acc_score,
    }

    return {"wemath_loose": data_dict, "wemath_strict": data_dict, "acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}


def wemath_aggregate_results(results, metric_name):
    data = pd.DataFrame(results)
    data["joker"] = data["acc_score"] == 1.0
    data_2steps = data[data["key"].str.contains("2steps")]
    data_3steps = data[data["key"].str.contains("3steps")]
    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_dict = update_main_results_df(total_counts, rates)
    if metric_name == "wemath_loose":
        return score_dict["Score (Loose)"]
    elif metric_name == "wemath_strict":
        return score_dict["Score (Strict)"]
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")


def wemath_aggregate_results_loose(results):
    return wemath_aggregate_results(results, "wemath_loose")


def wemath_aggregate_results_strict(results):
    return wemath_aggregate_results(results, "wemath_strict")
