import os
import re

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.tasks.hallusion_bench.utils import (
    assign_correctness,
    evaluate_by_chatgpt,
    get_eval_all,
    get_eval_fig,
    get_eval_pair_all,
)

cur_dir = os.path.dirname(os.path.abspath(__file__))
output_entry = "model_prediction"
correctness_entry = "gpt4v_output_gpt_check"

metric = ["aAcc", "fAcc", "qAcc"]


def _extract_answer_tag(text):
    """Extract answer from <answer></answer> tags."""
    match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text)
    if match:
        return match.group(1).strip()
    return text


def _rule_based_yes_no(prediction, gt_answer):
    """Rule-based yes/no matching. Returns '1'(correct), '0'(incorrect), or None(uncertain)."""
    pred_lower = prediction.lower().strip().rstrip(".")
    pred_answer = None
    if pred_lower in ("yes", "correct", "true", "right"):
        pred_answer = "1"
    elif pred_lower in ("no", "incorrect", "false", "wrong"):
        pred_answer = "0"
    elif pred_lower.startswith("yes,") or pred_lower.startswith("yes.") or pred_lower.startswith("yes ") or pred_lower == "yes":
        pred_answer = "1"
    elif pred_lower.startswith("no,") or pred_lower.startswith("no.") or pred_lower.startswith("no ") or pred_lower == "no":
        pred_answer = "0"

    if pred_answer is not None:
        return "1" if pred_answer == str(gt_answer) else "0"
    return None


def hb_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def hb_doc_to_visual(doc):
    """Convert document to visual input."""
    num_image = int(os.environ.get("NUM_IMAGE", "1"))

    if num_image == 1:
        return [doc["image"].convert("RGB")]
    elif num_image == 2:
        return [doc["image"].convert("RGB"), doc["image"].convert("RGB")]
    else:
        raise ValueError(f"num_image must be 1 or 2, got {num_image}")


def hb_process_results(doc, result):
    sample = {k: v for k, v in doc.items() if k != "image"}
    raw_prediction = result[0]
    extracted = _extract_answer_tag(raw_prediction)
    sample["model_prediction"] = extracted
    sample["raw_prediction"] = raw_prediction
    return {k: sample for k in metric}


def hb_aggregation_result(results, metric, args):
    data_vd = []
    data_vs = []
    for data in tqdm(results, desc="Split vd and vs"):
        if data["category"] == "VD":
            data_vd.append(data)
        if data["category"] == "VS":
            data_vs.append(data)

    # Rule-based pre-evaluation: try yes/no matching before GPT judge
    rule_matched = 0
    for data_list in [data_vd, data_vs]:
        for sample in data_list:
            rule_result = _rule_based_yes_no(sample[output_entry], sample.get("gt_answer"))
            if rule_result is not None:
                sample[correctness_entry] = rule_result
                rule_matched += 1
    eval_logger.info(f"Rule-based pre-evaluation matched {rule_matched}/{len(results)} samples, rest fall back to GPT judge")

    eval_logger.info("Do gpt eval vd ...")
    path = os.path.join(args.output_path, "gpt_response")
    os.makedirs(path, exist_ok=True)

    model_id = "model"
    if hasattr(args, "model_args") and args.model_args:
        for kv in args.model_args.split(","):
            if kv.startswith("model=") or kv.startswith("pretrained="):
                model_id = os.path.basename(kv.split("=", 1)[1].rstrip("/"))
                break

    save_json_path_vd = f"{path}/hallusion_output_vd_{model_id}.json"
    save_json_path_vs = f"{path}/hallusion_output_vs_{model_id}.json"
    data_vd = evaluate_by_chatgpt(data_vd, output_entry=output_entry, correctness_entry=correctness_entry, load_json=True, save_json_path=save_json_path_vd)
    data_vd = assign_correctness(data_vd, correctness_entry=correctness_entry)
    eval_logger.info("Do gpt eval vs")
    data_vs = evaluate_by_chatgpt(data_vs, output_entry=output_entry, correctness_entry=correctness_entry, load_json=True, save_json_path=save_json_path_vs)
    data_vs = assign_correctness(data_vs, correctness_entry=correctness_entry)
    results = data_vs + data_vd

    if metric == "aAcc":
        all_data = get_eval_all(results, model_correctness_entry=correctness_entry)
        return round(100 * all_data["correct"] / all_data["total"], 4)
    elif metric == "fAcc":
        fig_all = get_eval_fig(results)
        return round(100 * fig_all["correct"] / fig_all["total"], 4)
    elif metric == "qAcc":
        all_data = get_eval_pair_all(results, model_correctness_entry=correctness_entry)
        return round(100 * all_data["correct"] / all_data["total"], 4)


def hb_aggregation_result_qAcc(results, args):
    return hb_aggregation_result(results, "qAcc", args)


def hb_aggregation_result_fAcc(results, args):
    return hb_aggregation_result(results, "fAcc", args)


def hb_aggregation_result_aAcc(results, args):
    return hb_aggregation_result(results, "aAcc", args)


def hb_aggregation_result_intern(results, metric):
    scores = []
    for result in results:
        ans = "1" if result["model_prediction"].lower().find("yes") != -1 else "0"
        scores.append(ans == result["gt_answer"])
        result["answer"] = ans

    if metric == "aAcc":
        return sum(scores) / len(scores)
    elif metric == "qAcc":
        qlist = {}
        for r in results:
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            try:
                qlist[key].append(r["answer"] == r["gt_answer"])
            except:
                qlist[key] = [r["answer"] == r["gt_answer"]]
        out = []
        for q, v in qlist.items():
            out.append(min(v))

        return sum(out) / len(out)
    elif metric == "fAcc":
        qlist = {}
        for r in results:
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"])])
            try:
                qlist[key].append(r["answer"] == r["gt_answer"])
            except:
                qlist[key] = [r["answer"] == r["gt_answer"]]
        out = []
        for q, v in qlist.items():
            out.append(min(v))
        return sum(out) / len(out)


def hb_aggregation_result_qAcc_intern(results):
    eval_logger.info("Calculating qAcc ...")
    return hb_aggregation_result_intern(results, "qAcc")


def hb_aggregation_result_fAcc_intern(results):
    eval_logger.info("Calculating fAcc ...")
    return hb_aggregation_result_intern(results, "fAcc")


def hb_aggregation_result_aAcc_intern(results):
    eval_logger.info("Calculating aAcc ...")
    return hb_aggregation_result_intern(results, "aAcc")
