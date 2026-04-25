import re
import os
import time
import random
import logging

from openai import OpenAI

logger = logging.getLogger("mm_gad")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(name)s %(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(getattr(logging, os.environ.get("MM_GAD_LOG_LEVEL", "INFO").upper(), logging.INFO))

MAX_RETRIES = int(os.environ.get("MM_GAD_MAX_RETRIES", "2"))
RETRY_BACKOFF = float(os.environ.get("MM_GAD_RETRY_BACKOFF", "2.0"))
REQUEST_TIMEOUT = float(os.environ.get("MM_GAD_TIMEOUT", "60"))

openai_api_key = ""
openai_api_base_list = [
    "",
]

client_list = [
    OpenAI(api_key=openai_api_key, base_url=api_base, timeout=REQUEST_TIMEOUT)
    for api_base in openai_api_base_list
]

MODEL_NAME = "gpt-4o-mini"

FORMAT_PATTERN = re.compile(
    r"^\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _get_judge_prompt(model_answer: str, ground_truth: str, question: str = "") -> str:
    chat_template = """
    Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, 
    and [Model_answer] is the answer extracted from a model's output to this question. 

    Judge how consistent the two answers are.

    Scoring rules  
    • 1    — Fully consistent: they convey the same meaning (e.g., "pink" vs. "it is pink"), OR the [Model_answer] covers all the key information in the [Standard Answer] even if it includes additional details (e.g., Standard Answer is "cat" and Model_answer is "a cat sitting on the table").  
    • 0    — Inconsistent: they conflict or share no essential overlap.

    Output **only** one of the following numbers: 1, or 0.
    """

    question_display = question if question else "N/A"
    test_prompt = f"""
    [Question]: {question_display}
    [Standard Answer]: {ground_truth}
    [Model_answer] : {model_answer}
    Judgement:"""

    return chat_template + test_prompt


def _llm_judge(model_answer: str, ground_truth: str, question: str = "") -> float:
    prompt = _get_judge_prompt(model_answer, ground_truth, question)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        client = random.choice(client_list)
        try:
            logger.debug("Attempt %d/%d — calling %s", attempt, MAX_RETRIES, MODEL_NAME)
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                seed=42,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
            logger.debug("Raw judge response: %r", text)
            break
        except Exception as e:
            last_err = e
            wait = RETRY_BACKOFF ** attempt + random.uniform(0, 1)
            logger.warning(
                "LLM judge error on attempt %d/%d: %s. Retrying in %.1fs...",
                attempt, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)
    else:
        logger.error(
            "LLM judge failed after %d retries. Last error: %s", MAX_RETRIES, last_err
        )
        return 0.0

    if "Judgement:" in text:
        text = text.split("Judgement:")[-1].strip()

    if "1" in text:
        return 1.0
    elif "0" in text:
        return 0.0
    else:
        logger.warning("Unexpected judge response: %r — defaulting to 0.0", text)
        return 0.0


def compute_score(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> dict:
    """Compute score for mm_gad mode using LLM-as-judge.

    Format check: <caption>...</caption><think>...</think><answer>...</answer>
    If format is invalid, all scores are 0.
    If format is valid, extract the answer and call LLM judge for acc_reward.
    """
    if not FORMAT_PATTERN.match(predict_str):
        logger.debug("Format check failed, returning all zeros")
        return {"score": 0.0, "acc_reward": 0.0, "format_reward": 0.0}

    answer_text = _extract_answer(predict_str)
    if not answer_text:
        logger.debug("Empty answer extracted")
        return {"score": 0.0, "acc_reward": 0.0, "format_reward": 1.0}
    if len(answer_text) >= 1024:
        logger.warning("Answer too long (%d chars), skipping judge", len(answer_text))
        return {"score": 0.0, "acc_reward": 0.0, "format_reward": 1.0}

    question = ""
    if extra_info and isinstance(extra_info, dict):
        question = extra_info.get("question", "")

    acc = _llm_judge(answer_text, ground_truth, question)
    logger.info("acc_reward=%.1f | answer=%s | gt=%s", acc, answer_text[:80], ground_truth[:80])

    return {"score": acc, "acc_reward": acc, "format_reward": 1.0}
