"""DeepVision-style math_verify scoring with mm_gad-style format gating.

- Format check: ``<caption>...</caption><think>...</think><answer>...</answer>``
  (identical to mm_gad_no_llm.py). If the format is invalid, all rewards are 0.
- Accuracy: uses math_verify library to verify the FULL model output against
  ``ground_truth`` and every entry in ``equivalent_answers``. This matches
  DeepVision's math_verify.compute_score behavior, including the ``__EMPTY__``
  placeholder convention.
- Final score: ``0.8 * acc + 0.2 * format_reward`` (aligned with mm_gad).
"""
import logging
import re

logger = logging.getLogger("math_verify_with_format")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(name)s %(asctime)s %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        StringExtractionConfig,
    )

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False
    print(
        "To use math_verify_with_format, please install math-verify first: "
        "`pip install math-verify`."
    )


FORMAT_PATTERN = re.compile(
    r"^\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _math_verify_acc(
    model_output: str,
    ground_truth: str,
    equivalent_answers=None,
    timeout_score: float = 0.0,
) -> float:
    """Mirror DeepVision's math_verify.compute_score: 1.0 if any of
    ``[ground_truth] + equivalent_answers`` verifies against model_output."""
    if not HAS_MATH_VERIFY:
        return 0.0

    verify_func = math_metric(
        gold_extraction_target=(
            LatexExtractionConfig(),
            ExprExtractionConfig(),
            StringExtractionConfig(),
        ),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
            StringExtractionConfig(),
        ),
    )

    # Handle the DeepVision __EMPTY__ placeholder convention.
    if equivalent_answers and "__EMPTY__" in equivalent_answers:
        equivalent_answers = None

    all_valid = [ground_truth]
    if equivalent_answers:
        all_valid.extend(list(equivalent_answers))

    for answer in all_valid:
        try:
            answer_boxed = "\\boxed{" + str(answer) + "}"
            score, _ = verify_func([answer_boxed], [model_output])
            if score > 0:
                return 1.0
        except TimeoutException:
            return timeout_score
        except Exception:
            continue

    return 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    equivalent_answers=None,
    extra_info=None,
    **kwargs,
) -> dict:
    """Return ``{"score", "acc_reward", "format_reward"}``.

    ``equivalent_answers`` may be passed directly or nested inside
    ``extra_info`` (for backward-compatible callers that route reward_model
    fields through extra_info).
    """
    # Backward-compat: also look in extra_info.
    if equivalent_answers is None and isinstance(extra_info, dict):
        equivalent_answers = extra_info.get("equivalent_answers")

    if not FORMAT_PATTERN.match(predict_str):
        logger.debug("Format check failed, returning all zeros")
        return {"score": 0.0, "acc_reward": 0.0, "format_reward": 0.0}

    answer_text = _extract_answer(predict_str)
    if not answer_text:
        logger.debug("Empty answer extracted")
        return {"score": 0.2, "acc_reward": 0.0, "format_reward": 1.0}

    if len(answer_text) >= 1024:
        logger.warning("Answer too long (%d chars), skipping acc judge", len(answer_text))
        return {"score": 0.2, "acc_reward": 0.0, "format_reward": 1.0}

    # Feed the extracted <answer> content to math_verify (wrapped in \boxed{}
    # so its LatexExtractionConfig picks it up reliably).
    answer_boxed_pred = "\\boxed{" + answer_text + "}"
    acc = _math_verify_acc(answer_boxed_pred, ground_truth, equivalent_answers)
    format_reward = 1.0
    score = 0.8 * acc + 0.2 * format_reward

    return {"score": score, "acc_reward": acc, "format_reward": format_reward}
