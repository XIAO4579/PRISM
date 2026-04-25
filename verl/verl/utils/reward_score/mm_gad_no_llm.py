import logging
import re

logger = logging.getLogger("mm_gad")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(name)s %(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

try:
    from math_verify import parse, verify

    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

FORMAT_PATTERN = re.compile(
    r"^\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)

_UNIT_ALTERNATIVES = (
    r"°|°C|°F|%|meters?|cm|mm|km|inches?|ft|feet|"
    r"kg|mg|lb|lbs|pounds?|"
    r"seconds?|sec|minutes?|min|hours?|hr|days?|years?|"
    r"plates?|pieces?|items?|people|persons?|students?|times?|"
    r"dm|ml|mL|liters?|litres?|cm[²³]|m[²³]|"
    r"元|米|厘米|千克|克|度|个|条|张|块|人|次"
)
_UNIT_PATTERN = re.compile(
    rf"(?<=\d)\s*({_UNIT_ALTERNATIVES})\s*$",
    re.IGNORECASE,
)
_MCQ_GT_PATTERN = re.compile(
    r"^(?:"
    r"\(([A-Ha-h])\)\s+.+"
    r"|([A-Ha-h])\.\s+(?=[(\[{0-9+\-]).+"
    r")$",
    re.DOTALL,
)
_MCQ_SINGLE_PATTERN = re.compile(
    r"^\s*[\(\[\{]?\s*([A-Ha-h])\s*[\)\]\}]?\s*[.:]?\s*$"
)
_MCQ_PAREN_TOKEN_PATTERN = re.compile(
    r"(?:^|[\s,;:])\(([A-Ha-h])\)(?=$|[\s,;:.!?])"
)
_MCQ_DOT_TOKEN_PATTERN = re.compile(
    r"(?:^|[\s,;:])([A-Ha-h])[.)](?=$|[\s,;:.!?])"
)
_MCQ_ANSWER_PHRASE_PATTERN = re.compile(
    r"\b(?:the\s+)?(?:correct\s+)?(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)\s*[\(\[\{]?\s*([A-Ha-h])\s*[\)\]\}]?(?=$|[\s,;:.!?])",
    re.IGNORECASE,
)


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _strip_latex_delimiters(text: str) -> str:
    text = text.strip()
    for left, right in [("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left) : -len(right)].strip()
    return text


def _normalize_latex(text: str) -> str:
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\tfrac", "\\frac")
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    text = re.sub(r"\{\s*", "{", text)
    text = re.sub(r"\s*\}", "}", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _simple_parse(text: str) -> str:
    text = text.strip()
    text = _strip_latex_delimiters(text)
    text = _normalize_latex(text)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = text.replace("，", "")
    if text.endswith("."):
        text = text[:-1]
    return text.strip()


def _strip_units(text: str) -> str:
    return _UNIT_PATTERN.sub("", text).strip()


def _extract_trailing_unit(text: str) -> str:
    m = _UNIT_PATTERN.search(text.strip())
    return m.group(1).lower() if m else ""


def _parse_mcq(predict_str: str) -> str:
    if not predict_str or predict_str.strip() == "":
        return ""

    response = predict_str.strip()
    m = _MCQ_SINGLE_PATTERN.match(response)
    if m:
        return m.group(1).upper()

    candidates = []
    for m in _MCQ_ANSWER_PHRASE_PATTERN.finditer(response):
        candidates.append((m.group(1).upper(), m.start(), "answer_phrase"))
    for m in _MCQ_PAREN_TOKEN_PATTERN.finditer(response):
        candidates.append((m.group(1).upper(), m.start(), "parentheses"))
    for m in _MCQ_DOT_TOKEN_PATTERN.finditer(response):
        candidates.append((m.group(1).upper(), m.start(), "dot_or_right_paren"))

    if candidates:
        format_priority = {
            "answer_phrase": 10,
            "parentheses": 8,
            "dot_or_right_paren": 6,
        }
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]
    return ""


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）\[\]【】{}\",;:!?'\"().]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _try_parse_number(text: str):
    text = re.sub(r"(\d),(\d)", r"\1\2", text.strip()).replace("，", "")
    text = _strip_units(text)
    try:
        return float(text)
    except ValueError:
        return None


def _is_mcq_ground_truth(gt: str):
    gt = gt.strip()
    if gt in ("A", "B", "C", "D", "E", "F", "G", "H"):
        return True, gt
    m = _MCQ_GT_PATTERN.match(gt)
    if m:
        letter = (m.group(1) or m.group(2)).upper()
        return True, letter
    return False, ""


def _relax_exact_match(predict_str: str, ground_truth: str) -> float:
    is_mcq, mcq_letter = _is_mcq_ground_truth(ground_truth)
    if is_mcq:
        parsed = _parse_mcq(predict_str)
        return 1.0 if parsed == mcq_letter else 0.0

    if predict_str.strip() == ground_truth.strip():
        return 1.0

    pred_num = _try_parse_number(predict_str)
    gt_num = _try_parse_number(ground_truth)
    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        return 1.0 if abs(pred_num - gt_num) / abs(gt_num) < 1e-6 else 0.0

    if _normalize_answer(predict_str) == _normalize_answer(ground_truth):
        return 1.0

    pred_no_unit = _strip_units(predict_str)
    gt_no_unit = _strip_units(ground_truth)
    if pred_no_unit and gt_no_unit and _normalize_answer(pred_no_unit) == _normalize_answer(gt_no_unit):
        pred_unit = _extract_trailing_unit(predict_str)
        gt_unit = _extract_trailing_unit(ground_truth)
        if not pred_unit and not gt_unit:
            return 1.0
        if pred_unit and gt_unit and pred_unit == gt_unit:
            return 1.0

    return 0.0


def _answer_variants(text: str):
    results = [text]
    stripped = _strip_latex_delimiters(text)
    if stripped != text:
        results.append(stripped)
    normalized = _normalize_latex(stripped)
    if normalized != stripped:
        results.append(normalized)
    # 注意: 这里不加入去单位 variant，避免 5kg vs 5lb 被 math_verify 误判为等价
    return list(dict.fromkeys(results))


def _math_verify_match(predict_str: str, ground_truth: str) -> float:
    if not HAS_MATH_VERIFY:
        return 0.0

    if len(_split_multi_value(predict_str)) > 1 or len(_split_multi_value(ground_truth)) > 1:
        return 0.0

    variants_pred = _answer_variants(predict_str)
    variants_gt = _answer_variants(ground_truth)

    for gt_text in variants_gt:
        try:
            gold = parse(gt_text)
        except Exception:
            continue
        for pred_text in variants_pred:
            try:
                pred = parse(pred_text)
                if verify(gold, pred):
                    return 1.0
            except Exception:
                continue
    return 0.0


_NUMBERED_ITEM_RE = re.compile(r'\(\d+\)\s*')


def _split_by_top_level_comma(text: str) -> list:
    """Split by commas that are NOT nested inside parentheses/brackets/braces."""
    parts = []
    depth = 0
    current = []
    for char in text:
        if char in '({[':
            depth += 1
            current.append(char)
        elif char in ')}]':
            depth = max(0, depth - 1)
            current.append(char)
        elif char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def _split_multi_value(text: str) -> list:
    """Split a multi-value answer into individual parts.

    Handles numbered items like ``(1) ... (2) ...``, semicolons, and
    top-level commas.  Returns a single-element list for ordinary answers.
    """
    text = text.strip()

    markers = list(_NUMBERED_ITEM_RE.finditer(text))
    if len(markers) >= 2 and markers[0].start() <= 2:
        parts = []
        for i, m in enumerate(markers):
            start = m.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            part = text[start:end].strip().rstrip(',;，；')
            if part:
                parts.append(part)
        if len(parts) >= 2:
            return parts

    parts = re.split(r'\s*[;；]\s*', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts

    parts = _split_by_top_level_comma(text)
    if len(parts) >= 2:
        return parts

    return [text]


_COORD_TUPLE_PATTERN = re.compile(
    r"\(?\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)?"
)


def _extract_coord_tuples(text: str):
    """Extract all (x, y) coordinate pairs from text as list of (float, float)."""
    matches = _COORD_TUPLE_PATTERN.findall(text)
    if not matches:
        return None
    coords = []
    for m in matches:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", m)
        if len(nums) == 2:
            coords.append((float(nums[0]), float(nums[1])))
    return coords if coords else None


def _coord_match(pred_text: str, gt_text: str) -> float:
    """Compare coordinate tuples element-wise (order-sensitive)."""
    pred_coords = _extract_coord_tuples(pred_text)
    gt_coords = _extract_coord_tuples(gt_text)
    if pred_coords is None or gt_coords is None:
        return -1.0  # not coordinate-type, signal caller to skip
    if len(pred_coords) != len(gt_coords):
        return 0.0
    for (px, py), (gx, gy) in zip(pred_coords, gt_coords):
        if abs(px - gx) > 1e-6 or abs(py - gy) > 1e-6:
            return 0.0
    return 1.0


def _single_value_judge(pred: str, gt: str) -> float:
    """Judge a single predicted value against a single ground-truth value."""
    coord_score = _coord_match(pred, gt)
    if coord_score >= 0:
        return coord_score

    score = _relax_exact_match(pred, gt)
    if score > 0:
        return score

    score = _math_verify_match(pred, gt)
    if score > 0:
        return score

    return 0.0


def _rule_based_judge(model_answer: str, ground_truth: str) -> float:
    gt_parts = _split_multi_value(ground_truth)
    pred_parts = _split_multi_value(model_answer)

    if len(gt_parts) > 1 or len(pred_parts) > 1:
        if len(pred_parts) != len(gt_parts):
            return 0.0
        for p, g in zip(pred_parts, gt_parts):
            if _single_value_judge(_simple_parse(p), _simple_parse(g)) <= 0:
                return 0.0
        return 1.0

    pred = _simple_parse(model_answer)
    gt = _simple_parse(ground_truth)
    return _single_value_judge(pred, gt)


def compute_score(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> dict:
    """Compute score for mm_gad mode using pure rule matching.

    Format check: <caption>...</caption><think>...</think><answer>...</answer>
    If format is invalid, all scores are 0.
    If format is valid, extract the answer and compute accuracy by rules.
    """
    if not FORMAT_PATTERN.match(predict_str):
        logger.debug("Format check failed, returning all zeros")
        return {"score": 0.0, "acc_reward": 0.0, "format_reward": 0.0}

    answer_text = _extract_answer(predict_str)
    if not answer_text:
        logger.debug("Empty answer extracted")
        return {"score": 0.2, "acc_reward": 0.0, "format_reward": 1.0}
    if len(answer_text) >= 1024:
        logger.warning("Answer too long (%d chars), skipping rule judge", len(answer_text))
        return {"score": 0.2, "acc_reward": 0.0, "format_reward": 1.0}

    acc = _rule_based_judge(answer_text, ground_truth)
    format_reward = 1.0
    score = 0.8 * acc + 0.2 * format_reward
    # logger.info("score=%.2f | acc_reward=%.1f | answer=%s | gt=%s", score, acc, answer_text[:80], ground_truth[:80])

    return {"score": score, "acc_reward": acc, "format_reward": format_reward}
