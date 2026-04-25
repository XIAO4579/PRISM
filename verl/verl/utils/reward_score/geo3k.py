# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from mathruler.grader import grade_answer


def extract_answer_content(predict_str: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r".*<caption>.*</caption>.*<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
    return 1.0 if pattern.search(predict_str) else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_answer_tag: bool = True) -> float:
    if use_answer_tag:
        answer = extract_answer_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_answer_tag: bool = True, format_score: float = 0.1) -> dict:
    fmt = format_reward(predict_str)
    acc = acc_reward(predict_str, ground_truth, use_answer_tag)
    total = (1.0 - format_score) * acc + format_score * fmt
    return {
        "score": total,
        "acc_reward": acc,
        "format_reward": fmt,
    }
