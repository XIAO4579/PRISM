
import ast
import os
import random
import re

import concurrent.futures
from openai import OpenAI
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
import base64
import io
from PIL import Image

import requests

# from math_verify import parse, verify
from openai import OpenAI

# from rouge_score import rouge_scorer

openai_api_key = ""
openai_api_base_list = [
    "",
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)

def get_chat_template():
    chat_template = """
    Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, 
    and [Model_answer] is the answer extracted from a model's output to this question. 

    Judge how consistent the two answers are.

    Scoring rules  
    • 1    — Fully consistent: they convey the same meaning (e.g., “pink” vs. “it is pink”).  
    • 0    — Inconsistent: they conflict or share no essential overlap.

    Output **only** one of the following numbers: 1, or 0.
    """

    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
    [Question]: Is the countertop tan or blue?
    [Standard Answer]: The countertop is tan.
    [Model_answer] : tan
    Judgement: 1
    """  # noqa

    example_2 = """
    [Question]: On which side of the picture is the barrier?
    [Standard Answer]: The barrier is on the left side of the picture.
    [Model_answer] : left
    Judgement: 1
    """  # noqa

    example_3 = """
    [Question]: Is the man phone both blue and closed?
    [Standard Answer]: Yes, the man phone is both blue and closed.
    [Model_answer] : No.
    Judgement: 0
    """  # noqa

    example_4 = """
    [Question]: What color is the towel in the center of the picture?
    [Standard Answer]: The towel in the center of the picture is blue.
    [Model_answer] : The towel in the center of the picture is pink.
    Judgement: 0
    """  # noqa

    return [example_1, example_2, example_3, example_4]

def cut_sentences(text: str) -> list[str]:
    """
    Extract content from <think> tags and cut it into a list of sentences based on the '.' character.
    Removes empty strings and strips whitespace.
    """
    if not text:
        return []
    
    # Extract content between <think> and </think>
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not match:
        return []
        
    think_content = match.group(1)
    
    # Split by '.', strip whitespace, and filter out empty strings
    sentences = [s.strip() for s in think_content.split('.') if s.strip()]
    return sentences

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
    [Question]: {question}
    [Standard Answer]: {ground_truth}
    [Model_answer] : {predict_str}
    Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt

def extract_answer(text):
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

prompt_score_independent="""
You are a strict evaluator for visual reasoning.

Input:
1. An image.
2. A question about the image:{problem_text}
3. A reasoning step to evaluate:{step_text}

Your task is to judge whether this reasoning step is factually correct and relevant, based ONLY on the image and the question.

Evaluation criteria:
- The step must be logically valid on its own.
- The step must be consistent with and supported by the given image.
- The step must be relevant to answering the question.
- If the step is incorrect, unsupported by the image, irrelevant to the question, or factually wrong, score it as 0.
- Only if the step is completely correct, factually accurate, and relevant, score it as 1.0.

Output format (STRICT):
- Output ONLY a score wrapped in <score> and </score>.
- Do NOT output explanations, reasoning, or any other text.
- The score must be either 0 or 1.0.

Example outputs:
<score>0</score>
<score>1.0</score>
"""

prompt_score="""
You are a strict evaluator for visual reasoning.

Input:
1. An image.
2. A question about the image:{problem_text}
3. Previous reasoning steps (if any):{steps_before}
4. The current reasoning step to evaluate:{step_text}

Your task is to judge whether this current reasoning step is fully correct, given the image, question, and previous steps.

Evaluation criteria:
- The step must be logically valid.
- The step must be consistent with the given image.
- The step must correctly contribute toward answering the question.
- The step must logically follow from the previous reasoning steps (if any).
- The step must not contradict or repeat any previous steps.
- If the step is incorrect, partially correct, ambiguous, unsupported by the image, logically flawed, or inconsistent with previous steps, it should be scored as 0.
- Only if the step is completely correct, precise, and coherent with the reasoning chain, score it as 1.0.

Output format (STRICT):
- Output ONLY a score wrapped in <score> and </score>.
- Do NOT output explanations, reasoning, or any other text.
- The score must be either 0 or 1.0.

Example outputs:
<score>0</score>
<score>1.0</score>
"""

# Configuration
BASE_URL = ""
API_KEY = ""
MODEL_NAME = "qwen3-vl-thinking"

def get_client():
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

def encode_image_to_base64(image_input: Union[str, Image.Image]) -> str:
    """
    将图片输入（路径或 PIL.Image 对象）转换为 Base64 字符串。
    如果输入已经是 data:image 开头的字符串或 http 开头的 URL，则直接返回。
    """
    if isinstance(image_input, str):
        if image_input.startswith("data:image") or image_input.startswith("http"):
            return image_input
        
        # 假设是文件路径
        if os.path.exists(image_input):
            try:
                with Image.open(image_input) as img:
                    return image_to_base64_str(img)
            except Exception as e:
                print(f"无法读取图片文件 {image_input}: {e}")
                raise e
        else:
             # 可能是无效路径，或者本意就是字符串，但在这种上下文中大概率是错误
             # 这里为了容错，如果看起来不像路径且不是 url，可能抛错
             raise ValueError(f"无效的图片输入: {image_input}")
             
    elif isinstance(image_input, Image.Image):
        return image_to_base64_str(image_input)
    
    raise ValueError(f"不支持的图片输入类型: {type(image_input)}")

def image_to_base64_str(pil_image: Image.Image) -> str:
    """Helper to convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    # Convert to RGB if necessary (e.g. for RGBA images saving as JPEG, though PNG handles RGBA)
    # 统一存为 PNG 以保持质量
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def process_single_score_request(client: OpenAI, image_data: str, question: str, text: str, steps_before: str = "", model: str = MODEL_NAME) -> str:
    """
    处理单个评分请求。
    image_data: 已经是处理好的 Base64 字符串或 URL
    steps_before: 之前的推理步骤（如果是第一步，则为空字符串或 "None"）
    """
    # 处理 steps_before，如果为空则显示 "None"
    steps_before_display = steps_before if steps_before.strip() else "None (this is the first step)"
    
    # print(f"steps_before_display: {steps_before_display}")

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_score.format(problem_text=question, steps_before=steps_before_display, step_text=text)},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in request: {e}")
        return f"ERROR: {str(e)}"

def process_single_score_request_perception(client: OpenAI, image_data: str, question: str, text: str, model: str = MODEL_NAME) -> str:
    """
    处理单个独立评分请求（感知模式）。
    每一句独立打分，不依赖前面的步骤。
    
    Args:
        client: OpenAI 客户端
        image_data: 已经是处理好的 Base64 字符串或 URL
        question: 问题字符串
        text: 当前要评分的推理步骤
        model: 模型名称
    """
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_score_independent.format(problem_text=question, step_text=text)},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in request: {e}")
        return f"ERROR: {str(e)}"

def generate_batch_score(
    image_input: Union[str, Image.Image],
    question: str,
    text_list: List[str],
    steps_before_list: Optional[List[str]] = None,
    max_workers: int = 5,
    model: str = MODEL_NAME
) -> List[str]:
    """
    批量对同一个图片和问题，针对不同的文本列表进行评分/生成。
    
    Args:
        image_input: 图片路径、PIL Image 对象、Base64 字符串或 URL
        question: 问题字符串
        text_list: 待处理的文本列表（例如不同的 CoT 步骤或候选答案）
        steps_before_list: 每个步骤对应的之前推理步骤列表，长度应与 text_list 相同。
                          如果为 None，则自动根据 text_list 的顺序生成（假设 text_list 是连续的步骤）
        max_workers:并发线程数
        model: 模型名称
        
    Returns:
        List[str]: 对应的响应列表，顺序与 text_list 一致
    """
    client = get_client()
    results = [None] * len(text_list)
    
    # 如果没有提供 steps_before_list，则自动生成（假设 text_list 是按顺序的步骤）
    if steps_before_list is None:
        steps_before_list = []
        for i in range(len(text_list)):
            if i == 0:
                steps_before_list.append("")
            else:
                # 将之前的所有步骤拼接起来
                previous_steps = "\n".join([f"Step {j+1}: {text_list[j]}" for j in range(i)])
                steps_before_list.append(previous_steps)
    
    # 预处理图片：如果是本地文件或对象，先转成 Base64
    # 这样只需要处理一次图片，而不是在每个线程里都处理
    try:
        final_image_data = encode_image_to_base64(image_input)
    except Exception as e:
        return [f"ERROR: Image processing failed - {str(e)}"] * len(text_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将任务提交给线程池
        future_to_index = {
            executor.submit(process_single_score_request, client, final_image_data, question, text, steps_before, model): i 
            for i, (text, steps_before) in enumerate(zip(text_list, steps_before_list))
        }
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Unexpected error in thread for item {index}: {e}")
                results[index] = f"ERROR: {str(e)}"
                
    return results

def generate_batch_score_perception(
    image_input: Union[str, Image.Image],
    question: str,
    text_list: List[str],
    max_workers: int = 5,
    model: str = MODEL_NAME
) -> List[str]:
    """
    批量对同一个图片和问题，针对不同的文本列表进行独立评分（感知模式）。
    每一句独立打分，不依赖前面的步骤。
    
    Args:
        image_input: 图片路径、PIL Image 对象、Base64 字符串或 URL
        question: 问题字符串
        text_list: 待处理的文本列表（例如不同的 CoT 步骤或候选答案）
        max_workers: 并发线程数
        model: 模型名称
        
    Returns:
        List[str]: 对应的响应列表，顺序与 text_list 一致
    """
    client = get_client()
    results = [None] * len(text_list)
    
    # 预处理图片：如果是本地文件或对象，先转成 Base64
    # 这样只需要处理一次图片，而不是在每个线程里都处理
    try:
        final_image_data = encode_image_to_base64(image_input)
    except Exception as e:
        return [f"ERROR: Image processing failed - {str(e)}"] * len(text_list)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将任务提交给线程池
        future_to_index = {
            executor.submit(process_single_score_request_perception, client, final_image_data, question, text, model): i 
            for i, text in enumerate(text_list)
        }
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Unexpected error in thread for item {index}: {e}")
                results[index] = f"ERROR: {str(e)}"
                
    return results

def compute_score(predict_str: str, ground_truth: str, extra_info=None, **kwargs) -> float:

    is_format_error = False
    sentence_list = []
    # 1. 快速检查标签数量：think 和 answer 都必须成对出现且仅出现一次
    if (predict_str.count("<think>") != 1 or predict_str.count("</think>") != 1 or 
        predict_str.count("<answer>") != 1 or predict_str.count("</answer>") != 1):
        is_format_error = True
        sentence_list = cut_sentences(predict_str)
    else:
        # 2. 严格结构检查：<think>...</think><answer>...</answer>
        pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
        if not re.match(pattern, predict_str, re.DOTALL):
            is_format_error = True
            sentence_list = cut_sentences(predict_str)
        else:
            sentence_list = cut_sentences(predict_str)

    answer_text = extract_answer(predict_str)
    if answer_text is None:
        answer_text = ""

    # skip the case that the answer is empty
    if answer_text == "":
        acc_reward = 0.0
    else:
        if is_format_error == True:
            acc_reward = 0.0
        else:
            question_text = extra_info["question"]
            full_prompt = get_prompt(answer_text, ground_truth, question_text)

            # client_idx = random.randint(0, len(client_list) - 1)
            client = client_list[0]
            model_name = "qwen3-vl-thinking"

            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                seed=42,
                temperature=0.3,
            )
            response = chat_response.choices[0].message.content.strip()
            if "Judgement:" in response:
                response = response.split("Judgement:")[-1].strip()
                if "1" in response:
                    acc_reward = 1.0
                elif "0" in response:
                    acc_reward = 0.0
                else:
                    print(f" [WARNING] resp format error {response=}")
                    acc_reward = 0.0
            else:
                if response == "1":
                    acc_reward = 1.0
                elif response == "0":
                    acc_reward = 0.0
                else:
                    print(f" [WARNING] resp format error {response=}")
                    acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1024:
        acc_reward = 0.0
        is_format_error = True

    format_reward = 0.0 if is_format_error else 1.0

    # 过程判断奖励

    lenth_sentence_list = len(sentence_list)

    process_reward = 0.0

    if acc_reward > 0.0:
        if lenth_sentence_list > 0:
            if extra_info["ability"] == "perception":
                score_list = generate_batch_score_perception(extra_info["image"], extra_info["question"], sentence_list, max_workers=8)
                # 感知模式：每一步独立打分，直接求和
                score_sum = 0.0
                for score_str in score_list:
                    if not score_str:
                        continue
                    match = re.search(r"<score>(.*?)</score>", str(score_str), re.DOTALL)
                    if match:
                        try:
                            val = float(match.group(1).strip())
                            score_sum += val
                        except (ValueError, TypeError):
                            pass
                process_reward = score_sum / lenth_sentence_list
                
            elif extra_info["ability"] == "math":
                score_list = generate_batch_score(extra_info["image"], extra_info["question"], sentence_list, max_workers=8)
                # 数学模式：如果某一步错了，后面的步骤都不得分
                score_sum = 0.0
                
                for score_str in score_list:
                    if not score_str:
                        # 没有返回结果，视为错误，推理链断裂，直接跳出
                        break
                    
                    match = re.search(r"<score>(.*?)</score>", str(score_str), re.DOTALL)
                    if match:
                        try:
                            val = float(match.group(1).strip())
                            if val <= 0:
                                # 当前步骤错误，推理链断裂，直接跳出
                                break
                            else:
                                score_sum += val
                                pass
                        except (ValueError, TypeError):
                            # 解析失败，视为错误，推理链断裂，直接跳出
                            break
                    else:
                        # 没有匹配到分数，视为错误，推理链断裂，直接跳出
                        break

                process_reward = score_sum / lenth_sentence_list
            elif extra_info["ability"] == "eval":
                process_reward = 0.0
    else:
        process_reward = 0.0

    sum_reward = (0.9*acc_reward + 0.1*format_reward + 0.45*process_reward)*10
    sum_reward = sum_reward / 14.5

    if extra_info["ability"] == "eval":
        sum_reward = 0.9*acc_reward + 0.1*format_reward
    
    return ( sum_reward, acc_reward, format_reward, process_reward)
