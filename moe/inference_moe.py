from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 直接加载保存的 MoE 模型
# model = AutoModelForCausalLM.from_pretrained(
#     "/training-data/sudongwang/model/Qwen3-0.6B-MoE-8x",
#     torch_dtype=torch.bfloat16,
# )
# tokenizer = AutoTokenizer.from_pretrained("/training-data/sudongwang/model/Qwen3-0.6B-MoE-8x")

# model.cuda()
# test_inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
# with torch.no_grad():
#     test_outputs = model(**test_inputs, output_router_logits=True)
# print(f"Test output logits shape: {test_outputs.logits.shape}")
# print(f"Test aux_loss: {test_outputs.aux_loss}")
# print("\n✅ Model saved and loaded successfully!")
model_name = "/training-data/sudongwang/model/Qwen3-0.6B-MoE-8x"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
# 推理时关闭 router_logits 输出，避免 generate 时 aux_loss 计算出错
model.config.output_router_logits = False

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,  # 先用小的测试
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)