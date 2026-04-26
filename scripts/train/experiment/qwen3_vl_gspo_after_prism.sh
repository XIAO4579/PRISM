set -o pipefail

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export NCCL_TIMEOUT=1800000
export HF_HOME=/path/to/.cache/huggingface
export MOE_MODULE_PATH="/path/to/PRISM/moe"
export PYTHONPATH="/path/to/PRISM/scripts/train:/path/to/PRISM/transformers-4.57.0/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

export WANDB_PROJECT='PRISM'
export WANDB_API_KEY='<YOUR_WANDB_API_KEY>'

BASE_DIR=${BASE_DIR:-/path/to/PRISM}
EXPERIMENT_NAME="qwen3_vl_gspo_after_prism"

# ---- GSPO hyperparameters (ref: recipe/gspo/test_gspo_3b_math.sh; paper: https://arxiv.org/pdf/2507.18071) ----
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
reward_manager=dapo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# GSPO recommends extremely small clip ratios (Sec. 5.1)
clip_ratio_low=0.0003
clip_ratio_high=0.0004

max_prompt_length=2048
max_response_length=8192

train_batch_size=32
ppo_mini_batch_size=32
n_resp_per_prompt=16

enable_overlong_buffer=False
overlong_buffer_len=4096
overlong_penalty_factor=1.0

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
entropy_checkpointing=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    data.prompt_key=prompt \
    data.train_files=/path/to/PRISM/checkpoints/filter_output/rl_training_data_filtered.parquet \
    data.val_files=/path/to/PRISM/datasets/geo3k/test.parquet \
    data.shuffle=True \
    data.truncation=right \
    data.filter_overlong_prompts=True \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=/path/to/PRISM/models/Qwen3-VL-4B-Instruct-SFT-MMR1-TechAI-Gemini-Distill-exploration2-method1-stage1-500-steps \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=PRISM \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.test_freq=6 \
    trainer.save_freq=120 \
    trainer.total_training_steps=1500 \
    trainer.default_local_dir=/path/to/PRISM/checkpoints/${EXPERIMENT_NAME}/step2 \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir=/path/to/PRISM/checkpoints/${EXPERIMENT_NAME}/step2/rollout \
    "$@" \
    2>&1 | tee /path/to/PRISM/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log
