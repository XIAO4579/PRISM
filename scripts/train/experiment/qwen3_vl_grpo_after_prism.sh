set -o pipefail

export VLLM_USE_FLASHINFER_SAMPLER=0
export NCCL_TIMEOUT=1800000
export HF_HOME=/path/to/.cache/huggingface
export MOE_MODULE_PATH="/path/to/PRISM/moe"
export PYTHONPATH="/path/to/PRISM/scripts/train:/path/to/PRISM/transformers-4.57.0/src:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

export WANDB_PROJECT='PRISM'
export WANDB_API_KEY='<YOUR_WANDB_API_KEY>'

BASE_DIR=${BASE_DIR:-/path/to/PRISM}
EXPERIMENT_NAME="qwen3_vl_grpo_after_prism"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.prompt_key=prompt \
    data.train_files=/path/to/PRISM/checkpoints/filter_output/rl_training_data_filtered.parquet \
    data.val_files=/path/to/gad_project/datasets/geo3k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    actor_rollout_ref.model.path=/path/to/gad_project/models/Qwen3-VL-4B-Instruct-SFT-MMR1-TechAI-Gemini-Distill-exploration2-method1-stage1-500-steps \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=PRISM \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=120 \
    trainer.default_local_dir=/path/to/PRISM/checkpoints/${EXPERIMENT_NAME}/step2 \
    trainer.test_freq=6 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=1500 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    trainer.rollout_data_dir=/path/to/PRISM/checkpoints/${EXPERIMENT_NAME}/step2/rollout \
    2>&1 | tee /path/to/PRISM/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log
