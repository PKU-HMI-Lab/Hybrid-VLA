source /share/miniconda3/bin/activate /share/miniconda3/envs/4dvla_diff
cd /share/code/Hybrid-VLA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/share/huggingface
export PYTHONPATH=/share/code/Hybrid-VLA/models/vlms:$PYTHONPATH
export PYTHONPATH=/share/code/Hybrid-VLA:$PYTHONPATH

FUTURE_ACTION_STEPS=0
SETTING=reconstruction_test
FREEZE_VISON=true
FREEZE_LLM=false
ACTION_TOKENIZER_EXIST=true
USE_DIFF=true
AR_DIFF_LOSS=true
REPEATED_DIFFUSION_STEPS=4
CLASS_DROPOUT_PROB=0.0

DATA_MIX=rlbench
TASK=12tasks_selected_keyframe_mulview
NUM_GPUS=8
NODES=1
BATCH_SIZE=32
EPOCHS=200
LEARNING_RATE=2e-5
ACTION_DIM=7

DATA_ROOT=/share/rlds_data
EXP_ROOT=/share/code/test_exp

HF_TOKEN=

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nnodes ${NODES} --nproc-per-node ${NUM_GPUS} train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${DATA_MIX} \
  --vla.base_vlm prism-dinosiglip-224px+7b \
  --need_to_sub 0 \
  --vla.expected_world_size $((${NUM_GPUS} * ${NODES})) \
  --vla.per_device_batch_size ${BATCH_SIZE} \
  --vla.global_batch_size $((${NUM_GPUS} * ${NODES} * ${BATCH_SIZE})) \
  --vla.learning_rate ${LEARNING_RATE} \
  --vla.epochs ${EPOCHS} \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir ${DATA_ROOT}/${TASK} \
  --run_root_dir ${EXP_ROOT} \
  --run_id exp_${TASK}_${SETTING} \
  --image_aug false \
  --wandb_project  ""\
  --wandb_entity ""\
  --save_interval 100 \
  --action_dim ${ACTION_DIM} \
  --repeated_diffusion_steps ${REPEATED_DIFFUSION_STEPS} \
  --action_tokenizer_exist ${ACTION_TOKENIZER_EXIST} \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --class_dropout_prob ${CLASS_DROPOUT_PROB} \
  --use_diff ${USE_DIFF} \
  --ar_diff_loss ${AR_DIFF_LOSS} \
  --is_resume False \
  --hf_token ${HF_TOKEN} \
  --pretrained_checkpoint "/share/code/4D_VLA/exp/exp_rtx_dataset_4_our_pretrain_4_freeze_none_window0_ar+diff_boi_eoi_state_mlp_again_with_our_ep2/checkpoints/step-026810-epoch-01-loss=0.7753.pt" \