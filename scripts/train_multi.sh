# TASK=rtx_dataset_4
TASK=12tasks_selected_keyframe_mulview
FUTURE_ACTION_STEPS=0
# SETTING=our_pretrain_4_freeze_none_window${FUTURE_ACTION_STEPS}_ar+diff_boi_eoi_state_mlp_phi3b_224 #_again_with_our_ep2
SETTING=ours_pretrain_freeze_vit_window${FUTURE_ACTION_STEPS}_diff+ar_boi_eoi_state_mlp_ptx_new_0314
FREEZE_VISON=true
FREEZE_LLM=false
# DATA_MIX=rtx_dataset
DATA_MIX=rlbench
DATA_ROOT=/share/rlds_data
EXP_PATH=/share/code/test_exp
WANDB_PROJECT=
WANDB_ENTITY=
HF_TOKEN=

NODES=5
NUM_GPUS=8
BATCH_SIZE=32
EPOCH=600
LEARNING_RATE=2e-5
REPEATED=4

CLASS_DROPOUT_PROB=0.0
ACTION_TOKENIZER_EXIST=true
USE_DIFF=true
AR_DIFF_LOSS=true

ips=("10.0.1.4" "10.0.1.2" "10.0.1.20" "10.0.1.21" "10.0.1.22")
MASTER_ADDR="10.0.1.4"


for i in "${!ips[@]}"; do
    ip=${ips[$i]}
    ssh root@"${ip}" << EOF
    source /share/miniconda3/bin/activate /share/miniconda3/envs/4dvla_diff
    cd /share/code/Hybrid-VLA
    export HF_HOME=/share/huggingface
    export PYTHONPATH=/share/code/Hybrid-VLA/models/vlms:$PYTHONPATH
    export PYTHONPATH=/share/code/Hybrid-VLA:$PYTHONPATH
    mkdir -p ${EXP_PATH}/exp_${TASK}_${SETTING}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    torchrun --nnodes $NODES --nproc-per-node $NUM_GPUS --node_rank=$i --master_addr=${MASTER_ADDR} --master_port=29500 train.py \
      --vla.type prism-dinosiglip-224px+oxe+diffusion \
      --vla.data_mix $DATA_MIX \
      --vla.base_vlm prism-dinosiglip-224px+7b \
      --need_to_sub 0 \
      --vla.expected_world_size $(($NUM_GPUS * $NODES)) \
      --vla.global_batch_size $(($NUM_GPUS * $NODES * $BATCH_SIZE)) \
      --vla.per_device_batch_size $BATCH_SIZE \
      --vla.learning_rate $LEARNING_RATE \
      --vla.epochs $EPOCH \
      --vla.freeze_vision_backbone $FREEZE_VISON \
      --vla.freeze_llm_backbone $FREEZE_LLM \
      --data_root_dir $DATA_ROOT/$TASK \
      --run_root_dir $EXP_PATH \
      --run_id exp_${TASK}_${SETTING} \
      --image_aug false \
      --wandb_project  $WANDB_PROJECT \
      --wandb_entity $WANDB_ENTITY \
      --save_interval 100 \
      --action_dim 7 \
      --repeated_diffusion_steps $REPEATED \
      --action_tokenizer_exist $ACTION_TOKENIZER_EXIST \
      --future_action_window_size $FUTURE_ACTION_STEPS \
      --class_dropout_prob $CLASS_DROPOUT_PROB \
      --use_diff $USE_DIFF \
      --ar_diff_loss $AR_DIFF_LOSS \
      --is_resume False \
      --hf_token $HF_TOKEN \
      --pretrained_checkpoint "/share/code/4D_VLA/exp/exp_rtx_dataset_4_our_pretrain_4_freeze_none_window0_ar+diff_boi_eoi_state_mlp_again_with_our_ep2/checkpoints/step-026810-epoch-01-loss=0.7753.pt" \
      > ${EXP_PATH}/exp_${TASK}_${SETTING}/output_$ip.txt 2>&1 &
EOF
done