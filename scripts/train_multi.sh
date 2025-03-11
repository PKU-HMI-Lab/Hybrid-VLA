TASK=rtx_dataset_4
FUTURE_ACTION_STEPS=0
SETTING=our_pretrain_4_freeze_none_window${FUTURE_ACTION_STEPS}_ar+diff_boi_eoi_state_mlp_phi3b_224 #_again_with_our_ep2
FREEZE_VISON=false
FREEZE_LLM=false
LOAD_DIT=false
DATA_MIX=rtx_dataset
DATA_ROOT=/share/rlds_data
EXP_PATH=/share/code/4D_VLA/exp
WANDB_PROJECT=cogact
WANDB_ENTITY=1162737898-the-chinese-university-of-hong-kong
HF_TOKEN=

NODES=5
NUM_GPUS=8
BATCH_SIZE=24
EPOCH=5
LEARNING_RATE=2e-5
REPEATED=2

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
    cd /share/code/4D_VLA/CogACT
    export HF_HOME=/share/huggingface
    export PYTHONPATH=/share/code/4D_VLA/CogACT/vlm:$PYTHONPATH
    mkdir -p ${EXP_PATH}/exp_${TASK}_${SETTING}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    torchrun --nnodes $NODES --nproc-per-node $NUM_GPUS --node_rank=$i --master_addr=${MASTER_ADDR} --master_port=29500 scripts/train.py \
      --vla.type prism-dinosiglip-224px+oxe+diffusion \
      --vla.data_mix $DATA_MIX \
      --vla.base_vlm phi-2+3b \
      --need_to_sub 1 \
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
      --load_dit $LOAD_DIT \
      --class_dropout_prob $CLASS_DROPOUT_PROB \
      --use_diff $USE_DIFF \
      --ar_diff_loss $AR_DIFF_LOSS \
      --action_model_type DiT-B \
      --is_resume False \
      --hf_token $HF_TOKEN \
      > ${EXP_PATH}/exp_${TASK}_${SETTING}/output_$ip.txt 2>&1 &
EOF
done

# --pretrained_checkpoint '/share/huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363/checkpoints/CogACT-Base.pt' \

######  single machine
# #  --node_rank=0 --master_addr=${MASTER_ADDR} --master_port=29500
# ssh root@14.103.228.215 << EOF
# source /share/miniconda3/bin/activate /share/miniconda3/envs/cogact
# cd /share/code/CogACT
# export HF_HOME=/share/huggingface
# mkdir -p exp/exp_${TASK}_${SETTING}
# torchrun --nnodes $NODES --nproc-per-node $NUM_GPUS --node_rank=0 --master_addr=$MASTER_ADDR --master_port=29500 scripts/train.py \
#   --vla.type prism-dinosiglip-224px+oxe+diffusion \
#   --vla.data_mix $DATASET_NAME \
#   --vla.expected_world_size $(($NUM_GPUS * $NODES)) \
#   --vla.global_batch_size $(($NUM_GPUS * $NODES * $BATCH_SIZE)) \
#   --vla.per_device_batch_size $BATCH_SIZE \
#   --vla.learning_rate 2e-5 \
#   --vla.epochs 100 \
#   --vla.freeze_vision_backbone $FREEZE_VISON \
#   --vla.freeze_llm_backbone $FREEZE_LLM \
#   --data_root_dir /share/code/rlds_dataset_builder/dataset/$TASK \
#   --run_root_dir /share/code/CogACT/exp \
#   --run_id exp_${TASK}_${SETTING} \
#   --image_aug false \
#   --wandb_project cogact \
#   --wandb_entity 1162737898-the-chinese-university-of-hong-kong \
#   --save_interval 100 \
#   --action_dim 7 \
#   --repeated_diffusion_steps 8 \
#   --future_action_window_size $FUTURE_ACTION_STEPS \
#   --load_dit $LOAD_DIT \
#   --action_model_type DiT-B \
#   --is_resume False \
#   --pretrained_checkpoint '/share/huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363/checkpoints/CogACT-Base.pt' \
# EOF




# tasks=("put_rubbish_in_bin_sparse" "toilet_seat_down_sparse" "unplug_charger_sparse" "close_laptop_lid_sparse" "water_plants_sparse")
# for task in "${tasks[@]}"; do
#   echo "Running training for task: $task"
  
#   torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
#     --pretrained_checkpoint "/home/cx/chenhao/hub/models--CogACT--CogACT-Base/snapshots/ffc4db3bef7735ba7aa692d50b6454588a32b753/checkpoints/CogACT-Base.pt" \
#     --vla.type prism-dinosiglip-224px+oxe+diffusion \
#     --vla.data_mix rlbench \
#     --vla.expected_world_size 8 \
#     --vla.global_batch_size 256 \
#     --vla.per_device_batch_size 32 \
#     --vla.learning_rate 2e-5 \
#     --data_root_dir /home/cx/rlds_dataset_builder/dataset/${task} \
#     --run_root_dir /home/cx/4dvla/CogACT \
#     --run_id exp_${task}_freeze_vit_window15 \
#     --image_aug false \
#     --wandb_project cogact \
#     --wandb_entity 1162737898-the-chinese-university-of-hong-kong \
#     --save_interval 100 \
#     --repeated_diffusion_steps 8 \
#     --future_action_window_size 15 \
#     --action_model_type DiT-B \
#     --is_resume False
  
#   echo "Finished training for task: $task"
# done





#### close_box  close_laptop_lid   put_rubbish_in_bin  unplug_charger  water_plants  toilet_seat_down