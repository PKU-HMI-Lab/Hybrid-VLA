ENVIRONMENT=/share/miniconda3/envs/4dvla_diff
source /share/miniconda3/bin/activate $ENVIRONMENT
cd /share/code/4D_VLA/CogACT

export HF_HOME=/share/huggingface

export PYTHONPATH=/share/code/4D_VLA/CogACT:/share/code/4D_VLA/CogACT/vla:$PYTHONPATH
export PYTHONPATH=/share/code/4D_VLA/CogACT/vlm:$PYTHONPATH

python /share/code/Hybrid-VLA/test.py
