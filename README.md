<div align="center">

# HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  
[🌐**Project Page**](https://hybrid-vla.github.io/) | [✍️**Paper(Arxiv)**](https://hybrid-vla.github.io/) | [🎥**Demo**](https://hybrid-vla.github.io/)


Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, 
Mengzhen Liu, Chengkai Hou, Mengdi Zhao, KC alex Zhou, Pheng-Ann Heng, Shanghang Zhang

</div>


![](assets/teaser.png)
**🤖 HybridVLA innovatively integrates diffusion and autoregressive action prediction within a single LLM, fully leveraging the continuity and probabilistic nature of diffusion alongside the reasoning capabilities of autoregressive modeling.** It undergoes pretraining on large, diverse, cross-embodied real-world robotic datasets and is further fine-tuned on both simulation and self-collected real-world data. HybridVLA achieves remarkable performance across various tasks, demonstrating strong generalization to unseen manipulated objects, backgrounds, spatial positions, and lighting conditions.

## ✨ News ✨

- [2025/03/13] HybridVLA is now live on arXiv! The pre-trained checkpoint on a large-scale robotic dataset has also been released.🚀 

## ✨ Coming soon ✨

- The training config and code will be officially released as soon as possible!

## 📦 Installation

The code is built using Python 3.10, we also recommand to use Python above Python 3.10. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (It may run with lower versions, but we have not tested it).
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment as follows:

```bash
conda create --name hybridvla python=3.10
```

Next, clone our repo and install the required packages with the following commands:

```bash
git clone https://github.com/PKU-HMI-Lab/Hybrid-VLA
cd Hybrid-VLA
pip install -e .
```

If you need to use the traning code, please also install the [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```bash
# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
# =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 🧩 Framework

Our code is built based on [OpenVLA](https://github.com/openvla/openvla) and [CogACT](https://github.com/microsoft/CogACT) and is organized in the following framework:

- `conf`: config files for hydridvla training
- `scripts`: scripts for training and testing
- `training`: contains strategies for training
- `models`: contains hybridvla models, including backbone & diffusion & vlm & vla
- `util`: contains different kinds of tools funtion
- `vla`: from openvla's vla structure, including action tokenizer, etc.

## 📈Pretrain Parameter

We release our pretrained model's parameters as follows:

- [Robotic Large-Scale Pretrained Checkpoint](https://pan.baidu.com/s/134S9y8UwoNlyw3yUKozbRw?pwd=1spu)
- [Simulation-Finetuned Checkpoint](https://pan.baidu.com/s/1f5zpPKoAJDRIHFIH602Bqg?pwd=3ca1)

## 💡Getting Started

### Inference

Our model requires PIL image and text prompt as input, please refer to the code below for the minimal inference :

```python
# also see scripts/test_toy.py
from PIL import Image
from vla import load_vla
import torch
import numpy as np
import time

model = load_vla(
        '<absolute-path-to-ckpt>',
        load_for_training=False,
        future_action_window_size=0,
        use_diff=True, # choose weither to use diff
        action_dim=7,
        )

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16

model.to('cuda:0').eval()

example_image: Image.Image = Image.open('<path-to-Hybrid-VLA>/assets/000.png') 
example_prompt = "close the laptop"
example_cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ])
actions_diff, actions_ar, _ = model.predict_action(
            front_image=example_image,
            instruction=example_prompt,
            unnorm_key = 'rlbench',
            cfg_scale = 0.0, 
            use_ddim = True,
            num_ddim_steps = 4,
            action_dim = 7,
            cur_robot_state = example_cur_robot_state,
            predict_mode = 'diff+ar' # or 'ar' or 'diff'
            )
    
print(actions_diff)
```

## 🔍Test in RLBench

We evaluated our hybridvla in [RLBench](https://github.com/stepjam/RLBench), which based on the CoppeliaSim simulator, to build the testing environment quickly, please refer to [LIFT3D](https://github.com/PKU-HMI-Lab/LIFT3D)'s instructions to install and test in RLBench.

See the ``scripts/sim.py`` for more details.

**Please remember** to set the environment variable:

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

## 📜️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.