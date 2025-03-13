<div align="center">

# HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  
[🌐**Project Page**](https://hybrid-vla.github.io/)

</div>

## ✨ News ✨
- [2025/3/13] The Hybrid-VLA code has been officially released! 🎉 Check it out now for detailed implementation and usage.

## 📦 Installation

The code is built using Python 3.10, and can be run under any environment with Python 3.8 and above. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:
```bash
conda create --name hybridvla python=3.10
```
Next, clone our repo and install the required packages:
```bash
git clone https://github.com/PKU-HMI-Lab/Hybrid-VLA.git
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

Our framework is built based on OpenVLA, with a structure similar to Prismatic-VLM.

- `conf`: dataset & models & vla config

- `models`: models including backbones & vlm & hybridvla

- `overwatch`: record log info

- `training`: training strategies & metrics

- `util`: kinds of util function

- `vla`: vla-datasets and action-tokenizer

## 💡Usage

### Using Hybrid-VLA pretrained model.

- **Hybrid-VLA Model Checkpoints 📥** 
You can either manually download the model weights (ViT-B-32.pt and lift3d_clip_base.pth) from [Hugging Face](https://huggingface.co/jiayueru/Lift3d/blob/main/README.md) and place them in `lift3d/models/lift3d/ckpt`, or they will be automatically downloaded for you. In case of automatic download, the weights will be cached in the `lift3d/models/lift3d/ckpt`.

- **RLDS Dataset**

```python
# see also scripts/test_toy.py
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
actions_diff, _, _ = model.predict_action(
            front_image=example_image,
            instruction=example_prompt,
            unnorm_key = 'rlbench',
            cfg_scale = 0.0, 
            use_ddim = True,
            num_ddim_steps = 4,
            action_dim = 7,
            cur_robot_state = example_cur_robot_state,
            predict_mode = 'diff+ar'
            )
    
print(actions_diff)
```

### Evaluation in LIFT3D simulator.

we use [**LIFT3D**](https://github.com/PKU-HMI-Lab/LIFT3D) as our simulator. Please follow the [**Installation**](https://github.com/PKU-HMI-Lab/LIFT3D#-installation) section to set your environment.

Then follow scripts/sim.py to test your model. 

## 📜️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.