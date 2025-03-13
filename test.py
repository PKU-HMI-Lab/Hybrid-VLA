from PIL import Image
from vla import load_vla
import torch
import numpy as np
import time
def model():
    model = load_vla(
            '/home/lzy/4D_VLA-hao-beta/exp/exp_10tasks_selected_keyframe_state_cogact_pretrain_freeze_vit_window0_diff+ar_boi_eoi_state_mlp_500/checkpoints/step-004512-epoch-300-loss=0.0099.pt',
            load_for_training=False,
            action_model_type='DiT-B',
            future_action_window_size=0,
            load_dit = False,
            use_diff=True,
            action_dim=7,
            )
    model.vlm = model.vlm.to(torch.bfloat16)
    model.to('cuda:7').eval()

    image: Image.Image = Image.open('/home/lzy/4D_VLA-hao-kvcache/CogACT/000.png') 
    prompt = "close the laptop"
#     prompt = "pick up the laptop"
    a = 0
    b = 0
    for i in range(100):
        actions, _,conf,t = model.predict_action_diff_ar(
                front_image=image,
                instruction=prompt,
                unnorm_key = 'rlbench',
                cfg_scale = 0.0, 
                use_ddim = True,
                num_ddim_steps = 4,
                action_dim = 7,
            #     cur_robot_state = np.array([ 2.45506510e-01, -2.41859436e-01,  1.47199607e+00, -3.14157344e+00,
            #                                 2.42206345e-01,  2.27554191e-05,  1.00000000e+00,  3.02049518e-01,
            #                                 6.80768266e-02,  1.47148311e+00, -3.14156252e+00,  2.41816718e-01,
            #                                 -3.05434601e+00,  1.00000000e+00]),
                cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ]),
                )
        if i >=20 and i <80:
            t1 = t[0]
            t2 = t[1]
            a+=t1
            b+=t2
    print(a/60, b/60)
    # print(actions, _)

model()