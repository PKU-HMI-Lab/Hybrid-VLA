import os, sys, pathlib
import argparse
import tqdm
import shutil
from termcolor import cprint, colored

from lift3d.envs.rlbench_env import RLBenchEnv, RLBenchActionMode, RLBenchObservationConfig
from lift3d.helpers.gymnasium import VideoWrapper
from lift3d.helpers.common import Logger
from lift3d.helpers.graphics import EEpose

import numpy as np
import pickle

from models import load_vla
import torch
from PIL import Image

def recreate_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

def model_load(args):
    model = load_vla(
            args.model_path,
            load_for_training=False,
            future_action_window_size=int(args.model_action_steps),
            hf_token=args.hf_token,
            use_diff = args.use_diff,
            )
    # (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16
    model.to(f'cuda:{args.cuda}').eval()
    return model

def model_predict(args, model, image, prompt, cur_robot_state=None):
    
    if int(args.use_diff)==1 and int(args.use_ar)==0:
        predict_mode = 'diff'
    elif int(args.use_diff)==0 and int(args.use_ar)==1:
        predict_mode = 'ar'
    elif int(args.use_diff)==1 and int(args.use_ar)==1:
        predict_mode = 'diff+ar'

    actions_diff, actions_ar, _ = model.predict_action(
            front_image = image,
            instruction = prompt,
            unnorm_key='rlbench',
            cfg_scale = float(args.cfg_scale), 
            use_ddim = True,
            num_ddim_steps = 8,
            cur_robot_state = cur_robot_state,
            predict_mode = predict_mode
            )
    actions_ar = [actions_ar]
    return actions_ar, actions_diff

def cal_cos(a,b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_similarity = dot_product / (norm_a * norm_b + 1e-7)
    return cosine_similarity

def main(args):
    # Report the arguments
    Logger.log_info(f'Running {colored(__file__, "red")} with arguments:')
    Logger.log_info(f'task name: {args.task_name}')
    Logger.log_info(f'number of episodes: {args.num_episodes}')
    Logger.log_info(f'result directory: {args.result_dir}')
    Logger.log_info(f'replay data directory: {args.replay_data_dir}')
    Logger.log_info(f'exp name: {args.exp_name}')
    Logger.log_info(f'actions steps: {args.model_action_steps}')
    Logger.log_info(f'replay or predict: {args.replay_or_predict}')
    Logger.log_info(f'max steps: {args.max_steps}')
    Logger.log_info(f'cuda used: {args.cuda}')
    cprint('-' * os.get_terminal_size().columns, 'cyan')

    action_mode = RLBenchActionMode.eepose_then_gripper_action_mode(absolute=True)
    obs_config = RLBenchObservationConfig.single_view_config(camera_name='front', image_size=(224, 224))
    env = RLBenchEnv(
        task_name=args.task_name,
        action_mode=action_mode,
        obs_config=obs_config,
        point_cloud_camera_names=['front'],
        cinematic_record_enabled=True,
    )
    env = VideoWrapper(env)
    
    if args.replay_or_predict == 'predict':
        args.result_dir = os.path.join(args.result_dir, 'predict_results')
    elif args.replay_or_predict == 'replay':
        args.result_dir = os.path.join(args.result_dir, 'replay_results')
    
    if args.exp_name is None:
        args.exp_name = args.task_name

    video_dir = os.path.join(
        args.result_dir, args.task_name, args.exp_name, "videos"
    )
    recreate_directory(video_dir)
    
    success_num = 0

    # #----------- for model predict
    if args.replay_or_predict == 'predict':
        model = model_load(args)
        episode_length = args.max_steps

    for i in range(args.num_episodes):

        # #----------- for key frames replay
        if args.replay_or_predict == 'replay_key':
            dat = np.load(os.path.join(args.replay_data_dir, args.task_name, f'episode{i}.npy'),allow_pickle = True)
            prompt = dat[0]['language_instruction']
            episode_length = len(dat)

        # #----------- for all frames replay
        if args.replay_or_predict == 'replay_origin':
            file_path = f"{args.replay_data_dir}/{args.task_name}/variation0/episodes/episode{i}/low_dim_obs.pkl"
            with open(file_path, "rb") as f:
                demo = pickle.load(f)
            episode_length = len(demo)
        
        Logger.log_info(f'episode: {colored(i, "red")}, steps: {colored(episode_length, "red")}')
        obs_dict = env.reset()
        terminated = False
        rewards = 0
        success = False
        ar_sum = 0 
        ar_cnt = 0
        ar_time_sum = 0
        diff_time_sum =0

        #default_robo_state
        cur_robot_state = np.array([ 0.27849028, -0.00815899,  1.47193933, -3.14159094,  0.24234043,  3.14158629,  1.        ])
        for j in range(episode_length):
            
            # #--------- for key frames replay
            if args.replay_or_predict == 'replay_key':
                action = dat[j]['action']
                robo_state = dat[j]['state']
                action[:3] += robo_state[7:10]
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)

            # #----------- for all frames replay
            if args.replay_or_predict == 'replay_origin':
                action = demo[j].gripper_pose
                action = np.append(action, np.array(demo[j].gripper_open))
                print(j, "  :", action)

            # # #----------- for model predict
            if args.replay_or_predict == 'predict':
                image = obs_dict['image']
                image = Image.fromarray(image)
                robot_state = obs_dict['robot_state']
                prompt = env.text
                if args.use_ar==1 and  args.use_diff==0:
                    action = model_predict(args, model, image, prompt)[0]
                else:
                    if args.use_robot_state:
                        action_ar, action_diff, ar_action_conf, infer_time  = model_predict(args, model, image, prompt, cur_robot_state)
                    else:
                        action_ar, action_diff, ar_action_conf, infer_time  = model_predict(args, model, image, prompt)

                    print("action confidence : ", ar_action_conf)
                    ar_conf = np.array(ar_action_conf)
                    ar_sum_single = ar_conf[:6].sum()
                    print("sum of ar_confidence: ", ar_sum_single)
                    ar_sum = ar_sum + ar_sum_single 
                    
                    ar_time = infer_time[0]
                    diff_time = infer_time[1]
                    print("ar infer time is:  ", ar_time)
                    print("diff infer time is  ", diff_time)
                    ar_time_sum += ar_time
                    diff_time_sum += diff_time
                    
                    if ar_sum_single > float(args.threshold):
                        action = (action_ar[0] + action_diff[0]) / 2
                        print("diff_action was fixed by ar_action!")
                    else:
                        action = action_diff[0]

                action[:3] += robot_state[7:10]
                cur_robot_state = action
                gripper_open = action[-1]
                action = EEpose.pose_6DoF_to_7DoF(action[:-1])
                action = np.append(action, gripper_open)
                print(j, "  :", action)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            success = success or bool(reward)
            ar_cnt += 1

            if terminated or truncated or success:
                break
        
        if success:
            success_num += 1

        print("average ar_sum =  ", ar_sum / ar_cnt)
        print("average ar infer time =  ", ar_time_sum / ar_cnt)
        print("average diff infer time =  ", diff_time_sum / ar_cnt)

        image_dir = os.path.join(
            args.result_dir, args.task_name, args.exp_name, "images", f"episode{i}"
        )
        recreate_directory(image_dir)

        env.save_video(os.path.join(video_dir, f'episode{i}_video_steps.mp4'))
        env.save_images(image_dir, quiet=True)
        Logger.log_info(f'video saved to {video_dir}')
        Logger.log_info(f'episode{i}_{success}')
        Logger.print_seperator()
    
    Logger.log_ok(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')
    with open(os.path.join(args.result_dir, args.task_name, f'{args.exp_name}_success_rate.txt'), "w", encoding="utf-8") as file:
        file.write(f'Finished. {args.task_name} * {args.num_episodes}. Success rate {success_num/args.num_episodes*100}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-data-dir', type=str, default='./data/rlds')
    parser.add_argument('--task-name', type=str, default='close_box')
    parser.add_argument('--replay-or-predict', type=str, default='replay')
    parser.add_argument('--num-episodes', type=int, default=3)
    parser.add_argument('--model-action-steps', type=str, default='15')
    parser.add_argument('--result-dir', type=str, default='./result')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--ddim-steps', type=int, default=10)
    parser.add_argument('--cfg-scale', type=str, default='1.5')
    parser.add_argument('--cuda', type=str, default='7')
    parser.add_argument('--use-diff', type=int, default=0)
    parser.add_argument('--use-ar', type=int, default=0)
    parser.add_argument('--threshold', type=str, default='5.8')
    parser.add_argument('--hf-token', type=str, default='')
    parser.add_argument('--use_robot_state', type=int, default=0)
    main(parser.parse_args())