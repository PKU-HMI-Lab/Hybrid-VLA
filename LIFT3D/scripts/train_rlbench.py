from termcolor import colored
import subprocess


DATA_ROOT = './data/rlbench'

AGENTS = [
    ('clip_bnmlp', 256),
    ('vc1_bnmlp', 256),
    ('r3m_bnmlp', 256),
    ('spa_bnmlp', 256),
    ('pointnet_bnmlp', 16),
    ('point_next_bnmlp', 16),
    ('pointnet_plus_plus_bnmlp', 16),
    ('lift3d_bnmlp', 16),
]

TASKS = [
    'close_box',
    'put_rubbish_in_bin',
    'close_laptop_lid',
    'water_plants',
    'unplug_charger',
    'toilet_seat_down',
]

CAMERAS = [
    'front',
]


def test_codebase():
    for agent, batch_size in AGENTS:
        for task in TASKS:
            for camera in CAMERAS:
                cmd = [
                    'python', '-m', 'lift3d.tools.train_policy',
                    '--config-name=train_rlbench',
                    f'agent={agent}',
                    f'task_name={task}',
                    f'camera_name={camera}', 
                    f"dataloader.batch_size={batch_size}",
                    f'dataset_dir={DATA_ROOT}/{task}.zarr',
                ]
                print(colored('[INFO]', 'blue'), ' '.join(cmd))
                subprocess.run(cmd)



if __name__ == '__main__':
    test_codebase()
