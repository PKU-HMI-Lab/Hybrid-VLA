import os
import argparse
from prettytable import PrettyTable, ALL
from pyrep import PyRep
from rlbench.backend.const import TTT_FILE
from rlbench.backend.task import TASKS_PATH
from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend.robot import Robot
from rlbench.observation_config import ObservationConfig
from rlbench.backend.scene import Scene
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper


def main(args):
    table = PrettyTable()
    table.field_names = ['Task File', 'Task Class', 'Variation Count', 'Descriptions']

    # init sim for task instantiation
    sim = PyRep()
    ttt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rlbench', TTT_FILE)
    sim.launch(ttt_file, headless=True)
    sim.start()

    # init robot for task instantiation
    robot = Robot(
        arm=Panda(),
        gripper=PandaGripper(),
    )

    # init scene for task loading
    obs = ObservationConfig()
    obs.set_all(False)
    scene = Scene(sim, robot, obs)

    for task_file in os.listdir(TASKS_PATH):
        if not task_file.endswith('.py') or task_file == '__init__.py':
            continue
        
        # instantiate task
        task_class = task_file_to_task_class(task_file)
        task = task_class(sim, robot)

        try:
            # load task
            scene.load(task)
            task.init_task()
            table.add_row([
                task_file, 
                task_class.__name__, 
                task.variation_count(), 
                '\n'.join(task.init_episode(0)),
            ])
        except Exception as e:
            print(f'Error in {task_file}: {e}')
        finally:
            scene.unload()

    table.border = True
    table.hrules = ALL

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w') as f:
        f.write(str(table))
    print('Task report saved to task_report.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, default=os.path.join('/home/cx/LIFT3D/third_party/RLBench/tools', 'task_report.txt'), help='Output file for task report.')
    main(parser.parse_args())
