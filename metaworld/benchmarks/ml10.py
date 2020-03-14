
from metaworld.benchmarks.base import Benchmark
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_ARGS_KWARGS, MEDIUM_MODE_CLS_DICT


class ML10(MultiClassMultiTaskEnv, Benchmark):

    def __init__(self, env_type='train', sample_all=False, task_name=None):
        assert env_type == 'train' or env_type == 'test'
        if task_name is not None:
            assert task_name in MEDIUM_MODE_CLS_DICT[env_type]
            cls_dict = {task_name: MEDIUM_MODE_CLS_DICT[env_type][task_name]}
            args_kwargs = {task_name: MEDIUM_MODE_ARGS_KWARGS[env_type][task_name]}
        else:
            cls_dict = MEDIUM_MODE_CLS_DICT[env_type]
            args_kwargs = MEDIUM_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)
