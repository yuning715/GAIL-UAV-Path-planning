import numpy as np

from src.algorithms.apf import APFController
from src.envs.breakthrough_env import BreakthroughEnv


def test_rollout_record_uses_environment_path():
    env = BreakthroughEnv({"single_episode_max_steps": 8}, seed=0)
    state = env.reset(seed=0)
    controller = APFController()
    for _ in range(3):
        state, *_ = env.step(controller.act(env, state))
    rec = env.record()
    assert rec.total_path_length > 0
    assert rec.cumulative_time == env.step_count * env.dt
    assert rec.average_relative_distance >= rec.minimum_relative_distance

