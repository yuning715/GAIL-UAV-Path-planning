import numpy as np

from src.envs.breakthrough_env import BreakthroughEnv
from src.envs.interception_env import InterceptionEnv


def test_breakthrough_step_shape():
    env = BreakthroughEnv({"single_episode_max_steps": 5}, seed=0)
    obs = env.reset(seed=0)
    next_obs, reward, done, info = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (6,)
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert "distance_to_loitering" in info
    assert isinstance(done, bool)


def test_interception_step_shape():
    env = InterceptionEnv({"group_interval_minutes": 0.05, "max_group_size": 2}, seed=0)
    obs = env.reset(seed=0, group_size=2)
    next_obs, reward, done, info = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (6,)
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert "intercepted" in info
    assert isinstance(done, bool)

