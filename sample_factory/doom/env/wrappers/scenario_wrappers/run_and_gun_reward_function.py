# run_and_gun_reward_function.py

import gymnasium as gym
import numpy as np
from vizdoom import GameVariable


class RunAndGunRewardFunction(gym.Wrapper):
    """
    Rewards:
      +1 for each enemy killed (GameVariable.KILLCOUNT)
      +0.001 * distance walked per step

    Success metric: Kills (GameVariable.KILLCOUNT)
    """

    def __init__(self, env, reward_kill=1.0, reward_scaler_traversal=0.001):
        super().__init__(env)
        self.reward_kill = reward_kill
        self.scaler = reward_scaler_traversal
        self.prev_kills = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.total_distance = 0.0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_kills = self.game.get_game_variable(GameVariable.KILLCOUNT)
        self.prev_x = self.game.get_game_variable(GameVariable.POSITION_X)
        self.prev_y = self.game.get_game_variable(GameVariable.POSITION_Y)
        self.total_distance = 0.0
        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)

        x = self.game.get_game_variable(GameVariable.POSITION_X)
        y = self.game.get_game_variable(GameVariable.POSITION_Y)
        dist = np.sqrt((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2)
        self.total_distance += dist
        self.prev_x, self.prev_y = x, y

        kills = self.game.get_game_variable(GameVariable.KILLCOUNT)
        reward = self.reward_kill * (kills - self.prev_kills) + dist * self.scaler
        self.prev_kills = kills

        info.setdefault("episode_extra_stats", {}).update({
            "kills": kills,
            "movement": self.total_distance,
        })
        info["true_objective"] = reward
        
        reward = reward * 3  # Scale reward to match other scenarios

        return obs, reward, terminated, truncated, info