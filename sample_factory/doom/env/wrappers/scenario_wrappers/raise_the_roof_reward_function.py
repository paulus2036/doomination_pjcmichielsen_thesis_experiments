# raise_the_roof_reward_function.py
import gymnasium as gym
from vizdoom import GameVariable


class RaiseTheRoofRewardFunction(gym.Wrapper):
    """
    Rewards:
        +15  on switch press (USER2 increments)
        +0.01 per survived frame
        +0.001 * distance traveled (per step)

    SUCCESS = Survive longer
    """

    def __init__(self, env, reward_switch_pressed=15.0, reward_frame_survived=0.01, reward_scaler_traversal=0.001):
        super().__init__(env)
        self.r_switch = reward_switch_pressed
        self.r_survive = reward_frame_survived
        self.scaler = reward_scaler_traversal

        self._prev_x = 0.0
        self._prev_y = 0.0
        self._prev_user2 = 0
        self._distance_cum = 0.0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_user2 = self._gamevar(GameVariable.USER2)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._prev_y = self._gamevar(GameVariable.POSITION_Y)
        self._distance_cum = 0.0
        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)

        # 1. Frame survival reward
        rew = self.r_survive

        # 2. Movement reward
        x = info["POSITION_X"]
        y = info["POSITION_Y"]
        dist = ((x - self._prev_x) ** 2 + (y - self._prev_y) ** 2) ** 0.5
        rew += dist * self.scaler
        self._distance_cum += dist
        self._prev_x, self._prev_y = x, y

        # 3. Switch pressed
        u2 = self._gamevar(GameVariable.USER2)
        if u2 > self._prev_user2:
            rew += self.r_switch
        self._prev_user2 = u2

        info.setdefault("episode_extra_stats", {}).update({
            "switches_pressed": u2,
            "movement": self._distance_cum,
        })
        info["true_objective"] = rew
        
        rew = rew * 0.3  # Scale reward to match other scenarios

        return obs, rew, terminated, truncated, info
