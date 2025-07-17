import gymnasium as gym
from vizdoom import GameVariable


class ParkourRewardFunction(gym.Wrapper):
    def __init__(
        self,
        env,
        x_start: float = 608.0,
        y_start: float = 608.0,
        reward_scaler_traversal: float = 1e-3,
        reward_scaler_location: float = 0.005, # to scale to other max rewards
    ):
        super().__init__(env)
        self.x_start = x_start
        self.y_start = y_start
        self.scaler = reward_scaler_traversal
        self.loc_scaler = reward_scaler_location

        self._prev_x = 0.0
        self._prev_y = 0.0
        self._distance_cum = 0.0
        self._current_height = 0.0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._prev_y = self._gamevar(GameVariable.POSITION_Y)
        self._current_height = self._gamevar(GameVariable.POSITION_Z)
        self._distance_cum = 0.0
        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)

        x = info["POSITION_X"]
        y = info["POSITION_Y"]
        z = info["POSITION_Z"]

        dx = x - self._prev_x
        dy = y - self._prev_y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        self._distance_cum += dist
        self._prev_x, self._prev_y = x, y

        # reward = movement + location shaping
        movement_reward = dist * self.scaler
        location_reward = (abs(x - self.x_start) + abs(y - self.y_start)) * self.loc_scaler
        rew = movement_reward + location_reward

        self._current_height = z

        info.setdefault("episode_extra_stats", {}).update({
            "height": self._current_height,
            "movement": self._distance_cum,
        })
        info["true_objective"] = rew

        return obs, rew, terminated, truncated, info
