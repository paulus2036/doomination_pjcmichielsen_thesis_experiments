import gymnasium as gym
from vizdoom import GameVariable


class FloorIsLavaRewardFunction(gym.Wrapper):
    """
    Dense reward shaping:
        +0.01 constant survival reward per frame
        +0.1 per frame while standing on a platform (CUMULATIVE reward, var[0])
        +1.0 for stepping onto a platform (Z increases, var[3])
        -0.1 per frame spent in lava (HEALTH decreases while var[0] is high)
        +1e-3 * distance moved (per step)

    Sparse version:
        +0.01 constant reward per frame survived
    """

    def __init__(
        self,
        env,
        reward_on_platform: float = 0.1,
        reward_platform_reached: float = 1.0,
        reward_frame_survived: float = 0.01,
        penalty_lava: float = -0.1,
        reward_scaler_traversal: float = 1e-3,
    ):
        super().__init__(env)
        self.reward_on_platform = reward_on_platform
        self.reward_platform_reached = reward_platform_reached
        self.reward_frame_survived = reward_frame_survived
        self.penalty_lava = penalty_lava
        self.scaler = reward_scaler_traversal

        self._prev_z = 0.0
        self._prev_health = 0.0
        self._prev_on_platform = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0

        self._distance_cum = 0.0
        self._frames_survived = 0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._prev_z = self._gamevar(GameVariable.POSITION_Z)
        self._prev_health = self._gamevar(GameVariable.HEALTH)
        self._prev_on_platform = self._gamevar(GameVariable.USER1)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._prev_y = self._gamevar(GameVariable.POSITION_Y)
        self._distance_cum = 0.0
        self._frames_survived = 0

        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)
        self._frames_survived += 1

        rew = 0.0

        # 1. Constant survival reward
        rew += self.reward_frame_survived

        # 2. Traversal reward
        x = info["POSITION_X"]
        y = info["POSITION_Y"]
        dist = ((x - self._prev_x) ** 2 + (y - self._prev_y) ** 2) ** 0.5
        self._distance_cum += dist
        rew += dist * self.scaler
        self._prev_x, self._prev_y = x, y

        # 3. Platform reached reward
        z = self._gamevar(GameVariable.POSITION_Z)
        if z > self._prev_z:
            rew += self.reward_platform_reached
        self._prev_z = z

        # 4. On-platform reward (cumulative variable)
        on_platform = self._gamevar(GameVariable.USER1)
        if on_platform > self._prev_on_platform:
            rew += (on_platform - self._prev_on_platform) * self.reward_on_platform
        self._prev_on_platform = on_platform

        # 5. Lava penalty (based on health decrease)
        health = self._gamevar(GameVariable.HEALTH)
        if health < self._prev_health:
            rew += self.penalty_lava
        self._prev_health = health

        info.setdefault("episode_extra_stats", {}).update({
            "movement": self._distance_cum,
        })

        info["true_objective"] = rew
        return obs, rew, terminated, truncated, info
