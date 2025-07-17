import gymnasium as gym
from vizdoom import GameVariable


class HideAndSeekRewardFunction(gym.Wrapper):
    """
    Dense reward shaping:
        +5.0 when health increases (kit picked up)
        -5.0 when health decreases (hit by enemy)
        +1e-3 * distance moved per step
        +0.01 per frame survived

    Sparse reward:
        +0.01 per frame survived
    """

    def __init__(
        self,
        env,
        reward_health_kit: float = 5.0,
        penalty_health_loss: float = -5.0,
        reward_scaler_traversal: float = 1e-3,
        reward_frame_survived: float = 0.01,
    ):
        super().__init__(env)
        self.reward_health_kit = reward_health_kit
        self.penalty = penalty_health_loss
        self.scaler = reward_scaler_traversal
        self.reward_frame_survived = reward_frame_survived

        self._prev_health = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0

        self._distance_cum = 0.0
        self._frames_survived = 0
        self._hits_taken = 0
        self._kits_obtained = 0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._prev_health = self._gamevar(GameVariable.HEALTH)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._prev_y = self._gamevar(GameVariable.POSITION_Y)

        self._distance_cum = 0.0
        self._frames_survived = 0
        self._hits_taken = 0
        self._kits_obtained = 0

        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)
        self._frames_survived += 1
        rew = self.reward_frame_survived

        # Movement reward
        x = info["POSITION_X"]
        y = info["POSITION_Y"]
        dist = ((x - self._prev_x) ** 2 + (y - self._prev_y) ** 2) ** 0.5
        self._distance_cum += dist
        rew += dist * self.scaler
        self._prev_x, self._prev_y = x, y

        # Health change
        health = self._gamevar(GameVariable.HEALTH)
        if health > self._prev_health:
            self._kits_obtained += 1
            rew += self.reward_health_kit
        elif health < self._prev_health:
            self._hits_taken += 1
            rew += self.penalty
        self._prev_health = health

        info.setdefault("episode_extra_stats", {}).update({
            "kits_obtained": self._kits_obtained,
            "hits_taken": self._hits_taken,
            "movement": self._distance_cum,
        })

        info["true_objective"] = rew
        
        if rew > 0:
            rew = rew * 3  # Scale reward to match other scenarios
        return obs, rew, terminated, truncated, info
