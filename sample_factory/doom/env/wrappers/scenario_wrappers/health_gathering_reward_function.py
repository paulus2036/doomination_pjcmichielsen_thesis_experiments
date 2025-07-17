import gymnasium as gym
from vizdoom import GameVariable


class HealthGatheringRewardFunction(gym.Wrapper):
    """
    Rewards:
        +0.01 per frame survived
        +15.0 when health increases (health kit obtained)
        -0.01 per frame (constant acid damage)
    """

    def __init__(
        self,
        env,
        reward_health_kit: float = 15.0,
        reward_frame_survived: float = 0.01,
        penalty_health_loss: float = -0.01,
    ):
        super().__init__(env)
        self.reward_health_kit = reward_health_kit
        self.reward_frame_survived = reward_frame_survived
        self.penalty = penalty_health_loss

        self._prev_health = 0.0
        self._frames_survived = 0
        self._kits_obtained = 0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._prev_health = self._gamevar(GameVariable.HEALTH)
        self._frames_survived = 0
        self._kits_obtained = 0

        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)
        self._frames_survived += 1

        rew = 0.0

        # constant survival reward
        rew += self.reward_frame_survived

        # penalty for constant health decay
        rew += self.penalty

        # reward for collecting health kits
        health = self._gamevar(GameVariable.HEALTH)
        if health > self._prev_health:
            self._kits_obtained += 1
            rew += self.reward_health_kit
        self._prev_health = health

        info.setdefault("episode_extra_stats", {}).update({
            "kits_obtained": self._kits_obtained,
        })

        info["true_objective"] = rew
        
        rew = rew * 0.15  # Scale reward to match other scenarios
        
        return obs, rew, terminated, truncated, info
