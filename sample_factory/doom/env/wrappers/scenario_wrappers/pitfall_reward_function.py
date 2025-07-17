import gymnasium as gym
from vizdoom import GameVariable, DEAD


class PitfallRewardFunction(gym.Wrapper):
    def __init__(
        self,
        env,
        reward_scaler_pitfall: float = 0.05,     # scaled down forward reward
        penalty_death: float = -5.0,             # stronger punishment for death
        reward_goal: float = 1.0,
        success_threshold: float = 150000,       # only used if USER1 is still tracked
        penalty_idle: float = -0.02,
        idle_step_threshold: int = 3,
    ):
        super().__init__(env)
        self.scaler = reward_scaler_pitfall
        self.penalty = penalty_death
        self.reward_goal = reward_goal
        self.success_threshold = success_threshold
        self.penalty_idle = penalty_idle
        self.idle_step_threshold = idle_step_threshold

        self._no_progress_steps = 0
        self._prev_x = 0.0
        self._total_forward = 0.0
        self._frames = 0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._total_forward = 0.0
        self._frames = 0
        self._no_progress_steps = 0
        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)
        self._frames += 1

        rew = 0.0
        x = self._gamevar(GameVariable.POSITION_X)
        delta_x = x - self._prev_x
        self._prev_x = x

        # reward aligned movement in +X
        if delta_x > 0:
            rew += delta_x * self.scaler
            self._total_forward += delta_x
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1
            if self._no_progress_steps >= self.idle_step_threshold:
                rew += self.penalty_idle

        # death penalty
        dead = self._gamevar(DEAD)
        if dead:
            rew += self.penalty
            self._total_forward = 0.0

        # goal reward (if USER1 is meaningful and measured progress)
        dist = self._gamevar(GameVariable.USER1)
        if dist >= self.success_threshold:
            rew += self.reward_goal

        info.setdefault("episode_extra_stats", {}).update({
            "forward_distance": self._total_forward,
            "movement": self._total_forward / max(1, self._frames),
        })
        info["true_objective"] = rew
        
        rew = rew * 0.33  # Scale reward to match other scenarios (try to keep max around 100)

        return obs, rew, terminated, truncated, info