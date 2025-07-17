# sample_factory/doom/env/wrappers/scenario_wrappers/arms_dealer_reward_function.py
import gymnasium as gym
from vizdoom import GameVariable


class ArmsDealerRewardFunction(gym.Wrapper):
    """
    Rewards:
        +15  on every weapon pick-up     (USER1 increments)
        +30  on every successful delivery (USER2 increments)
        +1e-3 * travelled distance (per step)
        −0.1 constant passivity penalty (per step)

    SUCCESS = USER2  (arms dealt)
    """

    def __init__(
            self,
            env,
            reward_scaler_traversal: float = 1e-3,
            reward_weapon: float = 15.0,
            reward_delivery: float = 30.0,
            penalty_passivity: float = -0.1,
    ):
        super().__init__(env)
        self.scaler = reward_scaler_traversal
        self.r_weapon = reward_weapon
        self.r_delivery = reward_delivery
        self.penalty = penalty_passivity

        self._prev_u1 = 0
        self._prev_u2 = 0
        self._prev_x = 0.0
        self._prev_y = 0.0

        self._distance_cum = 0.0  # for episode stats

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    # --------------------------------------------------------------------- #
    # standard Gym API
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._prev_u1 = self._gamevar(GameVariable.USER1)
        self._prev_u2 = self._gamevar(GameVariable.USER2)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._prev_y = self._gamevar(GameVariable.POSITION_Y)
        self._distance_cum = 0.0

        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)

        # ----------------------------------------------------------------- #
        # 1. constant penalty
        rew = self.penalty

        # ----------------------------------------------------------------- #
        # 2. movement reward
        x = info["POSITION_X"]
        y = info["POSITION_Y"]
        dist = ((x - self._prev_x) ** 2 + (y - self._prev_y) ** 2) ** 0.5
        self._distance_cum += dist
        rew += dist * self.scaler
        self._prev_x, self._prev_y = x, y

        # ----------------------------------------------------------------- #
        # 3. event rewards (weapon pick-up / delivery)
        u1 = self._gamevar(GameVariable.USER1)
        u2 = self._gamevar(GameVariable.USER2)
        if u1 > self._prev_u1:
            rew += self.r_weapon
        if u2 > self._prev_u2:
            rew += self.r_delivery
        self._prev_u1, self._prev_u2 = u1, u2

        # ----------------------------------------------------------------- #
        # 4. propagate stats
        info.setdefault("episode_extra_stats", {}).update(
            {
                "weapons_acquired": u1,
                "arms_dealt": u2,
                "movement": self._distance_cum,
            }
        )
        info["true_objective"] = rew  # optional, keeps it consistent with other wrappers
        
        rew = rew * 0.20 # scale down the reward to match other scenarios

        return obs, rew, terminated, truncated, info
