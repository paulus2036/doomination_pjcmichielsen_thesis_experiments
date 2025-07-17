import gymnasium as gym
from vizdoom import GameVariable


class ChainsawRewardFunction(gym.Wrapper):
    """
    Rewards:
        +5.0 per enemy killed (GameVariable index 1)
        +1e-3 * distance moved (per step)
    Episode stats:
        - health: GameVariable index 0
        - kills: GameVariable index 1
        - movement: cumulated distance
        - hits_taken: decrements in health
    """

    def __init__(self, env, reward_kill=5.0, reward_scaler_traversal=1e-3):
        super().__init__(env)
        self.reward_kill = reward_kill
        self.scaler = reward_scaler_traversal

        self._prev_health = 0.0
        self._prev_kills = 0.0
        self._prev_x = 0.0
        self._prev_y = 0.0

        self._hits_taken = 0
        self._distance_cum = 0.0

    def _gamevar(self, gv: GameVariable):
        return float(self.game.get_game_variable(gv))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._prev_health = self._gamevar(GameVariable.HEALTH)
        self._prev_kills = self._gamevar(GameVariable.USER1)
        self._prev_x = self._gamevar(GameVariable.POSITION_X)
        self._prev_y = self._gamevar(GameVariable.POSITION_Y)

        self._hits_taken = 0
        self._distance_cum = 0.0

        return obs

    def step(self, action):
        obs, base_rew, terminated, truncated, info = self.env.step(action)

        rew = 0.0

        # movement reward
        x = info["POSITION_X"]
        y = info["POSITION_Y"]
        dist = ((x - self._prev_x)**2 + (y - self._prev_y)**2)**0.5
        self._distance_cum += dist
        rew += dist * self.scaler
        self._prev_x, self._prev_y = x, y

        # kill reward
        kills = self._gamevar(GameVariable.USER1)
        if kills > self._prev_kills:
            rew += (kills - self._prev_kills) * self.reward_kill
        self._prev_kills = kills

        # health change
        health = self._gamevar(GameVariable.HEALTH)
        if health < self._prev_health:
            self._hits_taken += 1
        self._prev_health = health

        # episode statistics
        info.setdefault("episode_extra_stats", {}).update({
            "health": health,
            "kills": kills,
            "movement": self._distance_cum,
            "hits_taken": self._hits_taken,
        })
        info["true_objective"] = rew
        
        rew = rew * 13  # Scale reward to match other scenarios

        return obs, rew, terminated, truncated, info