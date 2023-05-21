from itertools import chain
import random
from abc import abstractmethod, ABC
from typing import Final, Literal


import numpy as np
from numpy import ndarray, integer
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns

from collect_coin_gym.types import (
    Location,
    Move,
    ObjectId,
    ObjectSpawnQuadrants,
    PeerPolicy,
)


# The Element IDs
# BACKGROUND = 0
# BLUE_AGENT = 1
# RED_AGENT = 2
# RED_COIN = 3
# GREEN_COIN = 4
# BLUE_COIN = 5
# BORDER = 6

default_obj_id = ObjectId(0, 1, 2, 3, 4, 5)
default_obj_spawn_quads = ObjectSpawnQuadrants(3, 3, 2, 1, 4)


class CollectCoinEnv(gym.Env):
    """
    Gym environment for the collect coin game.
    """

    def __init__(
        self,
        num_coins: int,
        map_shape: tuple[int, int] = (10, 10),
        obj_id: ObjectId = default_obj_id,
        obj_spawn_quads: ObjectSpawnQuadrants = default_obj_spawn_quads,
        quad_spawn_prob: float = 0.7,
        coin_collect_reward: float = 1.0,
        step_penalty: float = 0.01,
        is_move_clipped: bool = True,
        agent_stays: bool = False,
        peer_policy: PeerPolicy = "random",
        num_max_steps: int = 300,
        observation_space: spaces.Dict | None = None,
        render_mode: Literal["human", "rgb_array"] = "human",
    ) -> None:
        """
        Initialize the environment.

        Parameters
        ----------
        num_coins : int
            The number of coins for each color.
        map_shape : tuple[int, int] = (10, 10)
            The shape of the map. The first element is the number of rows and the second element is the number of columns. Even values are preferred to split the map into four quadrants.
        obj_id : ObjectId = default_obj_id
            The object IDs. The default values are: BORDER = 0, BLUE_AGENT = 1, RED_AGENT = 2, RED_COIN = 3, GREEN_COIN = 4, BLUE_COIN = 5.
        obj_spawn_quads : ObjectSpawnQuadrants = default_obj_spawn_quads
            The quadrants where the objects are spawned. The default values are: BLUE_AGENT = 3, RED_AGENT = 3, RED_COIN = 2, GREEN_COIN = 1, BLUE_COIN = 4. 0: random, 1: upper-right, 2: upper-left, 3: lower-left, 4: lower-right.
        quad_spawn_prob : float = 0.7
            The probability of the object to be spawned in the quadrant specified by obj_spawn_quads.
        coin_collect_reward : float = 1.0
            The reward for collecting a coin.
        step_penalty : float = 0.01
            The penalty for each step.
        is_move_clipped : bool = True
            Whether the agent is clipped when it tries to move out of the map.
        agent_stays : bool = False
            Whether the agent can stay at the same location.
        peer_policy : PeerPolicy = "random"
            The policy of the peer agent. "random" is for random action. "stay" is for staying at the same location.
        num_max_steps : int = 300
            The maximum number of steps.
        observation_space : spaces.Dict | None = None
            The observation space. If None, the default observation space is used.
        render_mode : Literal["human", "rgb_array"] = "human"
            The render mode. "human" is for rendering the map in a GUI window. "rgb_array" is for rendering the map as a numpy array.

        """
        super().__init__()

        self._map_shape: Final[tuple[int, int]] = map_shape
        self._obj_id: Final[ObjectId] = obj_id
        self._obj_spawn_quads: Final[ObjectSpawnQuadrants] = obj_spawn_quads
        self._quad_spawn_prob: Final[float] = quad_spawn_prob
        self._coin_collect_reward: Final[float] = coin_collect_reward
        self._step_penalty: Final[float] = step_penalty
        self._is_move_clipped: Final[bool] = is_move_clipped
        self._agent_stays: Final[bool] = agent_stays
        self._num_max_steps: Final[int] = num_max_steps
        self._render_mode: Final[Literal["human", "rgb_array"]] = render_mode

        # 0: up, 1: right, 2: down, 3: left, 4: stay
        self._actions: Final[tuple[Move, ...]] = (
            (Move(1, 0), Move(0, 1), Move(-1, 0), Move(0, -1))
            if not self._agent_stays
            else (Move(1, 0), Move(0, 1), Move(-1, 0), Move(0, -1), Move(0, 0))
        )

        self.action_space = spaces.Discrete(len(self._actions))

        h, w = map_shape

        self._border: Final[list[Location]] = list(
            set(
                [Location(0, i) for i in range(0, w)]
                + [Location(h - 1, i) for i in range(0, w)]
                + [Location(i, 0) for i in range(0, h)]
                + [Location(i, w - 1) for i in range(0, h)]
            )
        )

        self.observation_space: Final = (
            spaces.Dict(
                {
                    "blue_agent": spaces.Box(
                        low=np.array([-1, -1]),
                        high=np.array(self._map_shape) - 1,
                        dtype=integer,
                    ),
                    "red_agent": spaces.Box(
                        low=np.array([-1, -1]),
                        high=np.array(self._map_shape) - 1,
                        dtype=integer,
                    ),
                    "border": spaces.Box(
                        low=np.array(
                            [[0, 0] for _ in range(len(self._border))]
                        ).flatten(),
                        high=np.array(
                            [self._map_shape for _ in range(len(self._border))]
                        ).flatten()
                        - 1,
                        dtype=integer,
                    ),
                    "red_coins": spaces.Box(
                        low=np.array([[0, 0] for _ in range(num_coins)]).flatten(),
                        high=np.array(
                            [self._map_shape for _ in range(num_coins)]
                        ).flatten()
                        - 1,
                        dtype=integer,
                    ),
                    "green_coins": spaces.Box(
                        low=np.array([[0, 0] for _ in range(num_coins)]).flatten(),
                        high=np.array(
                            [self._map_shape for _ in range(num_coins)]
                        ).flatten()
                        - 1,
                        dtype=integer,
                    ),
                    "blue_coins": spaces.Box(
                        low=np.array([[0, 0] for _ in range(num_coins)]).flatten(),
                        high=np.array(
                            [self._map_shape for _ in range(num_coins)]
                        ).flatten()
                        - 1,
                        dtype=integer,
                    ),
                }
            )
            if observation_space is None
            else observation_space
        )

    def _get_obs(self) -> ObsDict:
        observation: ObsDict = {
            "blue_agent": np.array(self._blue_agent_loc),
            "red_agent": np.array(self._red_agent_loc),
            "blue_flag": np.array(self._fixed_obj.blue_flag).flatten(),
            "red_flag": np.array(self._fixed_obj.red_flag).flatten(),
            "blue_background": np.array(self._fixed_obj.blue_background).flatten(),
            "red_background": np.array(self._fixed_obj.red_background).flatten(),
            "obstacle": np.array(self._fixed_obj.obstacle).flatten(),
            "is_red_agent_defeated": np.array(int(self._is_red_agent_defeated)),
        }
        return observation

    def _get_info(self) -> InfoDict:
        info = {}
        return info

    def reset(
        self,
        blue_agent_loc: Location | None = None,
        red_agent_loc: Location | None = None,
    ) -> tuple[ObsDict, InfoDict]:
        self._blue_agent_loc = (
            blue_agent_loc
            if blue_agent_loc is not None
            else random.choice(self._fixed_obj.blue_background)
        )
        self._red_agent_loc = (
            red_agent_loc
            if red_agent_loc is not None
            else random.choice(self._fixed_obj.red_background)
        )

        self.blue_traj = [self._blue_agent_loc]
        self.red_traj = [self._red_agent_loc]

        self._is_red_agent_defeated: bool = False

        self._step_count: int = 0
        self._episodic_reward: float = 0.0

        self._field = FieldObj(
            self._blue_agent_loc, self._red_agent_loc, *self._fixed_obj
        )

        observation: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list: list[ObsDict] = [observation]

        return observation, info

    def step(self, action: int) -> tuple[ObsDict, float, bool, bool, InfoDict]:
        if self._is_red_agent_defeated:
            pass
        else:
            self._red_agent_loc = self._enemy_act()

        self._blue_agent_loc = self._act(self._blue_agent_loc, action)

        self.blue_traj.append(self._blue_agent_loc)
        self.red_traj.append(self._red_agent_loc)

        reward, terminated, truncated = self._reward()

        self._step_count += 1
        self._episodic_reward += reward

        if self._num_max_steps <= self._step_count:
            truncated = True
        else:
            pass

        observation: ObsDict = self._get_obs()
        info: InfoDict = self._get_info()

        self.obs_list.append(observation)

        return observation, reward, terminated, truncated, info

    def _act(self, loc: Location, action: int) -> Location:
        direction = self._moves[action]
        new_loc = Location(loc.y + direction.y, loc.x + direction.x)
        match self._is_move_clipped:
            case True:
                num_row: int
                num_col: int
                num_row, num_col = self._field_map.shape
                new_loc = Location(
                    np.clip(new_loc.y, 0, num_row - 1),
                    np.clip(new_loc.x, 0, num_col - 1),
                )

                if new_loc in self._fixed_obj.obstacle:
                    pass
                else:
                    loc = new_loc
            case False:
                loc = new_loc
        return loc

    @abstractmethod
    def _enemy_act(self) -> Location:
        ...

    def _reward(self) -> tuple[float, bool, bool]:
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = self._step_count >= self._num_max_steps

        if self._blue_agent_loc == self._fixed_obj.red_flag:
            reward += 1.0
            terminated = True
        else:
            pass

        if self._red_agent_loc == self._fixed_obj.blue_flag:
            reward -= 1.0
            terminated = True
        else:
            pass

        if (
            distance_points(self._blue_agent_loc, self._red_agent_loc) <= 1
            and not self._is_red_agent_defeated
        ):
            blue_win: bool

            match self._blue_agent_loc in self._fixed_obj.blue_background:
                case True:
                    blue_win = np.random.choice(
                        [True, False], p=[self._randomness, 1.0 - self._randomness]
                    )
                case False:
                    blue_win = np.random.choice(
                        [False, True], p=[self._randomness, 1.0 - self._randomness]
                    )

            if blue_win:
                reward += self._battle_reward_alpha * self._capture_reward
                self._is_red_agent_defeated = True
            else:
                reward -= self._battle_reward_alpha * self._capture_reward
                terminated = True

        if self._obstacle_penalty_beta is not None:
            if self._blue_agent_loc in self._fixed_obj.obstacle:
                reward -= self._obstacle_penalty_beta * self._capture_reward
                terminated = True
            else:
                pass
        else:
            pass

        reward -= self._step_penalty_gamma * self._capture_reward

        return reward, terminated, truncated

    def get_agent_trajs(self) -> tuple[Path, Path]:
        return self.blue_traj, self.red_traj

    def render(
        self,
        figure_save_path: str | None = None,
        markersize: int = 24,
        is_gui_shown: bool = False,
    ) -> ndarray | None:
        fig, ax = self._render_fixed_objects(markersize)

        ax.plot(
            self._blue_agent_loc.x,
            self._blue_agent_loc.y,
            marker="o",
            color="royalblue",
            markersize=markersize,
        )
        ax.plot(
            self._red_agent_loc.x,
            self._red_agent_loc.y,
            marker="o",
            color="crimson" if not self._is_red_agent_defeated else "lightgrey",
            markersize=markersize,
        )

        if figure_save_path is not None:
            fig.savefig(figure_save_path, dpi=600)
            plt.close()
        else:
            pass

        if is_gui_shown:
            plt.show()
            plt.close()
        else:
            pass

        output: ndarray | None

        match self._render_mode:
            case "rgb_array":
                fig.canvas.draw()
                output = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            case "human":
                output = None
            case _:
                raise Exception("[tl_search] The render mode is not defined.")

        return output

    def render_animation(
        self,
        path: str,
        obs_list: list[ObsDict] | None = None,
        marker_size: int = 24,
        interval: int = 200,
        is_gui_shown: bool = False,
    ) -> None:
        fig, ax = self._render_fixed_objects()

        artists: list[list[Line2D]] = []

        obs_list = self.obs_list if obs_list is None else obs_list

        blue_path: list[Location] = []
        red_path: list[Location] = []
        red_agent_status_traj: list[bool] = []

        for obs in obs_list:
            blue_path.append(obs["blue_agent"].tolist())
            red_path.append(obs["red_agent"].tolist())
            red_agent_status_traj.append(bool(obs["is_red_agent_defeated"]))

        for blue, red, is_red_agent_defeated in zip(
            blue_path, red_path, red_agent_status_traj
        ):
            blue_artist = ax.plot(
                blue[1], blue[0], marker="o", color="royalblue", markersize=marker_size
            )
            red_artist = ax.plot(
                red[1],
                red[0],
                marker="o",
                color="crimson" if not is_red_agent_defeated else "lightgrey",
                markersize=marker_size,
            )
            artists.append(blue_artist + red_artist)

        anim = ArtistAnimation(fig, artists, interval=interval)
        anim.save(path)

        if is_gui_shown:
            plt.show()
        else:
            pass

        plt.clf()
        plt.close()

    def _render_fixed_objects(
        self, markersize: int = 30, is_gui_shown: bool = False
    ) -> tuple[Figure, Axes]:
        sns.reset_defaults()
        plt.rcParams["font.family"] = "Times New Roman"
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        h, w = self._field_map.shape

        blue_flag: Location
        red_flag: Location
        obstacle: list[Location]
        (
            blue_background,
            red_background,
            blue_flag,
            red_flag,
            obstacle,
            _,
            _,
            _,
        ) = self._fixed_obj
        ax.set_xlim(-0.5, w - 1)
        ax.set_ylim(-0.5, h - 1)
        ax.set_aspect(1)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0.5, w + 0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, h + 0.5, 1), minor=True)
        ax.set_xticks(np.arange(0, w, 1))
        ax.set_yticks(np.arange(0, h, 1))
        ax.grid(which="minor")
        ax.tick_params(which="minor", length=0)

        for obs in obstacle:
            obs_rec = Rectangle((obs.x - 0.5, obs.y - 0.5), 1, 1, color="black")
            ax.add_patch(obs_rec)

        for bb in blue_background:
            bb_rec = Rectangle((bb.x - 0.5, bb.y - 0.5), 1, 1, color="aliceblue")
            ax.add_patch(bb_rec)

        for rb in red_background:
            rf_rec = Rectangle((rb.x - 0.5, rb.y - 0.5), 1, 1, color="mistyrose")
            ax.add_patch(rf_rec)

        ax.plot(
            blue_flag.x,
            blue_flag.y,
            marker=">",
            color="mediumblue",
            markersize=markersize,
        )
        ax.plot(
            red_flag.x,
            red_flag.y,
            marker=">",
            color="firebrick",
            markersize=markersize,
        )

        if is_gui_shown:
            plt.show()
            plt.close()
        else:
            pass

        return fig, ax

    def seed(self, seed: int) -> None:
        self._seed = seed
