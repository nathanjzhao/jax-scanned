"""Definition of base humanoids environment with reward system and termination conditions."""

import argparse
from functools import partial
import logging
import os
import shutil
import subprocess
import tempfile
from flax import struct
from pathlib import Path
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx.base import State as MjxState

RUNNING_LENGTH = 10
EPSILON = 1e-4


logger = logging.getLogger(__name__)

REWARD_CONFIG = {
    "termination_height": 0.01,
    "height_limits": {"min_z": 0, "max_z": 2.0},
    "is_healthy_reward": 5,
    "original_pos_reward": {
        "exp_coefficient": 2,
        "subtraction_factor": 0.2,
        "max_diff_norm": 0.5,
    },
    "ctrl_cost_coefficient": 0.1,
    "weights": {"ctrl_cost": 0.1, "original_pos_reward": 1, "is_healthy": 5},
}


def download_model_files(repo_url: str, repo_dir: str, local_path: str) -> None:
    """Downloads or updates model files (XML + meshes) from a GitHub repository.

    Args:
        repo_url: The URL of the GitHub repository.
        repo_dir: The directory within the repository containing the model files.
        local_path: The local path where files should be saved.

    Returns:
        None
    """
    target_path = Path(local_path) / repo_dir

    # Check if the target directory already exists
    if target_path.exists():
        logger.info(
            f"Model files are already present in {target_path}. Skipping download."
        )
        return

    # Create a temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Clone the repository into the temporary directory
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True)

        # Path to the repo_dir in the temporary directory
        temp_repo_dir_path = temp_path / repo_dir

        if temp_repo_dir_path.exists():
            # If the target directory does not exist, create it
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Move the repo_dir from the temporary directory to the target path
            if target_path.exists():
                # If the target path exists, remove it first (to avoid FileExistsError)
                shutil.rmtree(target_path)
            shutil.move(str(temp_repo_dir_path), str(target_path.parent))
            logger.info(f"Model files downloaded to {target_path}")
        else:
            logger.info(f"The directory {repo_dir} does not exist in the repository.")


class HumanoidEnv(PipelineEnv):
    """Defines the environment for controlling a humanoid robot.

    This environment uses Brax's `mjcf` module to load a MuJoCo model of a
    humanoid robot, which can then be controlled using the `PipelineEnv` API.

    Parameters:
        n_frames: The number of times to step the physics pipeline for each
            environment step. Setting this value to be greater than 1 means
            that the policy will run at a lower frequency than the physics
            simulation.
    """

    initial_qpos: jnp.ndarray
    _action_size: int
    reset_noise_scale: float = 0.0

    def __init__(self, n_frames: int = 1, backend: str = "mjx", gamma = 0.99) -> None:
        """Initializes system with initial joint positions, action size, the model, and update rate."""
        # GitHub repository URL
        repo_url = "https://github.com/nathanjzhao/mujoco-models.git"

        # Directory within the repository containing the model files
        repo_dir = "humanoid-original"

        # Local path where the files should be saved
        environments_path = os.path.join(os.path.dirname(__file__), "environments")

        # Download or update the model files
        download_model_files(repo_url, repo_dir, environments_path)

        # Now use the local path to load the model
        xml_path = os.path.join(environments_path, repo_dir, "humanoid.xml")
        mj_model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_path)
        # self.initial_qpos = jnp.array(mj_model.keyframe("default").qpos)

        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        self._action_size = mj_model.nu
        sys: base.System = mjcf.load_model(mj_model)

        self.gamma = gamma

        super().__init__(sys, n_frames=n_frames, backend=backend)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        # qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        # initialize mjx state
        state = self.pipeline_init(qpos, qvel)
        obs = self.get_obs(state, jnp.zeros(self._action_size))
        metrics = {
            "episode_returns": 0,
            "episode_lengths": 0,
            "returned_episode_returns": 0,
            "returned_episode_lengths": 0,
            "timestep": 0,
            "returned_episode": False,
            "return_val": 0,
            "return_avg": 0,
            "obs_avg": 0,
            "return_var": 1,
            "obs_var": 1,
            "return_norm": 0,
            "obs_norm": 0,
            "return_count": EPSILON,
            "obs_count": EPSILON
        }

        return State(state, obs, jnp.array(0.0), False, metrics)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state: State, action: jnp.ndarray, rng: jnp.ndarray) -> State:
        """Run one timestep of the environment's dynamics and returns observations with rewards."""
        state = env_state.pipeline_state
        state_step = self.pipeline_step(state, action)
        obs_state = self.get_obs(state, action)

        # get obs/reward/done of action + states
        reward = self.compute_reward(state, state_step, action)
        done = self.is_done(state_step)

        # reset env if done
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self.reset_noise_scale, self.reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        # qpos = self.initial_qpos + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        state_reset = self.pipeline_init(qpos, qvel)
        obs_reset = self.get_obs(state, jnp.zeros(self._action_size))

        # selectively replace state/obs with reset environment based on if done
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.lax.select(done, obs_reset, obs_state)

        ########### METRIC TRACKING ###########
        # jax.debug.breakpoint()
        # breakpoint()

        # Get previous metrics
        metrics = env_state.metrics

        # state_obs_avg = metrics["obs_avg"] * (1 - done)
        # state_obs_var = metrics["obs_var"] * (1 - done) + done * 1
        # state_obs_count = metrics["obs_count"]*(1 - done) + done * EPSILON
        # obs_delta = jnp.mean(obs) - state_obs_avg
        # tot_obs_count = state_obs_count + 1
        # new_obs_mean = state_obs_avg + obs_delta*tot_obs_count

        # obs_m_a = (state_obs_var)*(state_obs_count)
        # #m_b = 0 since var of 1 item is 0
        # obs_M2 = obs_m_a + jnp.square(obs_delta) * state_obs_count / (tot_obs_count + EPSILON)
        # new_obs_var = obs_M2 / tot_obs_count
        # new_obs_count = tot_obs_count

        # metrics["obs_avg"] = new_obs_mean * (1 - done)
        # metrics["obs_var"] = new_obs_var * (1 - done) + done * 1
        # metrics["obs_count"] = new_obs_count * (1 - done) + done * 1

        # metrics["obs_norm"] = (jnp.mean(obs) - metrics["obs_avg"]) * (1 - done) / (jnp.sqrt(metrics["obs_var"] + EPSILON))

        state_return_val = metrics["return_val"]
        state_return_avg = metrics["return_avg"]
        state_return_var = metrics["return_var"]
        state_return_count = metrics["return_count"]
        new_return_val = state_return_val * self.gamma * (1 - done) + reward
        return_delta = new_return_val - state_return_avg
        tot_return_count = state_return_count + 1
        new_return_mean = state_return_avg + return_delta / (tot_return_count)
        return_m_a = state_return_var * state_return_count
        return_M2 = return_m_a + jnp.square(return_delta) * state_return_count / (tot_return_count)
        new_return_var = return_M2 / (tot_return_count)
        new_return_count = tot_return_count

        metrics["return_avg"] = new_return_mean
        metrics["return_var"] = new_return_var 
        metrics["return_count"] = new_return_count
        metrics["return_val"] = new_return_val

        metrics["return_norm"] = (reward) / (jnp.sqrt(metrics["return_var"] + EPSILON))




        # Calculate new episode return and length
        new_episode_return = metrics["episode_returns"] + reward
        new_episode_length = metrics["episode_lengths"] + 1


        # Update metrics -- we only count episode
        metrics["episode_returns"] = new_episode_return * (1 - done)
        metrics["episode_lengths"] = new_episode_length * (1 - done)
        metrics["returned_episode_returns"] = (
            metrics["returned_episode_returns"] * (1 - done) + new_episode_return * done
        )
        metrics["returned_episode_lengths"] = (
            metrics["returned_episode_lengths"] * (1 - done) + new_episode_length * done
        )
        metrics["timestep"] = metrics["timestep"] + 1
        metrics["returned_episode"] = done

        return env_state.replace(
            pipeline_state=state, obs=obs, reward=metrics["return_norm"], done=done, metrics=metrics
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_reward(
        self, state: MjxState, next_state: MjxState, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the reward for standing and height."""
        min_z, max_z = (
            REWARD_CONFIG["height_limits"]["min_z"],
            REWARD_CONFIG["height_limits"]["max_z"],
        )

        exp_coef, subtraction_factor, max_diff_norm = (
            REWARD_CONFIG["original_pos_reward"]["exp_coefficient"],
            REWARD_CONFIG["original_pos_reward"]["subtraction_factor"],
            REWARD_CONFIG["original_pos_reward"]["max_diff_norm"],
        )

        is_healthy = jnp.where(state.q[2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(state.q[2] > max_z, 0.0, is_healthy)

        diff = self.sys.qpos0 - state.qpos
        original_pos_reward = jnp.exp(
            -exp_coef * jnp.linalg.norm(diff)
        ) - subtraction_factor * jnp.clip(jnp.linalg.norm(diff), 0, max_diff_norm)

        ctrl_cost = -jnp.sum(jnp.square(action))

        # xpos = state.subtree_com[1][0]
        # next_xpos = next_state.subtree_com[1][0]
        # velocity = (next_xpos - xpos) / self.dt

        total_reward = (
            REWARD_CONFIG["weights"]["ctrl_cost"] * ctrl_cost
            + REWARD_CONFIG["weights"]["original_pos_reward"] * original_pos_reward
            + REWARD_CONFIG["weights"]["is_healthy"] * is_healthy
        )
        # total_reward = 0.1 * ctrl_cost + 5 * is_healthy + 1.25 * velocity

        return total_reward 

    @partial(jax.jit, static_argnums=(0,))
    def is_done(self, state: MjxState) -> bool:
        """Check if the episode should terminate."""
        # Get the height of the robot's center of mass
        com_height = state.q[2]

        # Set a termination threshold
        termination_height = REWARD_CONFIG["termination_height"]

        return com_height < termination_height

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, data: MjxState, action: jnp.ndarray) -> jnp.ndarray:
        obs_components = [
            data.qpos[2:],
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ]
        
        return jnp.concatenate(obs_components)


################## WRAPPERS ##################


class EnvWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


# @struct.dataclass
# class LogEnvState:
#     env_state: State
#     episode_returns: float
#     episode_lengths: int
#     returned_episode_returns: float
#     returned_episode_lengths: int
#     timestep: int

# class LogWrapper(EnvWrapper):
#     """Log the episode returns and lengths."""

#     def __init__(self, env: environment.Environment):
#         super().__init__(env)

#     @partial(jax.jit, static_argnums=(0,))
#     def reset(
#         self, key: jnp.array
#     ) -> Tuple[jnp.array, State]:
#         obs, env_state = self._env.reset(key)
#         state = LogEnvState(env_state, 0, 0, 0, 0, 0)
#         return obs, state

#     @partial(jax.jit, static_argnums=(0,))
#     def step(
#         self,
#         key: chex.PRNGKey,
#         state: State,
#         action: Union[int, float],
#         params: Optional[environment.EnvParams] = None,
#     ) -> Tuple[chex.Array, State, float, bool, dict]:
#         obs, env_state, reward, done, info = self._env.step(
#             key, state.env_state, action, params
#         )
#         new_episode_return = state.episode_returns + reward
#         new_episode_length = state.episode_lengths + 1
#         state = LogEnvState(
#             env_state=env_state,
#             episode_returns=new_episode_return * (1 - done),
#             episode_lengths=new_episode_length * (1 - done),
#             returned_episode_returns=state.returned_episode_returns * (1 - done)
#             + new_episode_return * done,
#             returned_episode_lengths=state.returned_episode_lengths * (1 - done)
#             + new_episode_length * done,
#             timestep=state.timestep + 1,
#         )
#         info["returned_episode_returns"] = state.returned_episode_returns
#         info["returned_episode_lengths"] = state.returned_episode_lengths
#         info["timestep"] = state.timestep
#         info["returned_episode"] = done
#         return obs, state, reward, done, info


class ClipAction(EnvWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, state, action, key):
        action = jnp.clip(action, self.low, self.high)
        env_state = self._env.step(state, action, key)
        return State(
            pipeline_state=env_state.pipeline_state,
            obs=env_state.obs,
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
        )


class TransformObservation(EnvWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key):
        env_state = self._env.reset(key)
        return State(
            pipeline_state=env_state.pipeline_state,
            obs=self.transform_obs(env_state.obs),
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
        )

    def step(self, state, action, key):
        env_state = self._env.step(state, action, key)
        return State(
            pipeline_state=env_state.pipeline_state,
            obs=self.transform_obs(env_state.obs),
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
        )


class TransformReward(EnvWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, state, action, key):
        env_state = self._env.step(state, action, key)
        return State(
            pipeline_state=env_state.pipeline_state,
            obs=env_state.obs,
            reward=self.transform_reward(env_state.reward),
            done=env_state.done,
            metrics=env_state.metrics,
        )


class VecEnv(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0))
        self.step = jax.vmap(self._env.step, in_axes=(None, 0, 0))  # state


class NormalizeVecObsState(State):
    norm_mean: jnp.ndarray
    norm_var: jnp.ndarray
    norm_count: float


class NormalizeVecObservation(EnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key):
        state = self._env.reset(key)
        obs = state.obs
        return NormalizeVecObsState(
            **state.__dict__,
            norm_mean=jnp.zeros_like(obs),
            norm_var=jnp.ones_like(obs),
            norm_count=EPSILON,
        )

    def step(self, state, action, key):
        next_state = self._env.step(state, action, key)
        obs = next_state.obs

        # Compute new normalization statistics
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.norm_mean
        tot_count = state.norm_count + batch_count

        new_mean = state.norm_mean + delta * batch_count / tot_count
        m_a = state.norm_var * state.norm_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.norm_count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        # Create new state with updated normalization data
        new_state = NormalizeVecObsState(
            **next_state.__dict__,
            norm_mean=new_mean,
            norm_var=new_var,
            norm_count=new_count,
        )

        # Normalize the observation
        new_state.obs = (obs - new_state.norm_mean) / jnp.sqrt(
            new_state.norm_var + EPSILON
        )

        return new_state


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: MjxState

    def __getattr__(self, name):
        return getattr(self.env_state, name)


class NormalizeVecReward(EnvWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key):
        env_state = self._env.reset(key)
        obs = env_state.obs
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=env_state,
        )
        return State(
            obs=obs,
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
        )

    def step(self, state, action, key):
        env_state = self._env.step(state.pipeline_state, action, key)
        return_val = (
            state.return_val * self.gamma * (1 - env_state.done) + env_state.reward
        )

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = env_state.obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        new_state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        normalized_reward = env_state.reward / jnp.sqrt(new_state.var + EPSILON)
        return State(
            pipeline_state=new_state,
            obs=env_state.obs,
            reward=normalized_reward,
            done=env_state.done,
            metrics=env_state.metrics,
        )


################## TEST ENVIRONMENT RUN ##################


def run_environment_adhoc() -> None:
    """Runs the environment for a few steps with random actions, for debugging."""
    try:
        import mediapy as media
        from tqdm import tqdm
    except ImportError:
        raise ImportError("Please install `mediapy` and `tqdm` to run this script")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_path", type=str, default="actor_params.pkl", help="path to actor model"
    )
    parser.add_argument(
        "--critic_path",
        type=str,
        default="critic_params.pkl",
        help="path to critic model",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=200, help="maximum steps per episode"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="random_video.mp4",
        help="path to save video",
    )
    parser.add_argument(
        "--render_every",
        type=int,
        default=2,
        help="how many frames to skip between renders",
    )
    parser.add_argument(
        "--video_length",
        type=float,
        default=10.0,
        help="desired length of video in seconds",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="width of the video frame"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="height of the video frame"
    )
    args = parser.parse_args()

    env = HumanoidEnv()
    action_size = env.action_size

    rng = jax.random.PRNGKey(0)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    rollout: list[Any] = []

    for episode in range(args.num_episodes):
        rng, _ = jax.random.split(rng)
        state = reset_fn(rng)

        total_reward = 0

        for step in tqdm(
            range(args.max_steps), desc=f"Episode {episode + 1} Steps", leave=False
        ):
            if len(rollout) < args.video_length * fps:
                rollout.append(state.pipeline_state)

            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(
                action_rng, (action_size,), minval=-1.0, maxval=1.0
            )  # placeholder for an action

            rng, step_rng = jax.random.split(rng)
            state = step_fn(state, action, step_rng)
            total_reward += state.reward

            if state.done:
                break

        logging.info(f"Episode {episode + 1} total reward: {total_reward}")

        if len(rollout) >= max_frames:
            break

    # Trim rollout to desired length
    rollout = rollout[:max_frames]

    images = jnp.array(
        env.render(
            rollout[:: args.render_every],
            camera="side",
            width=args.width,
            height=args.height,
        )
    )
    logging.info(f"Rendering video with {len(images)} frames at {fps} fps")
    media.write_video(args.video_path, images, fps=fps)
    logging.info(f"Video saved to {args.video_path}")


if __name__ == "__main__":
    # python environment.py
    run_environment_adhoc()
