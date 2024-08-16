"""Runs inference for the trained model."""

import argparse
import pickle
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import mediapy as media

from environment_wrapper import (
    ClipAction,
    HumanoidEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
)
from train import ActorCritic
import os

NORMALIZE_ENV = True
GAMMA = 0.99
ACTIVATION = "tanh"


def load_model(filename) -> ActorCritic:
    with open(filename, "rb") as f:
        return pickle.load(f)


def main() -> None:
    """Runs inference with pretrained models."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, default="Humanoid-v2", help="name of the environment"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="maximum steps per episode"
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
        default=20.0,
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
    env = ClipAction(env)
    env = VecEnv(env)

    if NORMALIZE_ENV:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, GAMMA)

    rng = jax.random.PRNGKey(0)

    model_path = f"models/{args.env_name}_model.pkl"
    video_path = f"videos/{args.env_name}_video.mp4"
    print("Loading model from", model_path)
    loaded_params = load_model(model_path)
    network = ActorCritic(env.action_size, activation=ACTIVATION)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    rollout: list[Any] = []
    episode_reward = 0
    total_reward = 0
    rng, *reset_rng = jax.random.split(rng)
    obs, state = env.reset(jnp.array(reset_rng))

    episodes = 0

    for step in range(max_frames):
        rollout.append(state.env_state.env_state.pipeline_state)

        rng, _rng = jax.random.split(rng)
        pi, _ = network.apply(loaded_params, obs)
        action = pi.sample(seed=_rng)

        rng, *step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(state, action, jnp.array(step_rng))

        total_reward += reward
        episode_reward += reward

        if done:
            episodes += 1
            print("Episode", episodes, "reward:", episode_reward)
            episode_reward = 0

        if len(rollout) >= max_frames:
            break

    # def scan_env(carry, t):
    #     rollout, total_reward, state, rng = carry

    #     obs = state.obs

    #     rng, _rng = jax.random.split(rng)
    #     pi, _ = network.apply(loaded_params, obs)
    #     action = pi.sample(seed=_rng)

    #     rng, _rng = jax.random.split(rng)
    #     state = step_fn(state, action, _rng)
    #     total_reward += state.reward

    #     rollout = rollout.at[t].set(state.obs)

    #     return (rollout, total_reward, state, rng), None

    # rollout = jnp.empty((max_frames, env.observation_size))
    # (rollout, total_reward, _, _ ), _ = jax.lax.scan(
    #     scan_env, (rollout, 0, state, rng), jnp.arange(max_frames)
    # )

    total_reward /= max_frames
    print(f"Average reward: {total_reward}")

    print(f"Rendering video with {len(rollout)} frames at {fps} fps")
    images = jnp.array(
        env.render(
            rollout[:: args.render_every],
            width=args.width,
            height=args.height,
        )
    )

    print("Video rendered")
    media.write_video(video_path, images, fps=fps)
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()