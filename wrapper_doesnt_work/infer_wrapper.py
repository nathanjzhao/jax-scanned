"""Runs inference for the trained model."""

import argparse
import importlib
import pickle

import jax
import jax.numpy as jnp
import mediapy as media

from train_wrapper import ActorCritic

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
    parser.add_argument(
        "--env_module",
        type=str,
        required=True,
        help="Name of the environment module to import.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for the environment."
    )
    parser.add_argument(
        "--normalize_env",
        type=bool,
        default=True,
        help="Whether to normalize the environment.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=2048,
        help="Number of environments to run in parallel.",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Whether to run in debug mode."
    )
    args = parser.parse_args()

    env_module = importlib.import_module(args.env_module)

    env = env_module.HumanoidEnv()
    env = env_module.ClipAction(env)
    env = env_module.VecEnv(env)

    if args.normalize_env:
        env = env_module.NormalizeVecObservation(env)
        env = env_module.NormalizeVecReward(env, args.gamma)

    rng = jax.random.PRNGKey(0)

    model_path = f"models/{args.env_name}_model.pkl"
    video_path = f"videos/{args.env_name}_video.mp4"
    print("Loading model from", model_path)
    loaded_params = load_model(model_path)
    network = ActorCritic(env.action_size, activation=ACTIVATION)

    fps = int(1 / env.dt)
    max_frames = int(args.video_length * fps)
    episode_reward = 0
    total_reward = 0
    rng, *reset_rng = jax.random.split(rng, args.num_envs + 1)
    obs, state = env.reset(jnp.array(reset_rng))

    episodes = 0

    # for step in range(max_frames):
    #     first_state = jax.tree_map(lambda x: x[0], state.env_state.env_state.pipeline_state)
    #     rollout.append(first_state)

    #     rng, _rng = jax.random.split(rng)
    #     pi, _ = network.apply(loaded_params, obs)
    #     action = pi.sample(seed=_rng)

    #     rng, *step_rng = jax.random.split(rng, 2049)
    #     obs, state, reward, done, info = env.step(state, action, jnp.array(step_rng))

    #     total_reward += reward
    #     episode_reward += reward

    #     if done[0]:
    #         episodes += 1
    #         print("Episode", episodes, "reward:", episode_reward)
    #         episode_reward = 0

    def step_fn(carry, step):
        rng, state, obs, total_reward, episode_reward, episodes, rollout_flat = carry

        # Get first state
        first_state = jax.tree.map(
            lambda x: x[0], state.env_state.env_state.pipeline_state
        )
        first_state_flat = jax.tree_util.tree_flatten(first_state)[0]

        # Update rollout_flat at the current step
        rollout_flat = jax.tree.map(
            lambda rollout, state_item: rollout.at[step].set(state_item),
            rollout_flat,
            first_state_flat,
        )

        rng, _rng = jax.random.split(rng)
        pi, _ = network.apply(loaded_params, obs)
        action = pi.sample(seed=_rng)

        rng, *step_rng = jax.random.split(rng, args.num_envs + 1)
        obs, state, reward, done, info = env.step(state, action, jnp.array(step_rng))

        # Rollout on only first environment --> track reward and done of only first
        total_reward += reward[0]
        episode_reward += reward[0]
        episodes = jnp.where(done[0], episodes + 1, episodes)
        episode_reward = jnp.where(done[0], 0, episode_reward)

        if args.debug:

            def callback(step, rewards, episodes):
                print(
                    f"Step: {step}, Episodes: {episodes}, Reward: {episode_reward.mean()}"
                )

            jax.debug.callback(callback, step, reward, episodes)

        return (
            rng,
            state,
            obs,
            total_reward,
            episode_reward,
            episodes,
            rollout_flat,
        ), None

    # Initialize rollout with the correct structure --- this is so that it is in JAX-scannable structure
    initial_state = jax.tree.map(
        lambda x: x[0], state.env_state.env_state.pipeline_state
    )
    initial_state_flat, tree_def = jax.tree_util.tree_flatten(initial_state)

    rollout_flat = jax.tree.map(
        lambda x: jnp.empty((max_frames,) + x.shape, dtype=x.dtype), initial_state_flat
    )

    # Initial carry value
    init_carry = (rng, state, obs, total_reward, episode_reward, episodes, rollout_flat)

    # Run the scan
    final_carry, _ = jax.lax.scan(
        step_fn, init_carry, jnp.arange(max_frames), unroll=16
    )

    # Unpack the final values
    rng, state, obs, total_reward, episode_reward, episodes, rollout_flat = final_carry

    # Unflatten and unroll the rollout
    def unflatten_frame(frame_idx):
        frame = jax.tree.map(lambda x: x[frame_idx], rollout_flat)
        return jax.tree_util.tree_unflatten(tree_def, frame)

    unflattened_rollout = [unflatten_frame(i) for i in range(max_frames)]

    total_reward /= max_frames
    print(f"Average reward: {total_reward}")

    print(f"Rendering video with {len(unflattened_rollout)} frames at {fps} fps")
    images = jnp.array(
        env.render(
            unflattened_rollout[:: args.render_every],
            width=args.width,
            height=args.height,
        )
    )

    print("Video rendered")
    media.write_video(video_path, images, fps=fps)
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
