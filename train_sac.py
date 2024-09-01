import argparse
import importlib
import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.core import frozen_dict
from typing import Sequence, NamedTuple, Any, Callable
from flax import struct
from flax.training.train_state import TrainState
import distrax
from brax.envs import State

# jax.config.update("jax_debug_nans", True)


class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.tanh if self.activation == "tanh" else nn.relu
        
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.tanh if self.activation == "tanh" else nn.relu
        
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x

class SACActorCritic:
    def __init__(self, action_dim, observation_dim, activation="tanh"):
        self.actor = Actor(action_dim, activation)
        self.critic1 = Critic(activation)
        self.critic2 = Critic(activation)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def init(self, rng):
        actor_key, critic1_key, critic2_key = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((1, self.observation_dim))
        dummy_obs_action = jnp.zeros((1, self.observation_dim + self.action_dim))
        actor_params = self.actor.init(actor_key, dummy_obs)
        critic1_params = self.critic1.init(critic1_key, dummy_obs_action)
        critic2_params = self.critic2.init(critic2_key, dummy_obs_action)
        return {"actor": actor_params, "critic1": critic1_params, "critic2": critic2_params}

    def get_action_and_log_prob(self, params, obs, rng):
        pi = self.actor.apply(params["actor"], obs)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob

    def get_q_values(self, params, obs, action):
        obs_action = jnp.concatenate([obs, action], axis=-1)
        q1 = self.critic1.apply(params["critic1"], obs_action)
        q2 = self.critic2.apply(params["critic2"], obs_action)
        return q1, q2


class SACTrainState(TrainState):
    target_params: frozen_dict.FrozenDict
    alpha: jnp.ndarray
    target_entropy: jnp.ndarray
    alpha_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    alpha_opt_state: optax.OptState

    @classmethod
    def create(cls, *, apply_fn, params, tx, target_params, alpha, target_entropy, alpha_optimizer, alpha_opt_state, **kwargs):
        self = cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            target_params=target_params,
            alpha=jnp.array(alpha, dtype=jnp.float32),
            target_entropy=jnp.array(target_entropy, dtype=jnp.float32),
            alpha_optimizer=alpha_optimizer,
            alpha_opt_state=alpha_opt_state,
            **kwargs,
        )
        return self

def create_train_state(rng, env_observation_size, env_action_size, config):
    """Create initial `TrainState`."""

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    network = SACActorCritic(env_action_size, env_observation_size, activation=config["ACTIVATION"])
    params = network.init(rng)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.scale_by_adam(eps=1e-5),
            optax.scale_by_schedule(linear_schedule)
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.scale_by_adam(eps=1e-5),
            optax.scale(config["LR"])
        )

    alpha_optimizer = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["ALPHA_LR"])
    )
    alpha_opt_state = alpha_optimizer.init(jnp.array(config["ALPHA"], dtype=jnp.float32))
    
    return SACTrainState.create(
        apply_fn=lambda params, obs, rng: network.get_action_and_log_prob(params, obs, rng),
        params=params,
        tx=tx,
        target_params=params,
        alpha=jnp.array(config["ALPHA"], dtype=jnp.float32),
        target_entropy=jnp.array(-np.prod(env_action_size), dtype=jnp.float32),
        alpha_optimizer=alpha_optimizer,
        alpha_opt_state=alpha_opt_state,
    )

class Memory(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    # value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def save_model(params, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env_module = importlib.import_module(config["ENV_MODULE"])

    env = env_module.HumanoidEnv()

    def train(rng):
        # INIT NETWORK
        print(f"Observation size: {env.observation_size}")
        print(f"Action size: {env.action_size}")
        initial_state = env.reset(jax.random.PRNGKey(0))
        print(f"Actual observation shape: {initial_state.obs.shape}")
        network = SACActorCritic(env.action_size, env.observation_size, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)

        train_state = create_train_state(_rng, env.observation_size, env.action_size, config)

        # jit-ifying and vmap-ing functions
        @jax.jit
        def reset_fn(rng: jnp.ndarray) -> State:
            rngs = jax.random.split(rng, config["NUM_ENVS"])
            return jax.vmap(env.reset)(rngs)

        @jax.jit
        def step_fn(states: State, actions: jnp.ndarray, rng: jnp.ndarray) -> State:
            return jax.vmap(env.step)(states, actions, rng)

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        env_state = reset_fn(jnp.array(reset_rng))

        obs = env_state.obs

        @jax.jit
        def get_action_and_log_prob(params, obs, rng):
            return network.get_action_and_log_prob(params, obs, rng)

        @jax.jit
        def get_q_values(params, obs, action):
            return network.get_q_values(params, obs, action)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            """Update steps of the model --- environment memory colelction then network update"""

            # COLLECT MEMORY
            def _env_step(runner_state, unused):
                """Runs NUM_STEPS across all environments and collects memory"""
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                action, log_prob = get_action_and_log_prob(train_state.params, last_obs, _rng)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                env_state = step_fn(env_state, action, rng_step)

                next_obs = env_state.obs
                reward = env_state.reward
                done = env_state.done
                info = env_state.metrics

                # STORE MEMORY
                memory = Memory(done, action, reward, log_prob, last_obs, next_obs, info)
                runner_state = (train_state, env_state, next_obs, rng)

                return runner_state, memory

            runner_state, mem_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_state, env_state, last_obs, rng = runner_state

            def _update_epoch(update_state, unused):
                def _update_minibatch(update_state, unused):
                    def _loss_fn(params, target_params, mem_batch, rng, alpha):

                        # Convert integer inputs to float
                        mem_batch = jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.int32 else x, mem_batch)

                        # Get actions
                        rng, _rng = jax.random.split(rng)
                        dist = network.actor.apply(params["actor"], mem_batch.obs)
                        action = dist.sample(seed=_rng)
                        log_prob = dist.log_prob(action)

                        # Critic loss
                        q1_pred, q2_pred = get_q_values(params, mem_batch.obs, mem_batch.action)
                        rng, _rng = jax.random.split(rng)
                        next_dist = network.actor.apply(params["actor"], mem_batch.next_obs)
                        next_action = next_dist.sample(seed=_rng)
                        next_log_prob = next_dist.log_prob(next_action)
                        next_q1, next_q2 = get_q_values(target_params, mem_batch.next_obs, next_action)
                        next_q = jnp.minimum(next_q1, next_q2)
                        
                        # Reshape values
                        next_q = next_q.squeeze(-1)
                        done = mem_batch.done.reshape(next_q.shape)
                        reward = mem_batch.reward.reshape(next_q.shape)
                        next_log_prob = next_log_prob.reshape(next_q.shape)
                        q1_pred = q1_pred.squeeze(-1)
                        q2_pred = q2_pred.squeeze(-1)

                        eps = 1e-6
                        target_q = reward + config["GAMMA"] * (1 - done) * (next_q - alpha * next_log_prob)
                        target_q = jax.lax.stop_gradient(target_q)
                        critic_loss = ((q1_pred - target_q)**2 + (q2_pred - target_q)**2).mean()

                        # Actor loss
                        rng, _rng = jax.random.split(rng)
                        new_action, new_log_prob = get_action_and_log_prob(params, mem_batch.obs, _rng)
                        q1, q2 = get_q_values(params, mem_batch.obs, new_action)
                        min_q = jax.lax.stop_gradient(jnp.minimum(q1, q2).squeeze(-1))
                        actor_loss = (alpha * new_log_prob - min_q).mean() 

                        # Alpha loss
                        alpha_loss = -alpha * jax.lax.stop_gradient(new_log_prob + train_state.target_entropy).mean()

                        return actor_loss + critic_loss + alpha_loss, (actor_loss, critic_loss, alpha_loss, new_log_prob)

                    train_state, mem_batch, rng = update_state
                    rng, _rng = jax.random.split(rng)

                    # Convert integer inputs to float
                    mem_batch = jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.int32 else x, mem_batch)

                    (total_loss, (actor_loss, critic_loss, alpha_loss, log_prob)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params, train_state.target_params, mem_batch, _rng, train_state.alpha
                    )
                                        
                    # Update target networks with gated update
                    new_target_params = optax.incremental_update(train_state.params, train_state.target_params, config["TAU"])

                    # Alpha loss and optimization
                    alpha_loss = -train_state.alpha * (log_prob + train_state.target_entropy).mean()
                    alpha_grads = jax.grad(lambda a: -a * (log_prob + train_state.target_entropy).mean())(train_state.alpha)
                    alpha_updates, new_alpha_opt_state = train_state.alpha_optimizer.update(
                        alpha_grads, train_state.alpha_opt_state
                    )
                    new_alpha = optax.apply_updates(train_state.alpha, alpha_updates)

                    train_state = train_state.replace(
                        params=optax.apply_updates(train_state.params, grads),
                        target_params=new_target_params,
                        alpha=new_alpha,
                        alpha_opt_state=new_alpha_opt_state
                    )

                    return (train_state, mem_batch, rng), (actor_loss, critic_loss, alpha_loss)  

                train_state, mem_batch, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = mem_batch

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                # organize into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                update_state = (train_state, mem_batch, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_minibatch, update_state, minibatches
                )
                return update_state, loss_info

            update_state = (train_state, mem_batch, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = mem_batch.info
            rng = update_state[-1]

            # jax.debug.breakpoint()
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                    
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a model with specified environment name."
    )
    parser.add_argument(
        "--env_name", type=str, required=True, help="Name of the environment"
    )
    parser.add_argument(
        "--env_module",
        type=str,
        default="environment",
        help="Module of the environment",
    )
    args = parser.parse_args()

    config = {
        "ALPHA": 0.1,
        "TAU": 0.005,  # for soft update of target parameters
        "ALPHA_LR": 3e-4, 
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        # "TOTAL_TIMESTEPS": 2048 * 2000,
        "TOTAL_TIMESTEPS": 5e8,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_MODULE": args.env_module,
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    print("done training")

    save_model(out["runner_state"][0].params, f"models/{args.env_name}_model.pkl")
