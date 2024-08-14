import pickle
from gymnax.visualize import Visualizer
from jax import numpy as jnp
import jax
from ppo_continuous_action import ActorCritic
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

config = {
    "LR": 3e-4,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "hopper",
    "ANNEAL_LR": False,
    "NORMALIZE_ENV": True,
    "DEBUG": True,
}


def load_model(filename) -> ActorCritic:
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
env = LogWrapper(env)
env = ClipAction(env)
if config["NORMALIZE_ENV"]:
    env = NormalizeVecObservation(env)
    env = NormalizeVecReward(env, config["GAMMA"])

network = ActorCritic(env.action_size, activation=config["ACTIVATION"])
model_path = f"models/purejax_ppo_continuous_model.pkl"
loaded_params = load_model(model_path)
rng = jax.random.PRNGKey(0)

state_seq, reward_seq = [], []
rng, reset_rng = jax.random.split(rng)
obs, env_state = env.reset(jnp.array(reset_rng))
while True:
    state_seq.append(env_state)
    rng, rng_act = jax.random.split(rng)
    rng, rng_step = jax.random.split(rng)
    pi, _ = network.apply(loaded_params, obs)
    action = pi.sample(seed=rng_act)
    next_obs, next_env_state, reward, done, info = env.step(
        rng_step, env_state, action
    )
    reward_seq.append(reward)
    if done:
        break
    else:
      obs = next_obs
      env_state = next_env_state

cum_rewards = jnp.cumsum(jnp.array(reward_seq))
vis = Visualizer(env, env_params, state_seq, cum_rewards)

vis.animate(f"videos/anim.gif")