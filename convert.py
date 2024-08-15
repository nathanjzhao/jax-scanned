"""For converting trained model into tfjs model"""

import pickle
import jax
import flax
import tensorflow as tf
import tensorflowjs as tfjs
from train import ActorCritic
import numpy as np

# IMPORTANT !
# SET TF_USE_LEGACY_KERAS=1 since tensorflowjs does not support Keras models built from Python with TensorFlow 2.0


def create_actor_critic_model(input_dim, action_dim, activation="tanh"):
    activation_fn = tf.nn.tanh if activation == "tanh" else tf.nn.relu

    # Actor model
    actor_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_layer"),
            tf.keras.layers.Dense(
                256,
                activation=activation_fn,
                kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                bias_initializer="zeros",
                name="actor_dense1",
            ),
            tf.keras.layers.Dense(
                256,
                activation=activation_fn,
                kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                bias_initializer="zeros",
                name="actor_dense2",
            ),
            tf.keras.layers.Dense(
                action_dim,
                kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                bias_initializer="zeros",
                name="actor_mean",
            ),
        ]
    )

    return actor_model


def load_model(filename) -> ActorCritic:
    with open(filename, "rb") as f:
        return pickle.load(f)


flax_params = load_model("/home/nathanzh/purejaxrl/models/more_timesteps_model.pkl")

# Convert Flax parameters to a flat dictionary
flax_params_flat = flax.traverse_util.flatten_dict(flax_params["params"])

# Get input and action dimensions from the Flax model
input_dim = flax_params["params"]["Dense_0"]["kernel"].shape[0]
action_dim = flax_params["params"]["Dense_2"]["kernel"].shape[1]

# Create a TensorFlow model with the same architecture as your Flax model
tf_model = create_actor_critic_model(input_dim, action_dim)

# Build the model
tf_model.build((None, input_dim))

# Step 3: Create a mapping between Flax and TensorFlow layer names
layer_mapping = {
    "actor_dense1": "Dense_0",
    "actor_dense2": "Dense_1",
    "actor_mean": "Dense_2",
}

# Step 4: Set the weights of the TF model
for layer in tf_model.layers[1:]:  # Skip the Input layer
    flax_layer_name = layer_mapping[layer.name]

    kernel_key = (flax_layer_name, "kernel")
    bias_key = (flax_layer_name, "bias")

    if kernel_key in flax_params_flat and bias_key in flax_params_flat:
        kernel = jax.numpy.array(flax_params_flat[kernel_key])
        bias = jax.numpy.array(flax_params_flat[bias_key])
        layer.set_weights([kernel, bias])

# Add log_std as a separate variable
# log_std_key = ('log_std',)
# if log_std_key in flax_params_flat:
#     log_std = jax.numpy.array(flax_params_flat[log_std_key])
#     tf_model.log_std = tf.Variable(initial_value=log_std, trainable=True, name="log_std")

# Step 5: Save the model in TensorFlow.js format
tfjs.converters.save_keras_model(tf_model, "tfjs_model")
print("Model saved in TensorFlow.js format")
