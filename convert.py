import argparse
import pickle
import jax
import flax
import tensorflow as tf
import tensorflowjs as tfjs
from train import ActorCritic
import numpy as np

class LogStdLayer(tf.keras.layers.Layer):
    def __init__(self, action_dim, **kwargs):
        super(LogStdLayer, self).__init__(**kwargs)
        self.action_dim = action_dim

    def build(self, input_shape):
        self.log_std = self.add_weight(
            'log_std',
            shape=(self.action_dim,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.broadcast_to(self.log_std, tf.shape(inputs)[:-1] + tf.shape(self.log_std))

    def get_config(self):
        config = super(LogStdLayer, self).get_config()
        config.update({"action_dim": self.action_dim})
        return config

def create_actor_critic_model(input_dim, action_dim, activation="tanh"):
    activation_fn = tf.nn.tanh if activation == "tanh" else tf.nn.relu

    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Actor network
    x = tf.keras.layers.Dense(
        256, activation=activation_fn,
        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
        bias_initializer="zeros", name="actor_dense1"
    )(inputs)
    x = tf.keras.layers.Dense(
        128, activation=activation_fn,
        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
        bias_initializer="zeros", name="actor_dense2"
    )(x)
    actor_mean = tf.keras.layers.Dense(
        action_dim,
        kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
        bias_initializer="zeros", name="actor_mean"
    )(x)
    
    log_std = LogStdLayer(action_dim, name="log_std")(actor_mean)
    
    # Critic network
    y = tf.keras.layers.Dense(
        256, activation=activation_fn,
        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
        bias_initializer="zeros", name="critic_dense1"
    )(inputs)
    y = tf.keras.layers.Dense(
        256, activation=activation_fn,
        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
        bias_initializer="zeros", name="critic_dense2"
    )(y)
    critic = tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
        bias_initializer="zeros", name="critic_output"
    )(y)

    model = tf.keras.Model(inputs=inputs, outputs=[actor_mean, log_std, critic])
    return model

def load_model(filename) -> ActorCritic:
    with open(filename, "rb") as f:
        return pickle.load(f)

# Argument parsing
parser = argparse.ArgumentParser(description="Convert Flax model to TensorFlow.js format")
parser.add_argument("--env_name", type=str, required=True, help="Name of the environment")
args = parser.parse_args()

flax_params = load_model(f"models/{args.env_name}_model.pkl")

# Convert Flax parameters to a flat dictionary
flax_params_flat = flax.traverse_util.flatten_dict(flax_params["params"])

# Get input and action dimensions from the Flax model
input_dim = flax_params["params"]["Dense_0"]["kernel"].shape[0]
action_dim = flax_params["params"]["Dense_2"]["kernel"].shape[1]

# Create a TensorFlow model
tf_model = create_actor_critic_model(input_dim, action_dim)

# Build the model
tf_model.build((None, input_dim))

# Create a mapping between Flax and TensorFlow layer names
layer_mapping = {
    "actor_dense1": "Dense_0",
    "actor_dense2": "Dense_1",
    "actor_mean": "Dense_2",
    "critic_dense1": "Dense_3",
    "critic_dense2": "Dense_4",
    "critic_output": "Dense_5",
}

# Set the weights of the TF model
for layer in tf_model.layers[1:]:  # Skip the Input layer
    if layer.name in layer_mapping:
        flax_layer_name = layer_mapping[layer.name]
        kernel_key = (flax_layer_name, "kernel")
        bias_key = (flax_layer_name, "bias")
        if kernel_key in flax_params_flat and bias_key in flax_params_flat:
            kernel = jax.numpy.array(flax_params_flat[kernel_key])
            bias = jax.numpy.array(flax_params_flat[bias_key])
            layer.set_weights([kernel, bias])
    elif layer.name == 'log_std':
        log_std_key = ('log_std',)
        if log_std_key in flax_params_flat:
            log_std = jax.numpy.array(flax_params_flat[log_std_key])
            layer.set_weights([log_std])

# Save the model in TensorFlow.js format
tfjs.converters.save_keras_model(tf_model, f"tfjs_models/{args.env_name}")
print("Model saved in TensorFlow.js format") 