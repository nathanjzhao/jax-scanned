import argparse
import torch
import tensorflow as tf
import tensorflowjs as tfjs

# IMPORTANT !
# SET TF_USE_LEGACY_KERAS=1 since tensorflowjs does not support Keras models built from Python with TensorFlow 2.0


def create_actor_critic_model(input_dim, action_dim):
    # Actor model
    actor_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_layer"),
            tf.keras.layers.Dense(512, activation="elu", name="actor_dense1"),
            tf.keras.layers.Dense(256, activation="elu", name="actor_dense2"),
            tf.keras.layers.Dense(128, activation="elu", name="actor_dense3"),
            tf.keras.layers.Dense(action_dim, name="actor_output"),
        ]
    )

    # Critic model
    critic_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_layer"),
            tf.keras.layers.Dense(512, activation="elu", name="critic_dense1"),
            tf.keras.layers.Dense(256, activation="elu", name="critic_dense2"),
            tf.keras.layers.Dense(128, activation="elu", name="critic_dense3"),
            tf.keras.layers.Dense(1, name="critic_output"),
        ]
    )

    return actor_model, critic_model


def load_pytorch_model(filename):
    return torch.load(filename, map_location=torch.device("cpu"))


# Argument parsing
parser = argparse.ArgumentParser(
    description="Convert PyTorch model to TensorFlow.js format"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the PyTorch model"
)
args = parser.parse_args()

# Load PyTorch model
pytorch_model = load_pytorch_model(args.model_path)["model_state_dict"]

# Get input and action dimensions from the PyTorch model
input_dim = pytorch_model["actor.0.weight"].shape[1]
action_dim = pytorch_model["actor.6.weight"].shape[0]

# Create TensorFlow models with the same architecture as your PyTorch model
tf_actor_model, tf_critic_model = create_actor_critic_model(input_dim, action_dim)

# Build the models
tf_actor_model.build((None, input_dim))
tf_critic_model.build((None, input_dim))


# Create a mapping between PyTorch and TensorFlow layer names
actor_layer_mapping = {
    "actor_dense1": "actor.0",
    "actor_dense2": "actor.2",
    "actor_dense3": "actor.4",
    "actor_output": "actor.6",
}

critic_layer_mapping = {
    "critic_dense1": "critic.0",
    "critic_dense2": "critic.2",
    "critic_dense3": "critic.4",
    "critic_output": "critic.6",
}

# Set the weights of the TF actor model
for layer in tf_actor_model.layers[1:]:  # Skip the Input layer
    pytorch_layer_name = actor_layer_mapping[layer.name]
    layer.set_weights(
        [
            pytorch_model[f"{pytorch_layer_name}.weight"].numpy().T,
            pytorch_model[f"{pytorch_layer_name}.bias"].numpy(),
        ]
    )

# Set the weights of the TF critic model
for layer in tf_critic_model.layers[1:]:  # Skip the Input layer
    pytorch_layer_name = critic_layer_mapping[layer.name]
    layer.set_weights(
        [
            pytorch_model[f"{pytorch_layer_name}.weight"].numpy().T,
            pytorch_model[f"{pytorch_layer_name}.bias"].numpy(),
        ]
    )

# Save the actor model in TensorFlow.js format
tfjs.converters.save_keras_model(tf_actor_model, "tfjs_pytorch_actor")
print("Actor model saved in TensorFlow.js format")

# Save the critic model in TensorFlow.js format
# tfjs.converters.save_keras_model(tf_critic_model, f"{args.output_path}_critic")
# print("Critic model saved in TensorFlow.js format")
