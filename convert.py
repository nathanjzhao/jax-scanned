import pickle
import jax
import flax
import tensorflow as tf
import tensorflowjs as tfjs
from train import ActorCritic
import tensorflow_probability as tfp
import numpy as np


class ActorCriticTF(tf.keras.Model):
    def __init__(self, action_dim, activation='tanh'):
        super().__init__()
        self.action_dim = action_dim
        self.activation = tf.nn.tanh if activation == 'tanh' else tf.nn.relu

        # Actor layers
        self.actor_dense1 = tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                                  bias_initializer='zeros')
        self.actor_dense2 = tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                                  bias_initializer='zeros')
        self.actor_mean = tf.keras.layers.Dense(action_dim, kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                                bias_initializer='zeros')
        self.log_std = tf.Variable(initial_value=tf.zeros(action_dim), trainable=True, name="log_std")

        # Critic layers
        self.critic_dense1 = tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                                   bias_initializer='zeros')
        self.critic_dense2 = tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),
                                                   bias_initializer='zeros')
        self.critic_out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(1.0),
                                                bias_initializer='zeros')
        
    def call(self, x):
        # Actor
        actor_mean = self.activation(self.actor_dense1(x))
        actor_mean = self.activation(self.actor_dense2(actor_mean))
        actor_mean = self.actor_mean(actor_mean)
        
        pi = tfp.distributions.MultivariateNormalDiag(loc=actor_mean, scale_diag=tf.exp(self.log_std))

        # Critic
        critic = self.activation(self.critic_dense1(x))
        critic = self.activation(self.critic_dense2(critic))
        critic = self.critic_out(critic)

        return pi, tf.squeeze(critic, axis=-1)
    
def load_model(filename) -> ActorCritic:
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
flax_params = load_model('/home/nathanzh/purejaxrl/models/more_timesteps_model.pkl')

# Convert Flax parameters to a flat dictionary
flax_params_flat = flax.traverse_util.flatten_dict(flax_params['params'])

# Create a TensorFlow model with the same architecture as your Flax model
tf_model = ActorCriticTF(action_dim=flax_params['params']['Dense_2']['kernel'].shape[1]) # action mean

# Build the model with a dummy input to initialize all layers
dummy_input = tf.random.normal((1, flax_params['params']['Dense_0']['kernel'].shape[0]))
_ = tf_model(dummy_input)

# Step 3: Create a mapping between Flax and TensorFlow layer names
layer_mapping = {
    'actor_dense1': 'Dense_0',
    'actor_dense2': 'Dense_1',
    'actor_mean': 'Dense_2',
    'log_std': 'log_std',
    'critic_dense1': 'Dense_3',
    'critic_dense2': 'Dense_4',
    'critic_out': 'Dense_5',
}

# Step 4: Set the weights of the TF model
for tf_layer_name, flax_layer_name in layer_mapping.items():
    tf_layer = getattr(tf_model, tf_layer_name)
    
    kernel_key = (flax_layer_name, 'kernel')
    bias_key = (flax_layer_name, 'bias')
    
    if kernel_key in flax_params_flat and bias_key in flax_params_flat:
        kernel = jax.numpy.array(flax_params_flat[kernel_key])
        bias = jax.numpy.array(flax_params_flat[bias_key])
        tf_layer.set_weights([kernel, bias])

# Set log_std separately as it's not a layer
log_std_key = ('log_std',)
if log_std_key in flax_params_flat:
    log_std = jax.numpy.array(flax_params_flat[log_std_key])
    tf_model.log_std.assign(log_std)

# Step 5: Save the model in TensorFlow.js format
tfjs.converters.save_keras_model(tf_model, 'tfjs_model')
print("Model saved in TensorFlow.js format")