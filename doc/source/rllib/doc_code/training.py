# flake8: noqa

# __preprocessing_observations_start__
try:
    import gymnasium as gym

    env = gym.make("ale_py:ALE/Pong-v5")
    obs, infos = env.reset()
except Exception:
    import gym

    env = gym.make("PongNoFrameskip-v4")
    obs = env.reset()

# RLlib uses preprocessors to implement transforms such as one-hot encoding
# and flattening of tuple and dict observations.
from ray.rllib.models.preprocessors import get_preprocessor

prep = get_preprocessor(env.observation_space)(env.observation_space)
# <ray.rllib.models.preprocessors.GenericPixelPreprocessor object at 0x7fc4d049de80>

# Observations should be preprocessed prior to feeding into a model
obs.shape
# (210, 160, 3)
prep.transform(obs).shape
# (84, 84, 3)
# __preprocessing_observations_end__

# __query_action_dist_start__
# Get a reference to the policy
import numpy as np
import torch

from ray.rllib.algorithms.dqn import DQNConfig

algo = (
    DQNConfig()
    .api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )
    .framework("torch")
    .environment("CartPole-v1")
    .env_runners(num_env_runners=0)
    .training(
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
        }
    )
).build()
# <ray.rllib.algorithms.ppo.PPO object at 0x7fd020186384>

policy = algo.get_policy()
# <ray.rllib.policy.eager_tf_policy.PPOTFPolicy_eager object at 0x7fd020165470>

# Run a forward pass to get model output logits. Note that complex observations
# must be preprocessed as in the above code block.
logits, _ = policy.model({"obs": torch.from_numpy(np.array([[0.1, 0.2, 0.3, 0.4]]))})
# (<tf.Tensor: id=1274, shape=(1, 2), dtype=float32, numpy=...>, [])

# Compute action distribution given logits
policy.dist_class
# <class_object 'ray.rllib.models.tf.tf_action_dist.Categorical'>
dist = policy.dist_class(logits, policy.model)
# <ray.rllib.models.tf.tf_action_dist.Categorical object at 0x7fd02301d710>

# Query the distribution for samples, sample logps
dist.sample()
# <tf.Tensor: id=661, shape=(1,), dtype=int64, numpy=..>
dist.logp(torch.tensor([1]))
# <tf.Tensor: id=1298, shape=(1,), dtype=float32, numpy=...>

# Get the estimated values for the most recent forward pass
policy.model.value_function()
# <tf.Tensor: id=670, shape=(1,), dtype=float32, numpy=...>

print(policy.model)
"""
Model: "model"
_____________________________________________________________________
Layer (type)               Output Shape  Param #  Connected to
=====================================================================
observations (InputLayer)  [(None, 4)]   0
_____________________________________________________________________
fc_1 (Dense)               (None, 256)   1280     observations[0][0]
_____________________________________________________________________
fc_value_1 (Dense)         (None, 256)   1280     observations[0][0]
_____________________________________________________________________
fc_2 (Dense)               (None, 256)   65792    fc_1[0][0]
_____________________________________________________________________
fc_value_2 (Dense)         (None, 256)   65792    fc_value_1[0][0]
_____________________________________________________________________
fc_out (Dense)             (None, 2)     514      fc_2[0][0]
_____________________________________________________________________
value_out (Dense)          (None, 1)     257      fc_value_2[0][0]
=====================================================================
Total params: 134,915
Trainable params: 134,915
Non-trainable params: 0
_____________________________________________________________________
"""
# __query_action_dist_end__


# __get_q_values_dqn_start__
# Get a reference to the model through the policy
import numpy as np
import torch

from ray.rllib.algorithms.dqn import DQNConfig

algo = (
    DQNConfig()
    .api_stack(
        enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
    )
    .framework("torch")
    .environment("CartPole-v1")
    .training(
        replay_buffer_config={
            "type": "MultiAgentPrioritizedReplayBuffer",
        }
    )
).build()
model = algo.get_policy().model
# <ray.rllib.models.catalog.FullyConnectedNetwork_as_DistributionalQModel ...>

# List of all model variables
list(model.parameters())

# Run a forward pass to get base model output. Note that complex observations
# must be preprocessed. An example of preprocessing is
# examples/offline_rl/saving_experiences.py
model_out = model({"obs": torch.from_numpy(np.array([[0.1, 0.2, 0.3, 0.4]]))})
# (<tf.Tensor: id=832, shape=(1, 256), dtype=float32, numpy=...)

# Access the base Keras models (all default models have a base)
print(model)
"""
Model: "model"
_______________________________________________________________________
Layer (type)                Output Shape    Param #  Connected to
=======================================================================
observations (InputLayer)   [(None, 4)]     0
_______________________________________________________________________
fc_1 (Dense)                (None, 256)     1280     observations[0][0]
_______________________________________________________________________
fc_out (Dense)              (None, 256)     65792    fc_1[0][0]
_______________________________________________________________________
value_out (Dense)           (None, 1)       257      fc_1[0][0]
=======================================================================
Total params: 67,329
Trainable params: 67,329
Non-trainable params: 0
______________________________________________________________________________
"""

# Access the Q value model (specific to DQN)
print(model.get_q_value_distributions(model_out[0])[0])
# tf.Tensor([[ 0.13023682 -0.36805138]], shape=(1, 2), dtype=float32)
# ^ exact numbers may differ due to randomness

print(model.advantage_module)

# Access the state value model (specific to DQN)
print(model.get_state_value(model_out[0]))
# tf.Tensor([[0.09381643]], shape=(1, 1), dtype=float32)
# ^ exact number may differ due to randomness

print(model.value_module)
# __get_q_values_dqn_end__
