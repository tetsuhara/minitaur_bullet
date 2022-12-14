import imageio
import matplotlib.pyplot as plt
import os
import reverb
# import tempfile
import numpy as np

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
# from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

# tempdir = tempfile.gettempdir()
tempdir = os.getcwd()

##########################################################
# env_name = "MinitaurBulletEnv-v0" # @param {type:"string"}
# env2_name = "MinitaurBulletEnv-v1" # @param {type:"string"}
env_name = "MinitaurBulletCarryEnv-v0" # @param {type:"string"}
env2_name = "MinitaurBulletCarryEnv-v1" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 1500000 # @param {type:"integer"}
# num_iterations = 30000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 100000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}
##########################################################

collect_env = suite_pybullet.load(env_name)
eval_env = suite_pybullet.load(env_name)
eval_env2 = suite_pybullet.load(env2_name)

use_gpu = False #@param {type:"boolean"}
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

observation_spec, action_spec, time_step_spec = (
    spec_utils.get_tensor_specs(collect_env))

with strategy.scope():
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
    
with strategy.scope():
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))
    
with strategy.scope():
    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

    tf_agent.initialize()
    
rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

dataset = reverb_replay.as_dataset(
    sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_collect_policy, use_tf_function=True)

random_policy = random_py_policy.RandomPyPolicy(
    collect_env.time_step_spec(), collect_env.action_spec())

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    reverb_replay.py_client,
    table_name,
    sequence_length=2,
    stride_length=1)

# add
rewards_dir = os.path.join(tempdir, 'rewards')
os.makedirs(rewards_dir, exist_ok=True)
forward_path = os.path.join(rewards_dir, 'forward_reward.txt')
energy_path = os.path.join(rewards_dir, 'energy_reward.txt')
roll_path = os.path.join(rewards_dir, 'roll_reward.txt')
pitch_path = os.path.join(rewards_dir, 'pitch_reward.txt')
#####

initial_collect_actor = actor.Actor(
    collect_env,
    random_policy,
    train_step,
    steps_per_run=initial_collect_steps,
    observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
    collect_env,
    collect_policy,
    train_step,
    steps_per_run=1,
    metrics=actor.collect_metrics(10),
    summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
    eval_env,
    eval_policy,
    train_step,
    episodes_per_run=num_eval_episodes,
    metrics=actor.eval_metrics(num_eval_episodes),
    summary_dir=os.path.join(tempdir, 'eval'),
)

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
    tempdir,
    train_step,
    tf_agent,
    experience_dataset_fn,
    triggers=learning_triggers,
    strategy=strategy)

def get_eval_metrics():
    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
        results[metric.name] = metric.result()
    return results

def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))

# add
def delete_line(fpath):
    with open(fpath, 'r') as f:
        text_lines = f.readlines()
    with open(fpath, 'w') as f:
        f.write("".join(text_lines[:-1]) + "\n")

def append_paragraph(fpath):
    with open(fpath, 'a') as f:
        f.write("\n")
#####

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]
# add
losses = []

for _ in range(num_iterations):
    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # Evaluating.
    step = agent_learner.train_step_numpy

    if eval_interval and step % eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        returns.append(metrics["AverageReturn"])
        # add
        append_paragraph(forward_path)
        append_paragraph(energy_path)
        append_paragraph(roll_path)
        append_paragraph(pitch_path)
    else:
        delete_line(forward_path)
        delete_line(energy_path)
        delete_line(roll_path)
        delete_line(pitch_path)
        #####

    if log_interval and step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
        # add
        losses.append(loss_info.loss.numpy().tolist())

rb_observer.close()
reverb_server.stop()

# add
saved_data_dir = os.path.join(tempdir, 'save_data')
os.makedirs(saved_data_dir, exist_ok=True)
saved_return_path = os.path.join(saved_data_dir, 'save_return.txt')
saved_loss_path = os.path.join(saved_data_dir, 'save_loss.txt')

with open(saved_return_path, 'w') as f:
    f.write("\n".join([str(r) for r in returns]))

with open(saved_loss_path, 'w') as f:
    f.write("\n".join([str(l) for l in losses]))
#####

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.savefig('ave_return.png')


num_episodes = 5
video_filename = 'sac_minitaur.mp4'
video2_filename = 'sac_minitaur2.mp4'
with imageio.get_writer(video_filename, fps=60) as video:
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        video.append_data(eval_env.render())
        while not time_step.is_last():
            action_step = eval_actor.policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            video.append_data(eval_env.render())
print('First video made!')

with imageio.get_writer(video2_filename, fps=60) as video2:
    for _ in range(num_episodes):
        time_step = eval_env2.reset()
        video2.append_data(eval_env2.render())
        while not time_step.is_last():
            action_step = eval_actor.policy.action(time_step)
            time_step = eval_env2.step(action_step.action)
            video2.append_data(eval_env2.render())
print('Second video made!')
