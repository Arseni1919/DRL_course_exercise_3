import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import box

from buffer import ReplayBuffer
from model import Network

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
plt.ion()

# look for a gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Parameters
network_params = {
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'hidden_dim': 64
}

training_params = {
    'batch_size': 256,
    'gamma': 0.95,
    'epsilon_start': 1.1,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.95,
    'target_update': 'soft',  # use 'soft' or 'hard'
    'tau': 0.01,  # relevant for soft update
    'target_update_period': 15,  # relevant for hard update
    'grad_clip': 0.1,
}

network_params = box.Box(network_params)
params = box.Box(training_params)

# Build neural networks
policy_net = Network(network_params, device).to(device)
target_net = Network(network_params, device).to(device)
target_net.load_state_dict(policy_net.state_dict())
# TODO: build the target network and set its weights to policy_net's wights (use state_dict from pytorch)


optimizer = optim.Adam(policy_net.parameters())
buffer = ReplayBuffer(100000)
epsilon = params.epsilon_start


# ============================================================================
# Plotting function
def plot_graphs(all_scores, all_losses, all_errors, axes):
    axes[0].plot(range(len(all_scores)), all_scores, color='blue')
    axes[0].set_title('Score over episodes')
    axes[1].plot(range(len(all_losses)), all_losses, color='blue')
    axes[1].set_title('Loss over episodes')
    axes[2].plot(range(len(all_errors)), all_errors, color='blue')
    axes[2].set_title('Mean Q error over episodes')


# Training functions
def select_action(s):
    '''
    This function gets a state and returns an action.
    The function uses an epsilon-greedy policy.
    :param s: the current state of the environment
    :return: a tensor of size [1,1] (use 'return torch.tensor([[action]], device=device, dtype=torch.long)')
    '''
    # TODO: implement action selection.
    global epsilon
    curr_action = torch.argmax(policy_net(s)).detach()
    if random.random() < epsilon:
        curr_action = random.randint(0,1)
    return torch.tensor([[curr_action]], device=device, dtype=torch.long)


def train_model():
    # Pros tips: 1. There is no need for any loop here!!!!! Use matrices!
    #            2. Use the pseudo-code.

    if len(buffer) < params.batch_size:
        # not enough samples
        return 0, 0

    # sample mini-batch
    transitions = buffer.sample(params.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    next_states_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)
    # not_done_batch = torch.cat(batch.not_done)
    not_done_batch = batch.not_done

    # Compute curr_Q = Q(s, a) - the model computes Q(s), then we select the columns of the taken actions.
    # Pros tips: First pass all s_batch through the network
    #            and then choose the relevant action for each state using the method 'gather'
    # TODO: fill curr_Q
    output_of_net = policy_net(state_batch)
    curr_Q = torch.gather(output_of_net, 1, action_batch)

    # Compute expected_Q (target value) for all states.
    # Don't forget that for terminal states we don't add the value of the next state.
    # Pros tips: Calculate the values for all next states ( Q_(s', max_a(Q_(s')) )
    #            and then mask next state's value with 0, where not_done is False (i.e., done).
    # TODO: fill expected_Q
    # output_of_net = target_net(next_states_batch)
    # Qs_of_best_actions = torch.gather(output_of_net, 1, torch.argmax(output_of_net))
    Qs_of_best_actions = target_net(next_states_batch).max(1)[0]
    Qs_of_best_actions[not not_done_batch] = 0.0
    Qs_of_best_actions = Qs_of_best_actions.detach()
    expected_Q = Qs_of_best_actions * params.gamma + reward_batch

    # Compute Huber loss. Smoother than MSE
    loss = F.smooth_l1_loss(curr_Q, expected_Q)

    # Optimize the model
    loss.backward()
    # clip gradients to help convergence
    nn.utils.clip_grad_norm_(policy_net.parameters(), params.grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    estimation_diff = torch.mean(curr_Q - expected_Q).item()

    return loss.item(), estimation_diff

# ============================================================================
def cartpole_play():

    FPS = 25
    visualize = 'True'

    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(env,'recording',force=True)
    net = Network(network_params, device).to(device)
    print('load best model ...')
    net.load_state_dict(torch.load('best.dat'))

    print('make movie ...')
    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False)).float()
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if visualize:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()

# ============================================================================
# Training loop
max_episodes = 200
max_score = 500
task_score = 0
# performances plots
all_scores = []
all_losses = []
all_errors = []
fig, axes = plt.subplots(3, 1)

# train for max_episodes
for i_episode in range(max_episodes):
    epsilon = max(epsilon*params.epsilon_decay, params.epsilon_end)
    ep_loss = []
    ep_error = []
    # Initialize the environment and state
    state = torch.tensor([env.reset()], device=device).float()
    done = False
    score = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        score += reward

        next_state = torch.tensor([next_state], device=device).float()
        reward = torch.tensor([reward], device=device).float()
        # Store the transition in memory
        buffer.push(state, action, next_state, reward, not done)

        # Update state
        state = next_state

        # Perform one optimization step (on the policy network)
        loss, Q_estimation_error = train_model()

        # save results
        ep_loss.append(loss)
        ep_error.append(Q_estimation_error)

        # soft target update
        if params.target_update == 'soft':
            # 0' ← τθ + (1 − τ )θ'
            # TODO: Implement soft target update.
            tau = 0.01
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
            # raise NotImplementedError

        if done or t >= max_score:
            print("Episode: {} | Current target score {} | Score: {}".format(i_episode+1, task_score, score))
            break

    # plot results
    all_scores.append(score)
    all_losses.append(np.average(ep_loss))
    all_errors.append(np.average(ep_error))
    plot_graphs(all_scores, all_losses, all_errors, axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.0001)

    # hard target update. Copying all weights and biases in DQN
    if params.target_update == 'hard':
        # TODO: Implement hard target update.
        # Copy the weights from policy_net to target_net after every x episodes
        if i_episode % 15 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # raise NotImplementedError

    # update task score
    if min(all_scores[-5:]) > task_score:
        task_score = min(all_scores[-5:])
        # TODO: store weights

print('------------------------------------------------------------------------------')
print('Final task score = ', task_score)

plt.ioff()
plt.show()
