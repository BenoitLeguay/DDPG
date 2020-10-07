import torch
from torch import nn
import numpy as np
from collections import namedtuple, deque
import random
from copy import deepcopy
import variable as v
import utils


class DDPG:
    def __init__(self, ddpg_init):
        self.discount_factor = ddpg_init["discount_factor"]
        self.update_target_rate = ddpg_init["update_target_rate"]
        self.num_action = ddpg_init["num_action"]

        self.update_count = 0.0
        self.update_after = ddpg_init["update_after"]
        self.update_every = ddpg_init["update_every"]

        self.state = None
        self.action = None

        self.actor = Actor(ddpg_init["actor_init"]).to(v.device)
        self.actor_target = Actor(ddpg_init["actor_init"]).to(v.device)
        self.critic = Critic(ddpg_init["critic_init"]).to(v.device)
        self.critic_target = Critic(ddpg_init["critic_init"]).to(v.device)

        self.replay_buffer = ReplayBuffer(ddpg_init["replay_buffer_init"])

        self.random_generator = np.random.RandomState(seed=ddpg_init['seed'])
        self.init_optimizers(critic_optimizer=ddpg_init['critic']['optimizer'],
                             actor_optimizer=ddpg_init['actor']['optimizer'])
        self.init_target_weights()  # do i have to do that ?

    def init_optimizers(self, critic_optimizer={}, actor_optimizer={}):
        self.critic.init_optimizer(critic_optimizer)
        self.actor.init_optimizer(actor_optimizer)

    def init_target_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update_target_weights(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.update_target_rate) +
                                    param.data * self.update_target_rate)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.update_target_rate) +
                                    param.data * self.update_target_rate)

    def policy(self, state):
        with torch.no_grad():
            action = self.actor(state)
        return action

    def episode_init(self, state):
        self.state = state
        state = utils.to_tensor(state).view((1, ) + state.shape)
        action = self.policy(state)
        self.action = action.cpu().tolist()

        return action.cpu().numpy()

    def update(self, next_state, reward, done):
        self.replay_buffer.append(deepcopy(self.state), deepcopy(self.action), reward, next_state, done)

        if len(self.replay_buffer) > self.update_after and self.update_count % self.update_every == 0.0:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()
            target = rewards + self.discount_factor * (1 - dones) * self.critic_target(next_states, self.actor_target(next_states))

            self.critic.update(states, actions, target)
            self.actor.update(states, ) 

            self.soft_update_target_weights()
        self.update_count += 1.0
        next_action = self.policy(next_state)
        self.action = next_action
        self.state = next_state
        return next_action


class Actor(torch.nn.Module):
    def __init__(self, actor_init):
        super(Actor, self).__init__()
        network_init = actor_init["network_init"]
        self.optimizer = None
        self.loss_history = list()
        self.action_upper = actor_init["action_upper"]
        self.action_down = actor_init["action_down"]

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(network_init["i_size"], network_init["l1_size"])
        self.l2 = nn.Linear(network_init["l1_size"], network_init["l2_size"])
        self.o = nn.Linear(network_init["l2_size"], 1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        o_tanh = self.tanh(self.o(x))

        o = (1/2) * (o_tanh * (self.action_upper - self.action_down) - (self.action_upper + self.action_down))

        return o

    def predict(self, state):
        pass

    def update(self):
        loss = None

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class Critic(torch.nn.Module):
    def __init__(self, critic_init):
        super(Critic, self).__init__()
        network_init = critic_init["network_init"]
        self.optimizer = None
        self.loss = torch.nn.MSELoss()
        self.loss_history = list()

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(network_init["i_size"], network_init["l1_size"])
        self.l2 = nn.Linear(network_init["l1_size"], network_init["l2_size"])
        self.o = nn.Linear(network_init["l2_size"], 1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, states, actions):
        x = torch.cat([states, actions])
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        o = self.o(x)

        return o

    def update(self, states, actions, target):
        loss = self.loss(self.forward(states, actions), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class ReplayBuffer:
    def __init__(self, replay_buffer_init):
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.max_len = replay_buffer_init["max_len"]
        self.batch_size = replay_buffer_init["batch_size"]
        self.buffer = deque(maxlen=self.max_len)

    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = utils.to_tensor(np.vstack([e.state for e in experiences if e is not None]))
        actions = utils.to_tensor(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = utils.to_tensor(np.vstack([e.reward for e in experiences if e is not None]))
        next_states = utils.to_tensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = utils.to_tensor(np.vstack([e.done for e in experiences if e is not None]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
