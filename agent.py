import torch
from torch import nn
import numpy as np
from collections import namedtuple, deque
import random
from copy import deepcopy
import variable as v
import utils
torch.autograd.set_detect_anomaly(True)


class DDPG:
    def __init__(self, ddpg_init):
        torch.manual_seed(ddpg_init['seed'])
        self.discount_factor = ddpg_init["discount_factor"]
        self.update_target_rate = ddpg_init["update_target_rate"]
        self.num_action = ddpg_init["action_shape"]
        self.train = True

        self.exploration_noise = noise_dict[ddpg_init["noise_type"]](**ddpg_init["noise"], seed=ddpg_init["seed"])

        self.update_count = 0.0
        self.update_after = ddpg_init["update_after"]
        self.update_every = ddpg_init["update_every"]

        self.state = None
        self.action = None

        self.actor = Actor(ddpg_init["actor"]).to(v.device)
        self.actor_target = Actor(ddpg_init["actor"]).to(v.device)
        self.critic = Critic(ddpg_init["critic"]).to(v.device)
        self.critic_target = Critic(ddpg_init["critic"]).to(v.device)

        self.replay_buffer = ReplayBuffer(ddpg_init["replay_buffer"], seed=ddpg_init["seed"])

        self.init_optimizers(critic_optimizer=ddpg_init['critic']['optimizer'],
                             actor_optimizer=ddpg_init['actor']['optimizer'])
        self.init_target_weights()

    def init_optimizers(self, critic_optimizer={}, actor_optimizer={}):
        self.critic.init_optimizer(critic_optimizer)
        self.actor.init_optimizer(actor_optimizer)

    def init_target_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update_target_weights(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * self.update_target_rate +
                                    param.data * (1.0 - self.update_target_rate))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * self.update_target_rate +
                                    param.data * (1.0 - self.update_target_rate))

    def set_test(self):
        self.train = False
        self.actor.eval()

    def set_train(self):
        self.train = True
        self.actor.train()

    def reset_noise(self):
        self.exploration_noise.reset()

    def policy(self, state):
        state = utils.to_tensor(state).view((1, ) + state.shape)

        with torch.no_grad():
            action = self.actor.predict(state).squeeze(1).cpu().numpy()
        if self.train:
            action = action + self.exploration_noise()
        return action

    def episode_init(self, state):
        self.state = state

        action = self.policy(state)
        self.action = action

        return action

    def update(self, next_state, reward, done):
        self.replay_buffer.append(deepcopy(self.state), deepcopy(self.action), reward, next_state, done)

        if len(self.replay_buffer) > self.update_after and self.update_count % self.update_every == 0.0:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()

            next_actions = self.actor_target(next_states).detach()
            target = rewards + self.discount_factor * (1 - dones) * self.critic_target(next_states, next_actions)
            self.critic.update(states, actions, target)

            q_values = self.critic(states, self.actor(states))
            self.actor.update(q_values)

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
        self.action_high = actor_init["action_high"]
        self.action_low = actor_init["action_low"]

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(network_init["i_shape"], network_init["l1_shape"])
        self.l2 = nn.Linear(network_init["l1_shape"], network_init["l2_shape"])
        self.o = nn.Linear(network_init["l2_shape"], network_init["o_shape"])

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        o_tanh = self.tanh(self.o(x))

        o = (1/2) * (o_tanh * (self.action_high - self.action_low) - (self.action_high + self.action_low))

        return o

    def predict(self, state):
        action = self.forward(state)
        return action

    def update(self, q_values):
        loss = - torch.mean(q_values)

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
        self.l_states = nn.Linear(network_init["i_shape"], network_init["l1_shape"])
        self.l2 = nn.Linear(network_init["l1_shape"] + network_init["action_shape"], network_init["l2_shape"])
        self.o = nn.Linear(network_init["l2_shape"], 1)

    def init_optimizer(self, optimizer_args):
        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_args)

    def forward(self, states, actions):

        embedded_states = self.relu(self.l_states(states))

        x = torch.cat([embedded_states, actions], axis=1)
        x = self.relu(self.l2(x))
        o = self.o(x)

        return o

    def update(self, states, actions, target):
        loss = self.loss(self.forward(states, actions), target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(loss.item())


class ReplayBuffer:
    def __init__(self, replay_buffer_init, seed=42):
        random.seed(seed)
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
        actions = utils.to_tensor(np.vstack([e.action for e in experiences if e is not None]))
        rewards = utils.to_tensor(np.vstack([e.reward for e in experiences if e is not None]))
        next_states = utils.to_tensor(np.vstack([e.next_state for e in experiences if e is not None]))
        dones = utils.to_tensor(np.vstack([e.done for e in experiences if e is not None]))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class GaussianNoise:
    def __init__(self, mu, sigma, action_dim, seed=42):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim
        self.random_generator = np.random.RandomState(seed=seed)

    def reset(self):
        pass

    def __call__(self):
        return self.random_generator.normal(loc=self.mu, scale=self.sigma, size=self.action_dim)

    def __repr__(self):
        return f'GaussianNoise(mu={self.mu}, sigma={self.sigma}, action_dim={self.action_dim})'


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, action_dim, theta=.15, dt=1e-2, x0=None, seed=42):
        self.theta = theta
        self.mu = np.ones(action_dim) * mu
        self.sigma = np.ones(action_dim) * sigma
        self.dt = dt
        self.x0 = x0
        self.random_state = np.random.RandomState(seed=seed)
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * self.random_state.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


noise_dict = {'ou': OrnsteinUhlenbeckActionNoise, 'gaussian': GaussianNoise}
