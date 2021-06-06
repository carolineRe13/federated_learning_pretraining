import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

"""
class A2CNet(nn.Module):
    def __init__(self, nr_input_features, nr_actions):
        super(A2CNet, self).__init__()
        nr_hidden_units = 64
        self.nr_input_features = nr_input_features
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.action_head = nn.Linear(nr_hidden_units, nr_actions)
        self.value_head = nn.Linear(nr_hidden_units, 1)

    def forward(self, x):
        x = x.view(-1, self.nr_input_features)
        x = self.fc_net(x)
        x = x.view(x.size(0), -1)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)
"""


class PolicyNet(nn.Module):
    def __init__(self, nr_input_features, nr_actions):
        super(PolicyNet, self).__init__()
        nr_hidden_units = 64
        self.nr_input_features = nr_input_features
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.action_head = nn.Linear(nr_hidden_units, nr_actions)

    def forward(self, x):
        x = x.view(x.size(0), self.nr_input_features)
        x = self.fc_net(x)
        x = x.view(x.size(0), -1)
        return F.softmax(self.action_head(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, nr_input_features):
        super(ValueNet, self).__init__()
        nr_hidden_units = 64
        self.nr_input_features = nr_input_features
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        # only difference to PolicyNet
        self.value_head = nn.Linear(nr_hidden_units, 1)

    def forward(self, x):
        x = x.view(-1, self.nr_input_features)
        x = self.fc_net(x)
        x = x.view(x.size(0), -1)
        return self.value_head(x)


"""
 Autonomous agent using Synchronous Actor-Critic.
"""


class A2CLearner:

    def __init__(self, params):
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.alpha = params["alpha"]
        self.nr_input_features = params["nr_input_features"]
        self.transitions = []
        # force torch to use cpu instead of gpu
        self.device = torch.device("cpu")
        self.policy_net = PolicyNet(self.nr_input_features, self.nr_actions).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=params["alpha"])
        self.value_net = ValueNet(self.nr_input_features).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=params["alpha"])
        parameters = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=params["alpha"])

    """
     Overwrites the current policy with a policy of another agent.
     Should only be called for local nodes.
    """

    def sync_policy(self, global_agent):
        self.policy_net.load_state_dict(global_agent.policy_net.state_dict())
        self.policy_net.eval()

    """
     Updates the global model by averaging all local models.
     Should only be called for the centralized node. 
    """

    def average_policies(self, local_agents):
        with torch.no_grad():
            for param in self.policy_net.parameters():
                param.fill_(0.0)
            for local_agent in local_agents:
                for global_param, local_param in \
                        zip(self.policy_net.parameters(), local_agent.policy_net.parameters()):
                    weight = 1.0 / len(local_agents)  # Can be weighted w.r.t. amount of data
                    global_param.add_(weight * local_param)
        return self

    """
     Samples a new action using the policy network.
    """

    def policy(self, state):
        action_probs = self.predict_policy([state])
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()

    """
     Predicts the action probabilities.
    """

    def predict_policy(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.policy_net(states)

    """
     Predicts state values.
    """

    def predict_value(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.value_net(states)

    """
     Performs a learning update of the currently learned policy and value function.
    """

    def update(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        loss = None
        if done:
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))
            discounted_returns = []

            # Calculate and normalize discounted returns
            R = 0
            for reward in reversed(rewards):
                R = reward + self.gamma * R
                discounted_returns.append(R)
            discounted_returns.reverse()
            states = states
            next_states = next_states
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
            discounted_returns = torch.tensor(discounted_returns, device=self.device, dtype=torch.float).detach()
            normalized_returns = (discounted_returns - discounted_returns.mean())
            normalized_returns /= (discounted_returns.std() + self.eps)

            # Calculate losses of policy and value function
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            action_probs = self.predict_policy(states)
            state_values = self.predict_value(states)
            states = torch.tensor(states, device=self.device, dtype=torch.float)
            policy_losses = []
            value_losses = []
            for probs, action, value, R in zip(action_probs, actions, state_values, normalized_returns):
                advantage = R - value.item()
                m = Categorical(probs)
                policy_losses.append(-m.log_prob(action) * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()

            # Optimize joint loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.transitions.clear()
        return loss
