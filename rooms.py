import pathlib
import random

import gym
import numpy
from gym import spaces
from gym.utils import seeding

MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3

ROOMS_ACTIONS = [MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]

AGENT_CHANNEL = 0
GOAL_CHANNEL = 1
OBSTACLE_CHANNEL = 2
NR_CHANNELS = len([AGENT_CHANNEL, GOAL_CHANNEL, OBSTACLE_CHANNEL])

max_room_width = 12
max_room_height = 12


class RoomsEnv(gym.Env):

    def __init__(self, width, height, obstacles, items, time_limit, stochastic=False):
        self.seed()
        self.action_space = spaces.Discrete(len(ROOMS_ACTIONS))
        self.observation_space = spaces.Box(-numpy.inf, numpy.inf, shape=(NR_CHANNELS, width, height))
        self.agent_position = None
        self.done = False
        self.goal_position = (width - 2, height - 2)
        self.obstacles = obstacles
        self.items = items
        self.time_limit = time_limit
        self.time = 0
        self.width = width
        self.height = height
        self.stochastic = stochastic
        self.undiscounted_return = 0
        self.state_history = []
        self.reset()

    def is_subgoal(self, state):
        is_at_goal = self.agent_position == self.goal_position
        x, y = self.agent_position
        is_at_door_vertical = state[y - 1][x][OBSTACLE_CHANNEL] == 1 and state[y + 1][x][OBSTACLE_CHANNEL] == 1
        is_at_door_horizontal = state[y][x - 1][OBSTACLE_CHANNEL] == 1 and state[y][x + 1][OBSTACLE_CHANNEL] == 1
        return is_at_goal or is_at_door_vertical or is_at_door_horizontal

    def state(self):
        state = numpy.zeros((NR_CHANNELS, self.width, self.height))
        x_agent, y_agent = self.agent_position
        state[AGENT_CHANNEL][x_agent][y_agent] = 1
        x_goal, y_goal = self.goal_position
        state[GOAL_CHANNEL][x_goal][y_goal] = 1
        for obstacle in self.obstacles:
            x, y = obstacle
            state[OBSTACLE_CHANNEL][x][y] = 1
        return numpy.swapaxes(state, 0, 2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.stochastic and numpy.random.rand() < 0.2:
            action = random.choice(ROOMS_ACTIONS)
        return self.step_with_action(action)

    def step_with_action(self, action):
        if self.done:
            return self.agent_position, 0, self.done, {}
        self.time += 1
        self.state_history.append(self.state())
        x, y = self.agent_position
        reward = 0
        if action == MOVE_NORTH and y + 1 < self.height:
            self.set_position_if_no_obstacle((x, y + 1))
        elif action == MOVE_SOUTH and y - 1 >= 0:
            self.set_position_if_no_obstacle((x, y - 1))
        if action == MOVE_WEST and x - 1 >= 0:
            self.set_position_if_no_obstacle((x - 1, y))
        elif action == MOVE_EAST and x + 1 < self.width:
            self.set_position_if_no_obstacle((x + 1, y))
        goal_reached = self.agent_position == self.goal_position
        if goal_reached:
            reward += 4
        if self.agent_position in self.items:
            reward += 0.2
            self.items.remove(self.agent_position)
        self.undiscounted_return += reward
        self.done = goal_reached or self.time >= self.time_limit
        return self.state(), reward, self.done, {}

    def set_position_if_no_obstacle(self, new_position):
        if new_position not in self.obstacles:
            self.agent_position = new_position

    def reset(self):
        self.done = False
        self.agent_position = (1, 1)
        self.time = 0
        self.state_history.clear()
        return self.state()

    def state_summary(self, state):
        return {
            "agent_x": self.agent_position[0],
            "agent_y": self.agent_position[0],
            "goal_x": self.goal_position[0],
            "goal_y": self.goal_position[0],
            "is_subgoal": self.is_subgoal(state),
            "time_step": self.time,
            "score": self.undiscounted_return
        }


def read_map_file(path):
    file = pathlib.Path(path)
    # assert file.is_file()
    with open(path) as f:
        content = f.readlines()
    obstacles = []
    items = []
    width = 0
    height = 0
    for y, line in enumerate(content):
        for x, cell in enumerate(line.strip().split()):
            if cell == '#':
                obstacles.append((x, y))
            elif cell == 'x':
                items.append((x,y))
            width = x
        height = y
    width += 1
    height += 1
    return width, height, obstacles, items


def map_to_flattened_matrix(path):
    file = pathlib.Path(path)
    assert file.is_file()
    with open(path) as f:
        content = f.readlines()
    flattened_matrix = []
    for i, line in enumerate(content):
        for j, cell in enumerate(line.strip().split()):
            if i != 0 and i < (max_room_height - 1) and j != 0 and j < (max_room_width - 1):
                if cell == '#':
                    flattened_matrix.append(0)
                elif cell == '.':
                    flattened_matrix.append(1)
                elif cell == 'x':
                    flattened_matrix.append(2)
    return flattened_matrix

def map_to_flattened_one_hot_matrix(path):
    file = pathlib.Path(path)
    assert file.is_file()
    with open(path) as f:
        content = f.readlines()
    flattened_matrix = []
    for i, line in enumerate(content):
        for j, cell in enumerate(line.strip().split()):
            if i != 0 and i < (max_room_height - 1) and j != 0 and j < (max_room_width - 1):
                if cell == '#':
                    flattened_matrix.append(0)
                    flattened_matrix.append(0)
                    flattened_matrix.append(1)
                elif cell == '.':
                    flattened_matrix.append(0)
                    flattened_matrix.append(1)
                    flattened_matrix.append(0)
                elif cell == 'x':
                    flattened_matrix.append(1)
                    flattened_matrix.append(0)
                    flattened_matrix.append(0)
    return flattened_matrix


def count_of_obstacles(path):
    file = pathlib.Path(path)
    assert file.is_file()
    with open(path) as f:
        content = f.readlines()
    count_of_obstacles = -32
    for y, line in enumerate(content):
        for x, cell in enumerate(line.strip().split()):
            if cell == '#':
                count_of_obstacles += 1
    return [count_of_obstacles]


def load_env(path, time_limit=1000, stochastic=False):
    width, height, obstacles, items = read_map_file(path)
    return RoomsEnv(width, height, obstacles, items, time_limit, stochastic)
