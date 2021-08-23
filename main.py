import os
import random

import matplotlib.pyplot as plot
import numpy

# import kmeans_clustering as clustering

import a2c as a
import random_rooms_generator as r
import rooms
import kmeans_clustering as clustering
import datetime

"""
 Simulates a trial of an agent within an environment.
"""

NUMBER_OF_CLUSTER_REPRESENTITIVES = 10
PRETRAIN = True


def episode(environment, agent, gamma):
    state = environment.reset()
    # environment.render()
    discounted_return = 0
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = environment.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += (gamma ** time_step) * reward
        time_step += 1
    return discounted_return


"""
 Pushes the global model to the local devices
"""


def sync_model(global_model, local_models):
    for local_model in local_models:
        local_model.sync_policy(global_model)


def pick_n_per_cluster(path, number):
    paths = []
    n_paths_per_cluster = []
    # layout_paths = os.listdir('kmeans_clusteredLayouts')
    layout_paths = os.listdir('kmeans_clusteredLayouts_count_of_obstacles')
    layout_paths.sort()
    current_cluster = 0
    rooms_of_current_cluster = []
    for layout_path in layout_paths:
        # current_file = "kmeans_clusteredLayouts/" + layout_path
        current_file = "kmeans_clusteredLayouts_count_of_obstacles/" + layout_path
        if str(current_cluster) < layout_path[:1]:
            current_cluster += 1
            paths.append(rooms_of_current_cluster)
            rooms_of_current_cluster = []
        rooms_of_current_cluster.append(current_file)
        if layout_path == layout_paths[-1]:
            paths.append(rooms_of_current_cluster)
    for layout_path in paths:
        n_paths_per_cluster.append(layout_path[random.randint(0, len(layout_path) - 1)])
    print(n_paths_per_cluster)
    return n_paths_per_cluster


def pick_n_rooms(path, number):
    picked_rooms = []
    layout_path = os.listdir(path)
    for i in range(number):
        picked_rooms.append(path + "/" + layout_path[random.randint(0, len(layout_path) - 1)])
    print(picked_rooms)
    return picked_rooms


def load_n_rooms(number):
    picked_rooms = []
    layout_path = os.listdir('layouts')
    for i in range(number):
        picked_rooms.append(rooms.load_env('layouts' + "/" + layout_path[random.randint(0, len(layout_path) - 1)]))
    print(picked_rooms)
    return picked_rooms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# r.random_rooms_generator(10000)
# clustering.kMeans_clustering()
params = {}

# Domain setup for unclustered rooms
layout_paths = pick_n_rooms('layouts', NUMBER_OF_CLUSTER_REPRESENTITIVES)

# Domain setup for kmeans clustered rooms
# layout_paths = pick_n_per_cluster("", NUMBER_OF_CLUSTER_REPRESENTITIVES)

environments = [rooms.load_env(layout_path) for layout_path in layout_paths]
nr_environments = len(environments)
params["nr_actions"] = environments[0].action_space.n
params["nr_input_features"] = numpy.prod(environments[0].observation_space.shape)

# Hyperparameters are not affected by training
# learning size
params["gamma"] = 0.99
# batch size
params["alpha"] = 0.001
training_epochs = 1
averaging_period = 10


def pretrain_global_model():
    global_model = a.A2CLearner(params)
    layout_paths = os.listdir(
        'layouts')
    # environments = [rooms.load_env('layouts/' + layout_path) for layout_path in layout_paths]
    environments = load_n_rooms(10)
    returns = []
    for epoch in range(5):
        room_returns = []
        for environment in environments:
            room_return = episode(environment, global_model, params["gamma"])
            room_returns.append(room_return)
            print(f'Episode {epoch}, reward {room_return}')
        returns.append(numpy.mean(room_returns))
    # es ist plt nicht plot nein ah du hast den namen geändert :nerd: ich war es nicht xD ich auch nicht :o es war schon so im code, alles gut
    # ich würde den titel aus allen plots rausmachen dafür is die caption da also in latex. ok
    plot.plot(returns)
    plot.ylim(0, 5)
    plot.xlabel("episode")
    plot.ylabel("undiscounted return")
    plot.legend()
    plot.show()
    return global_model


# Agent setups
if PRETRAIN:
    global_model = pretrain_global_model()
    global_model.save_model()
else:
    global_model = a.A2CLearner(params)
    global_model.load_model('a2c_model.pth')
agents = [a.A2CLearner(params) for _ in range(nr_environments)]
# 0. Initially distribute global model to all local devices
sync_model(global_model, agents)
returns = [[] for _ in range(nr_environments)]
for epoch in range(training_epochs):
    # 1. Distributed training of local models
    for environment, agent, local_returns in zip(environments, agents, returns):
        epoch_returns = [episode(environment, agent, params["gamma"]) for _ in range(averaging_period)]
        local_returns.append(numpy.mean(epoch_returns))
    last_average_return = numpy.mean([local_returns[-1] for local_returns in returns])
    print("Completed epoch {} with average return {}".format(epoch, last_average_return))
    # 2. Average local models and redistribute updated global model.
    global_model = global_model.average_policies(agents)
    sync_model(global_model, agents)

# sns.set(style='whitegrid')
plot.ylim(-0.1, 1.1)
x = range(training_epochs)
for i, local_returns, in enumerate(returns):
    plot.plot(x, local_returns, label="agent {}".format(i))
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.legend()
plot.show()
# plot.savefig("diagram_{}.png".format(datetime.datetime.now()))
