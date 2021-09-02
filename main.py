import os
import random
import time

import matplotlib.pyplot as plot
import numpy

import a2c as a
import rooms

# import kmeans_clustering as clustering

import random_rooms_generator as r

"""
 Simulates a trial of an agent within an environment.
"""

NUMBER_OF_CLUSTER_REPRESENTATIVES = 18
PRETRAIN = True
PRETRAINING_EPOCHS = 100
NUM_RANDOM_ROOMS = 50


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
    path = 'kmeans_clusteredLayouts_one_hot'
    paths = []
    n_paths_per_cluster = []
    # layout_paths = os.listdir('kmeans_clusteredLayouts')
    layout_paths = os.listdir(path)
    # print([i.split('_', 1)[0] for i in layout_paths])
    layout_paths.sort(key = lambda x: int(x.split('_')[0]))
    # layout_paths = sorted(layout_paths)
    current_cluster = 0
    count_per_cluster = []
    rooms_of_current_cluster = []
    i = 0
    for layout_path in layout_paths:
        i += 1
        # current_file = "kmeans_clusteredLayouts/" + layout_path
        current_file = "kmeans_clusteredLayouts_one_hot/" + layout_path
        if current_cluster < int(layout_path.split('_')[0]):
            current_cluster += 1
            paths.append(rooms_of_current_cluster)
            rooms_of_current_cluster = []
            count_per_cluster.append(i)
            i = 0
        rooms_of_current_cluster.append(current_file)
        if layout_path == layout_paths[-1]:
            paths.append(rooms_of_current_cluster)
    for layout_path in paths:
        n_paths_per_cluster.append(layout_path[random.randint(0, len(layout_path) - 1)])
        n_paths_per_cluster.append(layout_path[random.randint(0, len(layout_path) - 1)])
        n_paths_per_cluster.append(layout_path[random.randint(0, len(layout_path) - 1)])
        n_paths_per_cluster.append(layout_path[random.randint(0, len(layout_path) - 1)])
    print(n_paths_per_cluster)
    return n_paths_per_cluster


def pick_n_rooms(path, number):
    picked_rooms = []
    layout_path = os.listdir(path)
    for i in range(number):
        picked_rooms.append(path + "/" + layout_path[random.randint(0, len(layout_path) - 1)])
    print(picked_rooms)
    print(picked_rooms.__len__())
    return picked_rooms


def load_n_rooms(number):
    picked_rooms = []
    layout_path = os.listdir('layouts')
    for i in range(number):
        picked_rooms.append(rooms.load_env('layouts' + "/" + layout_path[random.randint(0, len(layout_path) - 1)]))
    print(picked_rooms)
    return picked_rooms


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
r.random_rooms_generator(50)
# clustering.kMeans_clustering()
params = {}

# Domain setup for unclustered rooms
# layout_paths = pick_n_rooms('layouts', NUMBER_OF_CLUSTER_REPRESENTATIVES)
layout_paths = os.listdir('test_layouts')
layout_paths = ["test_layouts/" + layout_path for layout_path in layout_paths]

# Domain setup for kmeans clustered rooms
# layout_paths = pick_n_per_cluster("", NUMBER_OF_CLUSTER_REPRESENTITIVES)

environments = [rooms.load_env(layout_path) for layout_path in layout_paths]
# environments = [rooms.load_env("layouts/rooms_0.txt")]
nr_environments = len(environments)
params["nr_actions"] = environments[0].action_space.n
params["nr_input_features"] = numpy.prod(environments[0].observation_space.shape)

# Hyperparameters are not affected by training
# learning size
params["gamma"] = 0.99
# batch size
params["alpha"] = 0.001
training_epochs = 50
averaging_period = 10


def pretrain_global_model():
    global_model = a.A2CLearner(params)
    global_model.load_model('a2c_model_clustering4.pth')
    pick_n_rooms('layouts', NUM_RANDOM_ROOMS)
    # environments = load_n_rooms(NUM_RANDOM_ROOMS)
    # environments_path = ['layouts/rooms_5435.txt', 'layouts/rooms_5460.txt', 'layouts/rooms_823.txt', 'layouts/rooms_443.txt', 'layouts/rooms_7951.txt', 'layouts/rooms_2117.txt', 'layouts/rooms_7520.txt', 'layouts/rooms_7223.txt', 'layouts/rooms_7617.txt', 'layouts/rooms_7304.txt', 'layouts/rooms_1832.txt', 'layouts/rooms_6540.txt', 'layouts/rooms_2981.txt', 'layouts/rooms_9422.txt', 'layouts/rooms_9361.txt', 'layouts/rooms_1445.txt', 'layouts/rooms_1606.txt', 'layouts/rooms_8163.txt', 'layouts/rooms_8589.txt', 'layouts/rooms_1608.txt', 'layouts/rooms_1891.txt', 'layouts/rooms_2000.txt', 'layouts/rooms_1757.txt', 'layouts/rooms_8055.txt', 'layouts/rooms_3993.txt', 'layouts/rooms_7899.txt', 'layouts/rooms_2505.txt', 'layouts/rooms_2549.txt', 'layouts/rooms_7068.txt', 'layouts/rooms_4261.txt', 'layouts/rooms_6197.txt', 'layouts/rooms_8908.txt', 'layouts/rooms_5255.txt', 'layouts/rooms_8203.txt', 'layouts/rooms_7302.txt', 'layouts/rooms_8562.txt', 'layouts/rooms_9829.txt', 'layouts/rooms_4229.txt', 'layouts/rooms_5270.txt', 'layouts/rooms_8224.txt', 'layouts/rooms_6545.txt', 'layouts/rooms_1205.txt', 'layouts/rooms_2513.txt', 'layouts/rooms_1329.txt', 'layouts/rooms_1700.txt', 'layouts/rooms_8407.txt', 'layouts/rooms_5484.txt', 'layouts/rooms_5836.txt', 'layouts/rooms_8673.txt', 'layouts/rooms_1439.txt']
    # environments_path = ['kmeans_clusteredLayouts_one_hot/0_186.txt', 'kmeans_clusteredLayouts_one_hot/1_528.txt', 'kmeans_clusteredLayouts_one_hot/2_353.txt', 'kmeans_clusteredLayouts_one_hot/3_677.txt', 'kmeans_clusteredLayouts_one_hot/4_492.txt', 'kmeans_clusteredLayouts_one_hot/5_155.txt', 'kmeans_clusteredLayouts_one_hot/6_97.txt', 'kmeans_clusteredLayouts_one_hot/7_428.txt', 'kmeans_clusteredLayouts_one_hot/8_763.txt', 'kmeans_clusteredLayouts_one_hot/9_532.txt', 'kmeans_clusteredLayouts_one_hot/10_674.txt', 'kmeans_clusteredLayouts_one_hot/11_272.txt', 'kmeans_clusteredLayouts_one_hot/12_744.txt', 'kmeans_clusteredLayouts_one_hot/13_742.txt', 'kmeans_clusteredLayouts_one_hot/14_579.txt', 'kmeans_clusteredLayouts_one_hot/15_299.txt', 'kmeans_clusteredLayouts_one_hot/16_708.txt', 'kmeans_clusteredLayouts_one_hot/17_332.txt']
    environments_path = ['kmeans_clusteredLayouts_one_hot/0_289.txt', 'kmeans_clusteredLayouts_one_hot/0_666.txt', 'kmeans_clusteredLayouts_one_hot/0_347.txt', 'kmeans_clusteredLayouts_one_hot/0_380.txt', 'kmeans_clusteredLayouts_one_hot/1_528.txt', 'kmeans_clusteredLayouts_one_hot/1_23.txt', 'kmeans_clusteredLayouts_one_hot/1_431.txt', 'kmeans_clusteredLayouts_one_hot/1_697.txt', 'kmeans_clusteredLayouts_one_hot/2_238.txt', 'kmeans_clusteredLayouts_one_hot/2_746.txt', 'kmeans_clusteredLayouts_one_hot/2_117.txt', 'kmeans_clusteredLayouts_one_hot/2_460.txt', 'kmeans_clusteredLayouts_one_hot/3_0.txt', 'kmeans_clusteredLayouts_one_hot/3_655.txt', 'kmeans_clusteredLayouts_one_hot/3_349.txt', 'kmeans_clusteredLayouts_one_hot/3_265.txt', 'kmeans_clusteredLayouts_one_hot/4_502.txt', 'kmeans_clusteredLayouts_one_hot/4_312.txt', 'kmeans_clusteredLayouts_one_hot/4_336.txt', 'kmeans_clusteredLayouts_one_hot/4_317.txt', 'kmeans_clusteredLayouts_one_hot/5_301.txt', 'kmeans_clusteredLayouts_one_hot/5_386.txt', 'kmeans_clusteredLayouts_one_hot/5_88.txt', 'kmeans_clusteredLayouts_one_hot/5_405.txt', 'kmeans_clusteredLayouts_one_hot/6_330.txt', 'kmeans_clusteredLayouts_one_hot/6_497.txt', 'kmeans_clusteredLayouts_one_hot/6_340.txt', 'kmeans_clusteredLayouts_one_hot/6_707.txt', 'kmeans_clusteredLayouts_one_hot/7_68.txt', 'kmeans_clusteredLayouts_one_hot/7_436.txt', 'kmeans_clusteredLayouts_one_hot/7_704.txt', 'kmeans_clusteredLayouts_one_hot/7_139.txt', 'kmeans_clusteredLayouts_one_hot/8_738.txt', 'kmeans_clusteredLayouts_one_hot/8_162.txt', 'kmeans_clusteredLayouts_one_hot/8_797.txt', 'kmeans_clusteredLayouts_one_hot/8_763.txt', 'kmeans_clusteredLayouts_one_hot/9_297.txt', 'kmeans_clusteredLayouts_one_hot/9_297.txt', 'kmeans_clusteredLayouts_one_hot/9_171.txt', 'kmeans_clusteredLayouts_one_hot/9_532.txt', 'kmeans_clusteredLayouts_one_hot/10_550.txt', 'kmeans_clusteredLayouts_one_hot/10_617.txt', 'kmeans_clusteredLayouts_one_hot/10_39.txt', 'kmeans_clusteredLayouts_one_hot/10_455.txt', 'kmeans_clusteredLayouts_one_hot/11_668.txt', 'kmeans_clusteredLayouts_one_hot/11_529.txt', 'kmeans_clusteredLayouts_one_hot/11_481.txt', 'kmeans_clusteredLayouts_one_hot/11_721.txt', 'kmeans_clusteredLayouts_one_hot/12_321.txt', 'kmeans_clusteredLayouts_one_hot/12_780.txt', 'kmeans_clusteredLayouts_one_hot/12_48.txt', 'kmeans_clusteredLayouts_one_hot/12_482.txt', 'kmeans_clusteredLayouts_one_hot/13_552.txt', 'kmeans_clusteredLayouts_one_hot/13_678.txt', 'kmeans_clusteredLayouts_one_hot/13_625.txt', 'kmeans_clusteredLayouts_one_hot/13_126.txt', 'kmeans_clusteredLayouts_one_hot/14_759.txt', 'kmeans_clusteredLayouts_one_hot/14_679.txt', 'kmeans_clusteredLayouts_one_hot/14_263.txt', 'kmeans_clusteredLayouts_one_hot/14_540.txt', 'kmeans_clusteredLayouts_one_hot/15_119.txt', 'kmeans_clusteredLayouts_one_hot/15_752.txt', 'kmeans_clusteredLayouts_one_hot/15_441.txt', 'kmeans_clusteredLayouts_one_hot/15_44.txt', 'kmeans_clusteredLayouts_one_hot/16_670.txt', 'kmeans_clusteredLayouts_one_hot/16_708.txt', 'kmeans_clusteredLayouts_one_hot/17_495.txt', 'kmeans_clusteredLayouts_one_hot/17_478.txt', 'kmeans_clusteredLayouts_one_hot/17_608.txt', 'kmeans_clusteredLayouts_one_hot/17_464.txt']
    environments = [rooms.load_env(layout_path) for layout_path in environments_path]
    returns = []
    from collections import defaultdict
    room_returns = defaultdict(list)
    for epoch in range(PRETRAINING_EPOCHS):
        episode_returns = []
        for environment in environments:
            room_return = episode(environment, global_model, params["gamma"])
            episode_returns.append(room_return)
            print(f'Room {environment.room_name}, reward {room_return}')
            room_returns[environment.room_name].append(room_return)
        returns.append(numpy.mean(episode_returns))
        print(f'Episode {epoch}, average reward {numpy.mean(episode_returns)}')

    print('-----------------------------------------')
    print(room_returns)
    print('-----------------------------------------')
    print(returns)
    plot.plot(returns)
    plot.ylim(0, 5)
    plot.xlabel("Episode")
    plot.ylabel("Average Reward")
    plot.legend()
    plot.show()
    return global_model


def pretrain_global_model_clustering():
    global_model = a.A2CLearner(params)
    # global_model.load_model('a2c_model.pth')
    # pick_n_rooms('layouts', NUM_RANDOM_ROOMS)
    # environments = load_n_rooms(NUM_RANDOM_ROOMS)
    environments_path = ['layouts/rooms_5435.txt', 'layouts/rooms_5460.txt', 'layouts/rooms_823.txt', 'layouts/rooms_443.txt', 'layouts/rooms_7951.txt', 'layouts/rooms_2117.txt', 'layouts/rooms_7520.txt', 'layouts/rooms_7223.txt', 'layouts/rooms_7617.txt', 'layouts/rooms_7304.txt', 'layouts/rooms_1832.txt', 'layouts/rooms_6540.txt', 'layouts/rooms_2981.txt', 'layouts/rooms_9422.txt', 'layouts/rooms_9361.txt', 'layouts/rooms_1445.txt', 'layouts/rooms_1606.txt', 'layouts/rooms_8163.txt', 'layouts/rooms_8589.txt', 'layouts/rooms_1608.txt', 'layouts/rooms_1891.txt', 'layouts/rooms_2000.txt', 'layouts/rooms_1757.txt', 'layouts/rooms_8055.txt', 'layouts/rooms_3993.txt', 'layouts/rooms_7899.txt', 'layouts/rooms_2505.txt', 'layouts/rooms_2549.txt', 'layouts/rooms_7068.txt', 'layouts/rooms_4261.txt', 'layouts/rooms_6197.txt', 'layouts/rooms_8908.txt', 'layouts/rooms_5255.txt', 'layouts/rooms_8203.txt', 'layouts/rooms_7302.txt', 'layouts/rooms_8562.txt', 'layouts/rooms_9829.txt', 'layouts/rooms_4229.txt', 'layouts/rooms_5270.txt', 'layouts/rooms_8224.txt', 'layouts/rooms_6545.txt', 'layouts/rooms_1205.txt', 'layouts/rooms_2513.txt', 'layouts/rooms_1329.txt', 'layouts/rooms_1700.txt', 'layouts/rooms_8407.txt', 'layouts/rooms_5484.txt', 'layouts/rooms_5836.txt', 'layouts/rooms_8673.txt', 'layouts/rooms_1439.txt']
    environments = [rooms.load_env(layout_path) for layout_path in environments_path]
    returns = []
    from collections import defaultdict
    room_returns = defaultdict(list)
    for epoch in range(PRETRAINING_EPOCHS):
        episode_returns = []
        for environment in environments:
            room_return = episode(environment, global_model, params["gamma"])
            episode_returns.append(room_return)
            print(f'Room {environment.room_name}, reward {room_return}')
            room_returns[environment.room_name].append(room_return)
        returns.append(numpy.mean(episode_returns))
        print(f'Episode {epoch}, average reward {numpy.mean(episode_returns)}')

    print('-----------------------------------------')
    print(room_returns)
    print('-----------------------------------------')
    print(returns)
    plot.plot(returns)
    plot.ylim(0, 5)
    plot.xlabel("Episode")
    plot.ylabel("Average Reward")
    plot.legend()
    plot.show()
    return global_model


pick_n_per_cluster('layouts/', 4)

# Agent setups
if PRETRAIN:
    global_model = pretrain_global_model()
    global_model.save_model('a2c_model_clustering4_2.pth')
    # global_model.save_model('a2c_model2.pth')
else:
    global_model = a.A2CLearner(params)
    global_model.load_model('a2c_model2.pth')
    # global_model.load_model('a2c_model_clustering2.pth')
agents = [a.A2CLearner(params) for _ in range(nr_environments)]
# 0. Initially distribute global model to all local devices
# global_model = a.A2CLearner(params)
sync_model(global_model, agents)
returns = [[] for _ in range(nr_environments)]
average_returns = []
for epoch in range(training_epochs):
    # 1. Distributed training of local models
    for environment, agent, local_returns in zip(environments, agents, returns):
        print("entered")
        epoch_returns = [episode(environment, agent, params["gamma"]) for _ in range(averaging_period)]
        local_returns.append(numpy.mean(epoch_returns))
    last_average_return = numpy.mean([local_returns[-1] for local_returns in returns])
    average_returns.append(last_average_return)
    print("Completed epoch {} with average return {}".format(epoch, last_average_return))
    # 2. Average local models and redistribute updated global model.
    global_model = global_model.average_policies(agents)
    sync_model(global_model, agents)
global_model.save_model('first_run.pth')

# sns.set(style='whitegrid')
plot.ylim(-0.1, 1.1)
x = range(training_epochs)
print(average_returns)
for i, local_returns, in enumerate(returns):
    plot.plot(x, local_returns, label="agent {}".format(i))
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.legend()
plot.show()
# plot.savefig("diagram_{}.png".format(datetime.datetime.now()))
