import pathlib
import random

import numpy as np

# max_room_width = random.randint(10, 50)
# max_room_height = random.randint(10, 50)


max_room_width = 12
max_room_height = 12


def random_rooms_generator(number):
    i = -1
    # until a room is solvable for i, we create new rooms
    # the size of the count of obstacles are random
    pathlib.Path('../layouts').mkdir(parents=True, exist_ok=True)
    while i < number:
        max_obstacles = ((max_room_height - 2) * (max_room_width - 2))
        max_booster = 5
        obstacle_matrix = [[0 for _ in range(max_room_width)] for _ in range(max_room_height)]
        for _ in range(random.randint(10, max_obstacles)):
            x = random.randint(1, max_room_height - 2)
            y = random.randint(1, max_room_width - 2)
            obstacle_matrix[x][y] = '#'
        for x in range(max_room_width):
            obstacle_matrix[0][x] = '#'
            obstacle_matrix[max_room_height - 1][x] = '#'
        for x in range(max_room_height):
            obstacle_matrix[x][0] = '#'
            obstacle_matrix[x][max_room_width - 1] = '#'
        for _ in range(random.randint(2, max_booster)):
            x = random.randint(1, max_room_height - 2)
            y = random.randint(1, max_room_width - 2)
            obstacle_matrix[x][y] = 'x'
        # when the room is solvable then we create a txt file containing it
        if is_room_solvable(obstacle_matrix, max_room_width, max_room_height, i):
            i += 1
            a = np.array(obstacle_matrix)
            a = np.where(a == '0', '.', a)
            mat = np.matrix(a)
            with open('layouts/rooms_{}.txt'.format(i), 'w+') as f:
                for line in mat:
                    np.savetxt(f, line, fmt='%s')
# hallo hallo moien :)

def is_room_solvable(obstacle_matrix, max_room_width, max_room_height, i, x=None, y=None):
    # initialize x,y with the coordinates of the start location
    if x is None:
        x, y = 1, 1

    # if there is no obstacle and the coordinates have not been visited yet
    if obstacle_matrix[y][x] == 0:

        # mark tile (x, y) as visited
        obstacle_matrix[y][x] = '.'

        # check if one of the adjacent paths from (x, y) lead to a solution
        # this is recursive, so when one of the calls encounter an exit, then the first call returns true
        if is_room_solvable(obstacle_matrix, max_room_width, max_room_height, i, x + 1, y) or \
                is_room_solvable(obstacle_matrix, max_room_width, max_room_height, i, x - 1, y) or \
                is_room_solvable(obstacle_matrix, max_room_width, max_room_height, i, x, y + 1) or \
                is_room_solvable(obstacle_matrix, max_room_width, max_room_height, i, x, y - 1):
            # mark the coordinates as visited
            obstacle_matrix[y][x] = '.'

            # a solution has been found
            return True
    # if the coordinates are the end of the game then a solution has been found
    elif obstacle_matrix[max_room_height - 2][max_room_width - 2] == '.':
        return True
    # if no if statements returned a value, then by default no solution has been found where (x, y) is in the
    # solution path.
    return False
