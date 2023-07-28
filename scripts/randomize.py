# Randomize lines

import random


def randomize():
    print(' \nPath to file to randomize:')
    path = input()
    with open(path) as file:
        lines = file.readlines()
        for element in lines:
            print(element)

        print("---")

        random.shuffle(lines)
        for element in lines:
            print(element)
