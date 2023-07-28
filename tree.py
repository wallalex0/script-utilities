# Search for files

import os
from os import walk

root_path = os.path.dirname(os.path.abspath(__file__)) + '\\put files here'
print('\nRootpath: \n' + root_path)


def get_file_end():
    print(' \nEnding to filter:')
    return input()


def get_tree_name():
    print('\nName the tree file:')
    return input()


def get_file_list(file_end, print_list):
    path_list = []
    file = []
    path_file_list = []

    for (dir_path, dir_names, file_names) in walk(root_path):
        for filename in file_names:
            if filename.endswith(file_end):
                path_list.append(dir_path)
                file.append(filename)

    if print_list is True:
        print('\nFiles found:')
    i = 0
    while i < len(file):
        path_file_list.append(path_list[i] + '\\' + file[i])
        if print_list is True:
            print(path_file_list[i])
        i = i + 1
    return path_file_list


def do_tree_file(tree_name, file_end):
    tree = open('tree_' + tree_name + '.txt', 'w+')
    file_list = get_file_list(file_end, False)
    i = 0
    while i < len(file_list):
        tree.write(file_list[i] + '\n')
        i = i + 1
    tree.close()
