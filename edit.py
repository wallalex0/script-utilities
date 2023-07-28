# Edit found files

import tree
import fileinput

search = []
replace = []


def get_search_replace():
    another_one = True
    while another_one is True:
        print(' \nText to search:')
        search.append(input())

        print('\nText to Replace: ("$filename" as a variable for the file name)')
        replace.append(input())

        print('\nReplace another one? (Yes / No)')
        another = input()
        if another == 'No' or another == 'N' or another == 'n':
            another_one = False


def do_replace():
    for tree_file in tree.get_file_list(tree.get_file_end(), True):
        try:
            i = 0
            while i < len(search):
                with fileinput.FileInput(tree_file, inplace=True, backup='.bak') as file:
                    for line in file:
                        if replace[i] == '$filename':
                            split1 = tree_file.split('\\')
                            n = len(split1) - 1
                            split2 = split1[n].split('.')
                            replace[i] = split2[0]
                            print(line.replace(search[i], replace[i]), end='')
                        else:
                            print(line.replace(search[i], replace[i]), end='')
                i = i + 1
        except Exception as e:
            print('Error: ' + str(e))

    print('\nSuccessful.')
