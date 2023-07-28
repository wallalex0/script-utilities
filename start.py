# Start file for different scripts

import scripts.convert as convert
import scripts.replace as replace
import scripts.randomize as randomize
import scripts.restore_backup as restore_backup
import scripts.tree as tree

version = "1.0.0"


def start():
    print("\nWhat do you want to do?"
          "\n 0 --- Randomize lines"
          "\n 1 --- Convert xlsx files to csv files"
          "\n 2 --- Get a list of a file ending and export it in to a separate file"
          "\n 3 --- Search and replace in file with same file ending, takes some time"
          "\n 4 --- Restore automatic backup"
          "\n 5 --- Exit"
          "\n Be aware, this may have a huge unrecoverable impact."
          "\n You should create manual backups.")
    choice = input()

    if choice == '0':
        randomize.randomize()
    if choice == '1':
        convert.convert()
    if choice == '2':
        file_end = tree.get_file_end()
        tree.get_file_list(file_end, True)
        tree.do_tree_file(tree.get_tree_name(), file_end)
    if choice == '3':
        replace.get_search_replace()
        replace.do_replace()
    if choice == '4':
        print("\nName the file end to restore e. g. '.csv'")
        restore_backup.do_restore_backup(tree.get_file_end())
    if choice == '5':
        print('\nClosing.')
        exit()
    start()


start()
