# Start file for search and replace and restore

import edit
import restore_backup
import tree


def start():
    print("\nWhat do you want to do?"
          "\n 1 --- Get a list of a file ending and export it in to a separate file"
          "\n 2 --- Search and replace in file with same file ending, takes some time"
          "\n 3 --- Restore Backup"
          "\n 4 --- Exit"
          "\n Be aware, this may have a huge unrecoverable impact."
          "\n You should create a manual backup.")
    choice = input()

    if choice == '1':
        file_end = tree.get_file_end()
        tree.get_file_list(file_end, True)
        tree.do_tree_file(tree.get_tree_name(), file_end)
    if choice == '2':
        edit.get_search_replace()
        edit.do_replace()
    if choice == '3':
        print("\nName the file end to restore e. g. '.csv'")
        restore_backup.do_restore_backup(tree.get_file_end())
    if choice == '4':
        print('\nExiting.')
        exit()
    start()


start()
