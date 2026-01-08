# Start file for different scripts

import scripts.convert as convert
import scripts.replace as replace
import scripts.randomize as randomize
import scripts.restore_backup as restore_backup
import scripts.tree as tree
import scripts.ics as ics
import scripts.pdf_table as pdf_table

version = "1.1.0"


def start():
    print(f"Starting script utilities v{version}.")

    print("\nBe aware, this may have a huge unrecoverable impact.")
    print("You should create manual backups.")

    print("\nWhat do you want to do?"
          "\n 0 --- Randomize lines"
          "\n 1 --- Convert xlsx files to csv files"
          "\n 2 --- Get a list of a file ending and export it in to a separate file"
          "\n 3 --- Search and replace in file with same file ending, takes some time"
          "\n 4 --- Export .ics events to an excel file"
          "\n 5 --- Export PDF tables to an csv file"
          "\n 6 --- Restore automatic backup"
          "\n 7 --- Exit")
    choice = input("Select an action to do: ")

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
        ics.export()
    if choice == '5':
        pdf_table.createTable()
    if choice == '6':
        print("\nName the file end to restore e.g. '.csv'")
        restore_backup.do_restore_backup(tree.get_file_end())
    if choice == '7':
        print('\nClosing.')
        exit()
    start()


start()
