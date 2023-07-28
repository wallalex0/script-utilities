import os
import tree


def do_restore_backup(file_end):
    for bak_file in tree.get_file_list(file_end + '.bak', True):
        for old_file in tree.get_file_list(file_end, False):
            split = bak_file.split('.')
            if "." + split[len(split) - 2] == file_end:
                split.pop()
                new_file = '.'.join(split)
                if old_file == new_file:
                    os.remove(old_file)
                    os.rename(bak_file, new_file)
    print('\nSuccessful.')
