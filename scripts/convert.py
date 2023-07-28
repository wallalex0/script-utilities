import pandas
import os
import warnings


def get_from_dir():
    print(' \nRoot folder:')
    return input()


def get_to_dir():
    print('\nFolder to paste converted files:')
    return input() + "\\converted"


def convert():

    from_dir = get_from_dir()

    to_dir = get_to_dir()
    try:
        os.mkdir(to_dir)
    except FileExistsError:
        pass

    print("\nStarting conversion.")

    i = 1
    for root, dirs, files in os.walk(from_dir):
        for file_name in files:
            if file_name.endswith(".xlsx"):
                file_name = file_name[:-5]

                print(f"Processing file {i}...")

                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")

                    file = pandas.read_excel(f"{root}\\{file_name}.xlsx", engine="openpyxl")

                    file.to_csv(f"{to_dir}\\file{str(i)}.csv")
                    i += 1

                    # # read csv file and convert
                    # # into a dataframe object
                    # df = pd.DataFrame(pd.read_csv("OpenDocument Tabellendokument (neu).csv"))
                    #
                    # # show the dataframe
                    # print(df)

    print("\nFinished converting.")
