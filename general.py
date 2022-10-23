import copy
import os
import shutil
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from env_config import WIPE_PROCESSED


def get_next_weekday(date: datetime) -> datetime:
    temp_date = copy.deepcopy(date)
    if temp_date.weekday() == 5 or temp_date.weekday() == 6:
        temp_date += relativedelta(days=(7 - temp_date.weekday()))
    return temp_date


def wipe_directory(folder_directory: str) -> None:
    for filename in os.listdir(folder_directory):
        file_path = os.path.join(folder_directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def join_csv_files(folder_directory: str, output_filename: str) -> None:
    merged_df = None
    for file in os.listdir(folder_directory):
        print("File being opened:", file)
        df = pd.read_csv(folder_directory + "/" + file)
        df = df.iloc[:, 1:]
        # print(df)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df])
    merged_df.to_csv(output_filename + ".csv")
    if WIPE_PROCESSED:
        wipe_directory(folder_directory)
