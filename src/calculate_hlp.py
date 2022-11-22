import os
import json

import pandas as pd

os.chdir(os.path.dirname(os.getcwd()))


def calculate_hlp():
    file = 'label_results/label.json'
    df = pd.read_json(file, orient='records').reset_index()

    # replace index column name to 'id'
    df.drop('index', axis=1, inplace=True)

    df_list = []
    for column in df.columns:
        tmp = df[column].apply(pd.Series)
        df_list.append(tmp)

    # merge df_list into one df
    result_df = pd.concat(df_list, axis=0, ignore_index=True)

    # get ground truth label
    gt_df = pd.read_json('Processed_data/metadata/epoch_info.json', orient='index').reset_index()

    # merge ground truth label with result_df
    result_df = pd.merge(result_df, gt_df, left_on='id', right_on='index', how='left')

    # calculate hlp for each user

    result_df = result_df[result_df['label_y'] != 'eog']

    print('Accuracy ignoring EOG')
    print((result_df['label_x'] == result_df['label_y']).mean())

    # group by subject
    print('Accuracy (ignoring EOG) for each subject')
    print(result_df.groupby('subject').apply(lambda x: (x['label_x'] == x['label_y']).mean()))


if __name__ == '__main__':
    calculate_hlp()
