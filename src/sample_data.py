import os
import pandas as pd

os.chdir('/Users/panpan/PycharmProjects/label_gui')

if __name__ == '__main__':
    record_file = pd.read_csv('Data/sample_data_record.csv')

    # select signal quality larger than 3
    record_file = record_file[record_file['signal_quality'] > 3]

    # get label, filename and subject
    label = record_file['label']
    filename = record_file['file_name']
    subject = record_file['subject']

    # check duplicate rows
    assert (record_file.duplicated(subset=['file_name', 'subject']).any()) == False

    # good quality data filename
    good_quality_filename = filename.values

    # load metadata from epochs
    metadata = pd.read_json('Processed_data/metadata/epoch_info.json', orient='index').reset_index()

    # replace index column name with file_id
    metadata = metadata.rename(columns={'index': 'file_id'})

    # select good quality data
    metadata_good = metadata[metadata['filename'].isin(good_quality_filename)]

    # stratify sampling for each subject ane label and epoch length
    selected_file = metadata_good.groupby(['subject', 'label', 'epoch_duration']).\
        apply(lambda x: x.sample(frac=0.2, random_state=42))

    # save selected file to csv
    selected_file.to_csv('Processed_data/sampling_meta.csv', index=False)

    # copy selected file to new folder
    import shutil
    selected_file_id = selected_file['file_id'].values.tolist()

    # remove old folder
    if os.path.exists('Processed_data/epochs_to_label'):
        shutil.rmtree('Processed_data/epochs_to_label')
    os.mkdir('Processed_data/epochs_to_label')

    for file in selected_file_id:
        shutil.copy('Processed_data/epochs/' + file + '-epo.fif', 'Processed_data/epochs_to_label/' + file + '-epo.fif')

    # check file number
    assert(len(selected_file_id) == len(os.listdir('Processed_data/epochs_to_label')))

    print("Done")
