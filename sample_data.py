import pandas as pd

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





