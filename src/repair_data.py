import os
import shutil
import warnings
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import json
import uuid


class MA():
    unit_scale = {
        'no': 1,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
    }

    def __init__(self, signal_start_timestamp, df_ma, voltage_scale='u', sfreq=500.5, file_name=None,
                 ch_types=['eeg', 'eeg'],
                 ch_names=['ch1', 'ch2']):
        """
        Params
        --------------------
        df_ma: pandas.DataFrame
            The column contains the package number in the first column
            while the remaining ones contain the data payload
            from the two channels alternatively
        voltage_scale: string
        sfreq: float
            sampling frequency in Hz
        ch_types: a list of string
            the types of signal of the channels
        ch_names: a list of string
            the channel names

        Returns
        --------------------
        """
        self.signal_start_timestamp = signal_start_timestamp
        self.df_ma = df_ma
        self.voltage_scale = voltage_scale
        self.sfreq = sfreq
        self.file_name = file_name
        self.ch_types = ch_types
        self.ch_names = ch_names

        # get the signal data
        self.signal_raw = self.df_ma.copy(deep=True)

        # separate the package number

        # +----------------+-----------+-----------+-----------+-----------+-----+
        # | package number | channel 1 | channel 2 | channel 1 | channel 2 | ... |
        # +----------------+-----------+-----------+-----------+-----------+-----+
        self.signal_raw = self.signal_raw.drop(axis=1, columns=0).values.reshape(-1, 2) * self.unit_scale[
            self.voltage_scale]
        self.packages = self.df_ma.copy(deep=True)[0].values

        self.check_package_sequence()

    def check_package_sequence(self):
        # Check package sequence of package

        package_diff = np.diff(self.packages)
        if max(package_diff) > 1:
            print('Missing package detected')

    def _check_raw_attri(self):

        if not hasattr(self, 'raw'):
            raise Exception('The MA class does not have the "raw" attribute yet.')

    @classmethod
    def load_file(cls, file_path_ma, voltage_scale='u', sfreq=500.5):

        # get filename from path
        file_name = file_path_ma.split('/')[-1].split('.')[0]

        with open(file_path_ma, "r") as f:
            dataline = f.readline()

        signal_start_timestamp = int(dataline)
        df_ma = pd.read_csv(file_path_ma, sep=',', header=None, skiprows=1)

        return cls(signal_start_timestamp, df_ma, voltage_scale, sfreq, file_name)

    def to_raw(self, resample_freq=None):

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types)
        self.raw = mne.io.RawArray(self.signal_raw.T, info, verbose=False)

        # log the filename to raw
        self.raw.info['subject_info'] = {'filename': self.file_name, 'start_timestamp': self.signal_start_timestamp}

        if resample_freq is not None:
            self.raw = self.raw.resample(sfreq=resample_freq).copy()
        # return self.raw

    def set_anno(self, anno_file=None, task_type=None):

        self._check_raw_attri()

        if anno_file is None:
            raise Exception('The annotation file is not defined.')

        if task_type is None:
            raise Exception('The task type is not defined.')

        if task_type == 'eye_closed':
            df_anno = pd.read_csv(anno_file)
            start = (df_anno["Start"].values[0] - self.signal_start_timestamp) / 1000
            end = (df_anno["End"].values[0] - self.signal_start_timestamp) / 1000

        elif task_type == 'eye_open':
            df_anno = pd.read_csv(anno_file)
            start = (df_anno["start_time_stamp"].values[0] - self.signal_start_timestamp) / 1000 + \
                    df_anno['eyeopen_start'].values[1]

            end = (df_anno["start_time_stamp"].values[0] - self.signal_start_timestamp) / 1000 + \
                  df_anno['eyeopen_end'].values[1]

        else:
            raise Exception('The selected task type is not included.')

        # test if the length of raw data is longer than the annotation duration
        if (len(self.raw) / self.sfreq) < end:
            warnings.warn("The length of raw data is shorter than the annotation duration.")
            warnings.warn('setting the end of the annotation to the end of the raw data.')
            end = len(self.raw) / self.sfreq

        # test if start is negative (psychopy protocols start before recording)
        if start < 0:
            warnings.warn("psychopy protocols start {} seconds before recording.".format(start))
            warnings.warn('set the start to 0')
            start = 0

        annotations = mne.Annotations(onset=start, duration=end - start, description=task_type)
        self.raw.set_annotations(annotations)
        self.raw.crop(tmin=start, tmax=end)
        # return self.raw

    def to_fif(self, save_path, file_name, file_type='eeg', compress=False, overwrite=False):

        self._check_raw_attri()

        suffix = {
            'eeg': '_eeg.fif',
        }

        if file_type in ['eeg']:
            file_name = file_name + suffix[file_type]
        else:
            raise Exception('File type not supported')

        if compress:
            file_name = file_name + '.gz'

        save_path = os.path.join(save_path, file_name)

        self.raw.save(fname=save_path, overwrite=overwrite)

    def get_raw(self):
        return self.raw


def repairedData(raw, threshold=100, is_plot=False, repaired_data_file=None):
    """
    Repair the data, remove the deviate points from online data.
    param threshold: how many std away from the mean
    param data: online data, shape: (n_samples,)
    return: repaired data, shape: (n_samples,)
    """
    data = raw.copy().get_data().reshape(-1)

    # for every 100 datapoint in data
    repaired_data = []
    n_interval = 1000
    for i in range(0, len(data), n_interval):
        T = data[i:i + n_interval].copy()
        Tdiff = np.diff(T)
        # get baseline values
        baseline = np.sort(Tdiff)[1:-1]
        standard = threshold * np.std(baseline)
        # find the deviate peak
        peak = np.abs(Tdiff - np.mean(baseline)) > standard
        # only find the one-point deviate peak, ignore other artifcats

        if (np.sum(peak) == 1) or (np.sum(peak) == 2):
            idx = np.where(peak == 1)[0][-1]
            print("Repaired Deviate data")
            # record the name of the file
            print(raw.info['subject_info']['filename'])

            # log into json file
            with open(repaired_data_file, 'r') as f:
                repaired_data_log = json.load(f)

            repaired_data_log[raw.info['subject_info']['filename']] = True
            with open(repaired_data_file, 'w') as f:
                json.dump(repaired_data_log, f)

            if idx != 0:
                T[idx] = T[idx - 1].copy()
            else:
                T[0] = T[1].copy()
        repaired_data.extend(T)

    repaired_data = np.array(repaired_data).reshape(1, -1)

    # check if the length of repaired data is the same as the original data
    assert repaired_data.shape[1] == len(data)

    if is_plot:
        plt.figure()
        plt.plot(repaired_data.reshape(-1), label="repaired data")
        plt.plot(data, label="original data")
        plt.legend()
        plt.show()

    repaired_raw = raw.copy()
    repaired_raw._data = repaired_data
    return repaired_raw


def load_raw(dir_ma):
    """
    repair the data with the repairedData function,  check and log if the data is repaired.
    :param dir_ma: .ma file directory
    :return:
    """
    # read eeg file
    ma = MA.load_file(dir_ma)
    ma.to_raw()
    raw = ma.get_raw().pick_channels(['ch1'])
    return raw


def get_raw_stamps(dir_raw, dir_anno, start_timestamp):
    # read raw data from fif file
    raw = mne.io.read_raw_fif(dir_raw, preload=True)
    # read annotation from csv file
    df_anno = pd.read_csv(dir_anno)
    raw_stamps = []

    for idx, row in df_anno.iterrows():
        start = (df_anno.loc[idx, 'Start'] - start_timestamp) / 1000
        end = (df_anno.loc[idx, 'End'] - start_timestamp) / 1000
        raw_stamps.append(raw.copy().crop(tmin=start, tmax=end))

    assert len(raw_stamps) != 0

    return raw_stamps


def log_metadata(json_file, record_df, filename, duration):
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            json.dump({}, f)

    # log the information
    with open(json_file, 'r') as f:
        data_log = json.load(f)

    data_log[epoch_index] = {}
    data_log[epoch_index]['filename'] = filename
    data_log[epoch_index]['epoch_duration'] = duration
    data_log[epoch_index]['label'] = filename.split('_')[0]
    data_log[epoch_index]['subject'] = record_df[record_df['file_name'] == filename]['subject'].values[0]

    with open(json_file, 'w') as f:
        json.dump(data_log, f)


if __name__ == '__main__':

    tasks = ['poor_contact', 'eog', 'normal']
    epoch_durations = [1, 2, 4]
    wkdir = '/Volumes/mindampshared/quality_assessment/sample_data/raw_data/'
    processed_wkdir = '/Users/panpan/PycharmProjects/label_gui/Processed_data/'

    # read the repaired data record
    record_dataframe = pd.read_csv('/Users/panpan/PycharmProjects/label_gui/data/sample_data_record.csv')

    # create a json file to log the repaired data
    repaired_data_file = os.path.join(processed_wkdir, 'metadata', 'repaired_data.json')

    #
    if os.path.exists(repaired_data_file):
        # delete the file
        os.remove(repaired_data_file)
    with open(repaired_data_file, 'w') as f:
        json.dump({}, f)

    # create a json file to log the repaired data
    repaired_data_log = {}
    json_file = os.path.join(processed_wkdir, 'metadata', 'epoch_info.json')
    with open(json_file, 'w') as f:
        json.dump(repaired_data_log, f)

    # create epoch folder for each task
    epoch_folder = os.path.join(processed_wkdir, 'epochs')
    if os.path.exists(epoch_folder):
        # remove the folder if it exists
        shutil.rmtree(epoch_folder)
    os.mkdir(epoch_folder)

    for task in tasks:
        # get eeg file path
        dir_ma = os.path.join(wkdir, task)
        eeg_files = os.listdir(dir_ma)
        eeg_files = [os.path.join(dir_ma, f) for f in eeg_files if f.endswith('.ma')]

        for file in eeg_files:
            # repair the data
            raw = load_raw(file)

            # repair the data
            repaired_raw = repairedData(raw, is_plot=True, repaired_data_file=repaired_data_file)

            # save the repaired data
            filename = file.split('/')[-1].split('.')[0]
            save_path = os.path.join(processed_wkdir, 'repaired_raw', filename + '_raw.fif')
            repaired_raw.save(save_path, overwrite=True, verbose=True)

            # crop raw data with time stamps from csv files
            csv_file = file.replace('.ma', '.csv')
            start_time = repaired_raw.info['subject_info']['start_timestamp']
            raw_stamps_list = get_raw_stamps(save_path, csv_file, start_time)

            if len(raw_stamps_list) == 0:
                print("No annotation in csv file")

            else:
                for duration in epoch_durations:
                    # crop the raw data into epochs with different fixed length
                    for raw in raw_stamps_list:
                        try:
                            epochs = mne.make_fixed_length_epochs(raw.copy(), duration=duration, preload=True)
                        except ValueError:
                            print("Epochs are too short")
                            continue

                        for i in range(len(epochs)):
                            epoch_index = uuid.uuid4().hex
                            # save epoch data
                            epoch_file_path = os.path.join(processed_wkdir, 'epochs', '{}-epo.fif'.format(epoch_index))
                            epochs[i].save(epoch_file_path, overwrite=True)

                            # log the metadata
                            log_metadata(json_file, record_dataframe, filename, duration)
