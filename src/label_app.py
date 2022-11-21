import sys
import os
import time

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import pyqtgraph as pg

import mne
import numpy as np
import random
from scipy import signal, fft

import json

os.chdir(os.path.dirname(os.getcwd()))


def get_psd(data, sfreq=500.5):
    """
    :param sfreq: default 500.5
    :param data: data (type: deque after filtering, unit: microV), use last 5s data for PSD
    :return: power spectral density in microV^2/Hz
    """
    data = np.array(data).reshape(-1)

    # hanning window
    window = signal.windows.hann(len(data))
    data = window * data

    # fft
    fft_data = fft.fft(data)

    # power spectral density
    psd_data = np.abs(fft_data) ** 2 / len(data) / sfreq
    psd_data = 10 * np.log10(psd_data[0:int(len(psd_data) / 2)])
    return psd_data


class label_tool(qtw.QWidget):

    def __init__(self):
        super(label_tool, self).__init__(flags=qtc.Qt.Window)

        # user login
        self.is_login = False

        # add data and label file
        self.data_directory = None
        self.label_file = 'label_results/label.json'

        # connect to label dataset
        self._connect_to_label_dataset()

        # add plot to the main window
        self.layout = qtw.QGridLayout()

        # data popup window
        self._make_epoch_vis_widget()

        # add user login widget
        self.layout.addWidget(self._user_login_widget(), 0, 0, 1, 2)

        # add import folder button
        self.layout.addWidget(self._connect_data_folder_widget(), 2, 0, 1, 2)

        # add button to the main window for label selection
        self.layout.addWidget(self._choose_label_widget(), 3, 0, 1, 2)

        # info widget
        self.layout.addWidget(self._info_widget(), 4, 0, 1, 2)

        self.setLayout(self.layout)
        self.setStyleSheet('''
                QTabWidget::tab-bar {
                    alignment: center;
                }''')

    def _info_widget(self):
        self.info = qtw.QLabel()
        self.info.setText("Info: ")
        return self.info

    def _connect_to_label_dataset(self):
        # connect to the label dataset
        # if file not exist, create a new one
        if not os.path.exists(self.label_file):
            self.label = {}
            with open(self.label_file, 'w') as f:
                json.dump(self.label, f)

    def _make_epoch_vis_widget(self):
        self.epoch_plot_widget = qtw.QWidget()
        self.epoch_plot_layout = qtw.QGridLayout()
        self.epoch_plot_widget.setLayout(self.epoch_plot_layout)

    def _connect_data_folder_widget(self):
        # todo: change to connect to server later
        # create a widget for import folder
        self.import_folder_widget = qtw.QWidget()
        self.import_folder_layout = qtw.QGridLayout()
        self.import_folder_widget.setLayout(self.import_folder_layout)

        # create import folder button
        self.import_folder_bt = qtw.QPushButton("Connect\nDataset")
        self.import_folder_bt.setEnabled(False)
        self.import_folder_bt.clicked.connect(self._import_data_folder)

        # todo: remove confirm button later
        # confirm import folder button
        self.import_folder_confirm_bt = qtw.QPushButton("Confirm")
        self.import_folder_confirm_bt.setEnabled(False)
        self.import_folder_confirm_bt.clicked.connect(self._confirm_import_data_folder)

        # add button to the widget
        self.import_folder_layout.addWidget(self.import_folder_bt, 0, 0, 1, 1)
        self.import_folder_layout.addWidget(self.import_folder_confirm_bt, 0, 1, 1, 1)

        return self.import_folder_widget

    def _user_login_widget(self):
        # create a widget for user login
        self.user_login_widget = qtw.QWidget()
        self.user_login_layout = qtw.QGridLayout()

        # create user login button
        self.user_login_bt = qtw.QPushButton("Login")
        self.user_login_bt.setEnabled(True)
        self.user_login_bt.clicked.connect(self._user_login_bt_clicked)

        # create user selection combo
        self.user_combo = qtw.QComboBox()
        self.user_combo.addItems(["Select User", "Lester", "Arthit", "Panda"])

        # add button and combo to the widget
        self.user_login_layout.addWidget(self.user_combo, 0, 0, 1, 1)
        self.user_login_layout.addWidget(self.user_login_bt, 0, 1, 1, 1)

        self.user_login_widget.setLayout(self.user_login_layout)
        return self.user_login_widget

    def _user_login_bt_clicked(self):
        if self.user_combo.currentText() == "Select User":
            print("Please select user")
            self.info.setText("Please select user")
            return
        else:
            self.info.setText("User: " + self.user_combo.currentText())

        # get user name
        if not self.is_login:
            self.is_login = True
            # get user name
            self.user_name = self.user_combo.currentText()
            # enable import folder widget
            self.import_folder_bt.setEnabled(True)
            self.import_folder_confirm_bt.setEnabled(True)
            # hide user names
            self.user_combo.setEnabled(False)
            # changed to logout
            self.user_login_bt.setText("Logout")
        else:
            self.is_login = False
            # disable import folder button
            # remove import folder
            self.data_directory = None
            self.import_folder_bt.setStyleSheet("background-color: white")
            self.import_folder_bt.setEnabled(False)
            self.import_folder_confirm_bt.setEnabled(False)
            # show user names
            self.user_combo.setEnabled(True)
            # changed to login
            self.user_login_bt.setText("Login")

    def _choose_label_widget(self):
        # create a widget for label selection
        self.label_widget = qtw.QWidget()
        self.label_layout = qtw.QGridLayout()
        self.label_widget.setLayout(self.label_layout)

        # create label selection button
        self.submit_bt = qtw.QPushButton("Submit")
        self.submit_bt.setEnabled(False)
        self.submit_bt.clicked.connect(self._submit_label)

        # create label selection combo
        self.label_combo = qtw.QComboBox()
        self.label_combo.setEnabled(False)
        self.label_combo.addItems(["Not select", "normal", "poor", "eog", "na"])

        # previous button
        self.previous_bt = qtw.QPushButton("Previous")
        self.previous_bt.setEnabled(False)
        self.previous_bt.clicked.connect(self._previous_bt_clicked)

        # next button
        self.next_bt = qtw.QPushButton("Next")
        self.next_bt.setEnabled(False)
        self.next_bt.clicked.connect(self._next_bt_clicked)

        # add button and combo to the widget
        self.label_layout.addWidget(self.label_combo, 0, 0, 1, 1)
        self.label_layout.addWidget(self.submit_bt, 0, 1, 1, 1)
        self.label_layout.addWidget(self.previous_bt, 1, 0, 1, 1)
        self.label_layout.addWidget(self.next_bt, 1, 1, 1, 1)

        return self.label_widget

    def _submit_label(self):
        if self.label_combo.currentText() == "Not select":
            self.info.setText("Please select label")
            return
        else:
            self.info.setText("Label: " + self.label_combo.currentText() +
                              '\n' +
                              'Remaining: ' + str(self.epoch_number - self.epoch_index - 1))

        # disable submit button
        self.submit_bt.setEnabled(False)
        self.label_combo.setEnabled(False)

        self.previous_bt.setEnabled(True)
        self.next_bt.setEnabled(True)

        # get the selected label2
        selected_label = self.label_combo.currentText()
        print(selected_label)

        # save the label to the label file
        self._add_label_record(selected_label)

        # enable previous and next button
        if self.epoch_index < self.epoch_number - 1:
            self.next_bt.setEnabled(True)
        if self.epoch_index > 0:
            self.previous_bt.setEnabled(True)

    def _add_label_record(self, label):
        # write the epoch_index and label to json file
        with open(self.label_file, 'r') as f:
            self.label_dict = json.load(f)

        # add label to the dict
        file_id = self.epoch_list[self.epoch_index].split('_')[0]

        if self.user_name not in self.label_dict:
            self.label_dict[self.user_name] = {}

        if file_id not in self.label_dict[self.user_name]:
            self.label_dict[self.user_name][file_id] = {}

        self.label_dict[self.user_name][file_id]['id'] = file_id
        self.label_dict[self.user_name][file_id]['user'] = self.user_name
        self.label_dict[self.user_name][file_id]["label"] = label
        self.label_dict[self.user_name][file_id]["time"] = int(time.time())

        # record to json file
        with open(self.label_file, 'w') as f:
            json.dump(self.label_dict, f, indent=4)

    def _clear_plot_widget(self):
        # clear the plot widget
        for i in reversed(range(self.epoch_plot_layout.count())):
            self.epoch_plot_layout.itemAt(i).widget().setParent(None)

    def _set_new_plot_widget(self, epoch_index):
        self.epoch_index = epoch_index

        # load next epoch file
        raw_data = mne.read_epochs(os.path.join(self.data_directory, self.epoch_list[epoch_index]), preload=True)

        # add new plot widget
        self.data_plot_widget = raw_data.plot(block=False, show=False, theme="light")

        # add new plot widget to the layout
        filtered_data = raw_data.copy().filter(l_freq=4, h_freq=45, method="iir")
        self.filtered_data_plot_widget = filtered_data.plot(block=False, show=False, theme="light")

        # add new psd widget
        self.psd_plot_widget = pg.PlotWidget()
        self.psd_plot_widget.setBackground("w")
        pen = pg.mkPen(color=(0, 0, 0), width=3, style=qtc.Qt.SolidLine)
        self.psd_plot = self.psd_plot_widget.plot(pen=pen)
        self.psd_plot_widget.setXRange(4, 45)

        psd_data, freqs = mne.time_frequency.psd_welch(filtered_data, fmin=4, fmax=45, n_fft=len(filtered_data.times))
        self.psd_plot.setData(x=freqs, y=psd_data[0, 0, :])

        # group three plot widgets
        # add text
        self.epoch_plot_layout.addWidget(qtw.QLabel("Raw Data"), 0, 0, 1, 1)
        self.epoch_plot_layout.addWidget(self.data_plot_widget, 1, 0)

        self.epoch_plot_layout.addWidget(qtw.QLabel("Filtered Data"), 2, 0, 1, 1)
        self.epoch_plot_layout.addWidget(self.filtered_data_plot_widget, 3, 0)

        self.epoch_plot_layout.addWidget(qtw.QLabel("PSD"), 4, 0, 1, 1)
        self.epoch_plot_layout.addWidget(self.psd_plot_widget, 5, 0)

    def _previous_bt_clicked(self):
        # back to previous epoch
        # enable submit button
        self.submit_bt.setEnabled(True)
        self.label_combo.setEnabled(True)

        # change the plot widget
        self._clear_plot_widget()

        self.previous_bt.setEnabled(False)
        self.next_bt.setEnabled(False)

        # add new plot widget
        if self.epoch_index > 0:
            self._set_new_plot_widget(self.epoch_index - 1)
        else:
            return
        # show the epoch plots
        self.epoch_plot_widget.show()

    def _next_bt_clicked(self):
        # load next epoch file
        # enable submit button
        self.submit_bt.setEnabled(True)
        self.label_combo.setEnabled(True)

        # disable previous and next
        self.previous_bt.setEnabled(False)
        self.next_bt.setEnabled(False)

        # change the plot widget
        self._clear_plot_widget()

        # add new plot widget
        if self.epoch_index < self.epoch_number - 1:
            self._set_new_plot_widget(self.epoch_index + 1)
        else:
            self.info.setText("All epochs have been labeled\n"
                              "Please select next folder")
            self.import_folder_bt.setEnabled(True)

        # show the epoch plots
        self.epoch_plot_widget.show()

    def _import_data_folder(self):
        # get data folder
        self.data_directory = qtw.QFileDialog.getExistingDirectory(self, 'Select data folder',
                                                                   'Processed_data/epochs_to_label')
        self.import_folder_confirm_bt.setEnabled(True)
        print(self.data_directory)

    def _confirm_import_data_folder(self):
        # import_folder_bt turn to green
        self.import_folder_bt.setStyleSheet("background-color: green")
        self.import_folder_confirm_bt.setEnabled(False)

        # get list of epoch files in the data folder
        self.epoch_list = [item for item in os.listdir(self.data_directory) if item.endswith("-epo.fif")]
        self.epoch_number = len(self.epoch_list)
        # label_bt enable
        self.submit_bt.setEnabled(True)
        self.label_combo.setEnabled(True)

        # change the plot widget
        self._clear_plot_widget()

        # load the epoch file
        self._set_new_plot_widget(0)

        self.epoch_plot_widget.show()

        # get label record files
        self._label_records()

    def _label_records(self):
        # create label file inside current folder
        if not os.path.exists(self.label_file):
            with open(self.label_file, 'w') as f:
                json.dump({}, f)

        with open(self.label_file, 'r') as f:
            self.label_dict = json.load(f)

    def closeEvent(self, event):
        reply = qtw.QMessageBox.question(self, 'Message', 'Are you sure to close this window?',
                                         qtw.QMessageBox.Yes | qtw.QMessageBox.No, qtw.QMessageBox.No)
        if reply == qtw.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    main = label_tool()
    main.show()
    sys.exit(app.exec_())
