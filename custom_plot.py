import sys
import os

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import mne

import json


class label_tool(qtw.QWidget):

    def __init__(self):
        super(label_tool, self).__init__(flags=qtc.Qt.Window)

        # user login
        self.is_login = False

        # add data and label file
        self.data_directory = None
        self.label_file = 'label.json'

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

        self.setLayout(self.layout)
        self.setStyleSheet('''
                QTabWidget::tab-bar {
                    alignment: center;
                }''')

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
        self.label_combo.addItems(["Not select", "Normal", "Poor Contact", "NA"])

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
            print("Please select a label")
            return

        # disable submit button
        self.submit_bt.setEnabled(False)
        self.label_combo.setEnabled(False)

        # get the selected label2
        self.label = self.label_combo.currentText()
        print(self.label)

        # write the epoch_index and label to json file
        with open(self.label_file, 'r') as f:
            self.label_dict = json.load(f)

        # change label dict from the user
        if self.user_name not in self.label_dict:
            self.label_dict[self.user_name] = {}

        # add label to the dict
        self.label_dict[self.user_name][self.epoch_index] = self.label

        # record to json file
        with open(self.label_file, 'w') as f:
            json.dump(self.label_dict, f)

        # enable previous and next button
        self.previous_bt.setEnabled(True)
        self.next_bt.setEnabled(True)

    def _previous_bt_clicked(self):
        # back to previous epoch
        # enable submit button
        self.submit_bt.setEnabled(True)
        self.label_combo.setEnabled(True)
        # load next epoch file
        # open the next plot
        self.data = mne.read_epochs(os.path.join(self.data_directory, self.epoch_list[0]))
        self.epoch_index = self.epoch_list[0].split("-")[0]
        # change the plot widget
        for i in reversed(range(self.epoch_plot_layout.count())):
            self.epoch_plot_layout.itemAt(i).widget().setParent(None)

        self.data_plot_widget = self.data.plot(block=False, show=False, theme="light")
        self.epoch_plot_layout.addWidget(self.data_plot_widget, 0, 0)
        self.epoch_plot_widget.show()

        self.setLayout(self.layout)
        self.setStyleSheet('''
                        QTabWidget::tab-bar {
                            alignment: center;
                        }''')

    def _next_bt_clicked(self):
        # load next epoch file
        # enable submit button
        self.submit_bt.setEnabled(True)
        self.label_combo.setEnabled(True)
        # load next epoch file
        # open the next plot
        self.data = mne.read_epochs(os.path.join(self.data_directory, self.epoch_list[1]))
        self.epoch_index = self.epoch_list[1].split("-")[0]
        # change the plot widget
        for i in reversed(range(self.epoch_plot_layout.count())):
            self.epoch_plot_layout.itemAt(i).widget().setParent(None)

        self.data_plot_widget = self.data.plot(block=False, show=False, theme="light")
        self.epoch_plot_layout.addWidget(self.data_plot_widget, 0, 0)
        self.epoch_plot_widget.show()

        self.setLayout(self.layout)
        self.setStyleSheet('''
                QTabWidget::tab-bar {
                    alignment: center;
                }''')

    def _import_data_folder(self):
        # todo: change to connect to server later
        # get data folder
        self.data_directory = qtw.QFileDialog.getExistingDirectory(self, 'Select data folder', '')
        self.import_folder_confirm_bt.setEnabled(True)
        print(self.data_directory)

    def _confirm_import_data_folder(self):
        # todo: change to connect to server later
        # confirm data folder
        if self.data_directory.endswith("data"):
            # import_folder_bt turn to green
            self.import_folder_bt.setStyleSheet("background-color: green")
            self.import_folder_confirm_bt.setEnabled(False)

            # get list of epoch files in the data folder
            self.epoch_list = [item for item in os.listdir(self.data_directory) if item.endswith("-epo.fif")]

            # todo: randomize the epoch list and split into train and test

            # label_bt enable
            self.submit_bt.setEnabled(True)
            self.label_combo.setEnabled(True)

            # load the epoch file
            self.data = mne.read_epochs(os.path.join(self.data_directory, self.epoch_list[0]))
            self.epoch_index = self.epoch_list[0].split("-")[0]

            # change the plot widget
            self.data_plot_widget = self.data.plot(block=False, show=False, theme="light")
            self.epoch_plot_layout.addWidget(self.data_plot_widget, 0, 0)
            self.epoch_plot_widget.show()

            # create label file inside current folder
            self.label_file = "label.json".format(self.user_name)
            if not os.path.exists(self.label_file):
                with open(self.label_file, 'w') as f:
                    json.dump({}, f)
            else:
                with open(self.label_file, 'r') as f:
                    self.label_dict = json.load(f)

        else:
            self.import_folder_bt.setStyleSheet("background-color: red")
            self.import_folder_confirm_bt.setEnabled(True)

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
