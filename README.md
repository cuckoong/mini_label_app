# mini_label_app
mini label app for EEG data

# 1. run sample_data.py to get Processed_data/repaired_raw, which contains repaired raw data, and log the repaired data 
# into Processed_data/metadata/repaired_data.json ; 
# and get the epochs data with 1,2,4s duration and save each epoch into Processed_data/epochs folder, and log the infor into 
# Processed_data/metadata/epochs_info.json;

# 2. run sample_data.py to select epochs that are from good quality files (signal quality > 3), and stratify sampling 
# to get 20% from each condition(subject x label x duration) and save them into Processed_data/epochs_to_label folder.

# 3. run label_app.py to label the epochs in Processed_data/epochs_to_label folder, and save the labeled data 
