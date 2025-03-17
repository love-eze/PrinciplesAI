import pandas as pd
import numpy as np

# --- 1. LOADING THE DATA ---
# Adjust the paths to match where you’ve unzipped "UCI HAR Dataset"

PATH = "path to /UCI HAR Dataset/"
features_path = PATH + "features.txt"
activity_labels_path = PATH + "activity_labels.txt"
X_train_path = PATH + "train/X_train.txt"
y_train_path = PATH + "train/y_train.txt"
X_test_path = PATH + "test/X_test.txt"
y_test_path = PATH + "test/y_test.txt"



# Load feature names, This appends the column index to any duplicate names.
features_df = pd.read_csv(features_path, sep="\s+", header=None, names=["idx", "feature"])
feature_names = features_df["feature"].tolist()


# his appends the column index to any duplicate names.
features_df["feature"] = features_df["feature"].astype(str) + "_" + features_df.index.astype(str)
feature_names = features_df["feature"].tolist()


# Load activity labels (mapping IDs 1-6 to string names)
activity_labels_df = pd.read_csv(activity_labels_path, sep="\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df["id"], activity_labels_df["activity"]))

# Load train/test sets
X_train = pd.read_csv(X_train_path, sep="\s+", header=None, names=feature_names)
y_train = pd.read_csv(y_train_path, sep="\s+", header=None, names=["Activity"])
X_test = pd.read_csv(X_test_path, sep="\s+", header=None, names=feature_names)
y_test = pd.read_csv(y_test_path, sep="\s+", header=None, names=["Activity"])

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# --- 2. CONVERT MULTI-CLASS TO BINARY ---
def to_binary_label(activity):
    if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        return 1  # Active
    else:
        return 0  # Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

