import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

def read_dataset(filepath):
    """Return the features, the labels, and the rooms from the dataset specified by the "filepath" """
    entries = []
    entry_labels = []
    # Go through each line on the file
    for line in open(filepath):
        # Check only non empty lines
        if line.strip() != "":
            row = line.strip().split()
            entries.append(list(map(float, row[:-1])))
            # Get the labels in float format
            entry_labels.append(float(row[-1]))

    # Get the unique labels in an array
    # and an list of indices showing each row's label
    # in terms of the array created
    [rooms, labels] = np.unique(entry_labels, return_inverse=True)
    entries = np.array(entries)
    labels = np.array(entry_labels)
    return (entries, labels, rooms)


def find_split(dataset, num_features, label_set):
    """Find the best split and return the best split with the maximum gain gained"""
    # Dictonary to store the best split
    best_split = {}

    max_info_gain = 0

    for feature_index in range(num_features):
        # Get the specific feature
        feature_values = dataset[:, feature_index]
        # Get the unique features
        # This is an alternative to sorting the feature values
        # and then only checking for each value change
        thresholds = np.unique(feature_values)

        # For each unique value in the feature
        for threshold in thresholds:
            # Get split datasets
            current_left, current_right = split(dataset, feature_index, threshold)
            # If both splits are of non-zero length
            if len(current_left) > 0 and len(current_right) > 0:
                left_labels = current_left[:, -1]
                right_labels = current_right[:, -1]
                dataset_labels = dataset[:, -1]

                # Get the information gain on the split
                current_info_gain = information_gain(dataset_labels, left_labels, right_labels, label_set)
                # If the gain is higher from the max then assign new max threshold
                if current_info_gain > max_info_gain:
                    max_info_gain = current_info_gain
                    best_split["best_feature_index"] = feature_index
                    best_split["best_threshold"] = threshold
                    best_split["best_left"] = current_left
                    best_split["best_right"] = current_right

    return (max_info_gain, best_split)


def split(dataset, index, threshold):
    """Splits dataset by feature at index given a treshold for the feature"""
    dataset_left = np.array([row for row in dataset if row[index] <= threshold])
    dataset_right = np.array([row for row in dataset if row[index] > threshold])

    return dataset_left, dataset_right



def information_gain(dataset, dataset_left, dataset_right, label_set):
    """Calculates information gain on certain split"""
    left_weight = len(dataset_left) / len(dataset)
    right_weight = len(dataset_right) / len(dataset)

    info_gain = entropy(dataset, label_set) \
        - left_weight * entropy(dataset_left, label_set) \
        - right_weight * entropy(dataset_right, label_set)

    return info_gain


def entropy(entry_labels, label_set):
    """Calculates entropy of the dataset """
    entropy = 0

    for label in label_set:
        probability_label = len(entry_labels[entry_labels == label]) / len(entry_labels)
        if probability_label != 0:
            entropy -= probability_label * np.log2(probability_label)

    return entropy

def k_fold_split(k, num_instances, random_generator = default_rng()):
    """Randomizes k folds and returns their indices"""
    shuffled_indices = random_generator.permutation(num_instances)
    split_indices = np.array_split(shuffled_indices, k)

    return split_indices

def train_test_k_fold(number_folds, num_instances, random_generator = default_rng()):
    """Creates k folds with random indices and produces k * (k-1) different
       triples of test set, validation set and train set indices"""

    # Get k random folds
    split_indices = k_fold_split(number_folds, num_instances, random_generator)

    # Go through each fold and use it as test set
    folds = []
    for k in range(number_folds):
        test_indices = split_indices[k]
        # For each test set, pick any another fold to be the validation set
        # and use the remaining k-2 folds for training
        for j in range(number_folds):
            if j == k:
                continue
            validation_indices = split_indices[j]
            if j < k:
                train_indices = np.hstack(split_indices[:j] + split_indices[j + 1:k] + split_indices[k+1:])
            else:
                train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:j] + split_indices[j+1:])
            folds.append([test_indices, validation_indices, train_indices])

    return folds
