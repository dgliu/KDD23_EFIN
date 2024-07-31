import yaml
import random
import numpy as np
import pandas as pd


def criteo_statistics(path, name, sep, normalize):
    raw_data = pd.read_csv(path + name, sep=sep, header=0)
    data_matrix = raw_data.to_numpy()

    sample_size = np.shape(data_matrix)[0]
    print('Original Size: {0}'.format(sample_size))
    nums_treatment = np.sum(data_matrix[:, 12] == 1)
    nums_control = sample_size - nums_treatment
    print('Ratio of Treatment to Control: {0}'.format(nums_treatment/nums_control))
    visit_ratio = np.sum(data_matrix[:, 14] == 1) / sample_size
    print('Average Visit Ratio: {0}'.format(visit_ratio))
    uplift_treatment = np.sum((data_matrix[:, 12] == 1) & (data_matrix[:, 14] == 1)) / nums_treatment
    uplift_control = np.sum((data_matrix[:, 12] == 0) & (data_matrix[:, 14] == 1)) / nums_control
    print('Relative Average Uplift: {0}'.format((uplift_treatment - uplift_control) / uplift_control))
    print('Average Uplift: {0}'.format(uplift_treatment - uplift_control))

    data_matrix = raw_data.drop_duplicates().to_numpy()
    data_matrix = data_matrix[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]]
    if normalize:
        data_matrix[:, :12] = (data_matrix[:, :12] - np.min(data_matrix[:, :12])) / (
                np.max(data_matrix[:, :12]) - np.min(data_matrix[:, :12]))
    print('Current Size: {0}'.format(np.shape(data_matrix)[0]))
    return data_matrix


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)
