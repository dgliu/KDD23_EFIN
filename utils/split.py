import numpy as np


def split_seed_randomly(data_matrix, ratio, split_seed):
    """
    Split based on a deterministic seed randomly
    """

    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Randomly shuffle the data
    sample_size = np.shape(data_matrix)[0]
    index_shuffle = np.arange(sample_size)
    np.random.shuffle(index_shuffle)
    data_matrix = data_matrix[index_shuffle]

    cut_point = int(ratio[0] * sample_size)

    train = data_matrix[:cut_point]

    test = data_matrix[cut_point:]

    return train, test
