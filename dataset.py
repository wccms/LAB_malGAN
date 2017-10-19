import numpy as np
from utils import split_matrix

class dataset(object):
    """
    dataset class: read and offer data
    """
    def __init__(self):
        self.data_train = np.loadtxt('./data/API_truncation50_random_split_trainval_1gram_feature.csv',
                                     delimiter=',', dtype=np.int32)
        self.data_test = np.loadtxt('./data/API_truncation50_random_split_test_1gram_feature.csv',
                                    delimiter=',', dtype=np.int32)

        self.train_index = 0
        self.test_index = 0

    def
