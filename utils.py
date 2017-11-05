imimport numpy as np

def dataset():
    def __init__(self):
        self.train_data = np.loadtxt('../data/API_truncation50_random_split_trainval_1gram_feature.csv',
                               delimiter=',', dtype=np.int32)
        self.test_data = np.loadtxt('../data/API_truncation50_random_split_test_1gram_feature.csv',
                               delimiter=',', dtype=np.int32)
        self.train_data_benign, self.training_data_malware = split_matrix(self.train_data)
        self.test_data_benign, self.test_data_malware = split_matrix(self.test_data)
        self.

def split_matrix(data):
    """ split the data with binary label into two matrix, each with one class
    :param data: the data matrix to be split, the last column is the class label
    :return: two matrix, one for each class
    """
    data0 = np.zeros((len(data) - data[:, -1].sum(), data.shape[1]), dtype=np.int32)
    data1 = np.zeros((data[:, -1].sum(), data.shape[1]), dtype=np.int32)
    i0 = 0
    i1 = 0
    for i in range(len(data)):
        if data[i, -1] == 0:
            data0[i0] = data[i]
            i0 += 1
        else:
            data1[i1] = data[i]
            i1 += 1
    return data0, data1


def arff2csv(arff_path, csv_path):
    import arff
    data = arff.load(open(arff_path, 'rb'))
    f = open(csv_path, 'w')
    for datum in data[u'data']:
        output_line = []
        for element in datum:
            if (element is u'F') or (element is u'Benign'):
                output_line.append('0')
            elif (element is u'T') or (element is u'Malware'):
                output_line.append('1')
            else:
                print 'unsupported symbol ' + element
                return
        f.write(','.join(output_line) + '\n')
    f.close()


def arff2csv_main():
    arff2csv


def test_random_merge():
    max_length = 100
    from seq_adversarial_learning import _get_rnn_model
    D = _get_rnn_model('/seq_discriminator_model', max_length=2 * max_length)
    from seq_adversarial_learning import load_dataset
    X_malware, malware_length, X_benign, benign_length = \
        load_dataset('../data/API_rand_trainval_len_2048.txt', max_length, 0)
    X = np.vstack((X_malware, X_benign))
    sequence_length = np.hstack((malware_length, benign_length))
    y = np.array([1] * len(X_malware) + [0] * len(X_benign))
    D.train(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    generated_malware = np.hstack((np.zeros_like(X_malware), np.zeros_like(X_malware)))
    generated_length = np.zeros_like(malware_length)
    succeed_flag = np.zeros_like(malware_length)
    import random
    for t in range(100):
        for i in range(len(X_malware)):
            if succeed_flag[i] == 0:
                benign_idx = random.randint(0, len(X_benign) - 1)
                generated_length[i] = malware_length[i] + benign_length[benign_idx]
                idx_0 = 0
                idx_1 = 0
                while idx_0 < malware_length[i] or idx_1 < benign_length[benign_idx]:
                    selected = random.randint(0, 1)
                    if idx_1 >= benign_length[benign_idx] or (idx_0 < malware_length[i] and selected == 0):
                        generated_malware[i, idx_0 + idx_1] = X_malware[i, idx_0]
                        idx_0 += 1
                    else:
                        generated_malware[i, idx_0 + idx_1] = X_benign[benign_idx, idx_1]
                        idx_1 += 1

        generated_result = D.predict(generated_malware, generated_length)
        print t, generated_result.mean()
        for i in range(len(X_malware)):
            if succeed_flag[i] == 0 and generated_result[i] == 0:
                succeed_flag[i] = 1


def test_random_insert():
    max_length = 100
    from seq_adversarial_learning import _get_rnn_model
    D = _get_rnn_model('/seq_discriminator_model', max_length=2 * max_length)
    from seq_adversarial_learning import load_dataset
    X_malware, malware_length, X_benign, benign_length = \
        load_dataset('../data/API_rand_trainval_len_2048.txt', max_length, 0)
    X = np.vstack((X_malware, X_benign))
    sequence_length = np.hstack((malware_length, benign_length))
    y = np.array([1] * len(X_malware) + [0] * len(X_benign))
    D.train(np.hstack((X, np.zeros_like(X))), sequence_length, y)
    generated_malware = np.hstack((np.zeros_like(X_malware), np.zeros_like(X_malware)))
    generated_length = np.zeros_like(malware_length)
    succeed_flag = np.zeros_like(malware_length)
    import random
    for t in range(100):
        for i in range(len(X_malware)):
            if succeed_flag[i] == 0:
                idx = 0
                for j in range(malware_length[i]):
                    generated_malware[i, idx] = X_malware[i, j]
                    idx += 1
                    selected = random.randint(0, 161)
                    if selected < 161:
                        generated_malware[i, idx] = selected
                        idx += 1
                generated_length[i] = idx

        generated_result = D.predict(generated_malware, generated_length)
        print t, generated_result.mean()
        for i in range(len(X_malware)):
            if succeed_flag[i] == 0 and generated_result[i] == 0:
                succeed_flag[i] = 1

if __name__ == '__main__':
    test_random_insert()
