from discriminator import RandomForest
from discriminator import GBDT
from discriminator import LR
from discriminator import DT
from discriminator import NB
from discriminator import SVM
from discriminator import KNN
from discriminator import MLP
from discriminator import VOTE
from generator import BernoulliGenerator
from generator import MalGAN
from generator import AdversarialExamples
import numpy as np
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import split_matrix
import shutil
import sys


def learning_BernoulliGenerator():
    params = {
        'layers': [160, 256, 160],
        'batch size': 128,
        'learning rate': 0.001,
        'training set fraction': 0.75,
        'max epochs': 100,
        'max epochs no improvement': 25,
        'num trials': 10,
    }
    tag = '20161025_pre_train_API_rand_1gram_layers_%s_batch_%d_lr_%g_epo_%d_%d_trials_%d' % (
        '_'.join([str(layer) for layer in params['layers']]),
        params['batch size'],
        params['learning rate'],
        params['max epochs'],
        params['max epochs no improvement'],
        params['num trials']
    )
    dir_path = '../model/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    params['model path'] = dir_path + '/model'
    params['log path'] = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    D = RandomForrest()
    G = BernoulliGenerator(D, params)
    training_data = np.loadtxt('../data/API_truncation50_random_split_trainval_1gram_feature.csv',
                               delimiter=',', dtype=np.int32)
    test_data = np.loadtxt('../data/API_truncation50_random_split_test_1gram_feature.csv',
                           delimiter=',', dtype=np.int32)
    log_message = str(datetime.now()) + '\tTraining discriminative model on original dataset\n'
    D.train(training_data[:, :-1], training_data[:, -1])
    log_message += str(datetime.now()) + '\tTraining set result\t'
    log_message += score_template % D.evaluate(training_data[:, :-1], training_data[:, -1])
    log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    log_message += score_template % D.evaluate(test_data[:, :-1], test_data[:, -1])
    with open(params.get('log path'), 'a') as f:
        f.write(log_message + '\n')
    training_data_benign, training_data_malware = split_matrix(training_data)
    test_data_benign, test_data_malware = split_matrix(test_data)
    for i in range(50):
        log_message = str(datetime.now()) + '\tTraining generative model for the %d-th time\n' % (i,)
        #G.train(training_data_malware[:, :-1])
        G.train((training_data_malware[:, :-1], training_data_benign[:, :-1]))
        log_message += str(datetime.now()) + '\tGenerating examples\n'
        generated_training_malware, num_trials_training = G.sample(training_data_malware[:, :-1])
        generated_training_malware = np.concatenate((generated_training_malware,
                                                     training_data_malware[:, -1:]), axis=1)
        generated_training_data = np.concatenate((generated_training_malware, training_data_benign))
        generated_test_malware, num_trials_test = G.sample(test_data_malware[:, :-1])
        generated_test_malware = np.concatenate((generated_test_malware,
                                                 test_data_malware[:, -1:]), axis=1)
        generated_test_data = np.concatenate((generated_test_malware, test_data_benign))
        log_message += str(datetime.now()) + '\tMean number of trials for training and test set: %f, %f\n' % \
                                             (num_trials_training.mean(), num_trials_test.mean())
        log_message += str(datetime.now()) + '\tTraining set result before re-training\t'
        log_message += score_template % D.evaluate(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tTest set result before re-training\t'
        log_message += score_template % D.evaluate(generated_test_data[:, :-1], generated_test_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tRe-training discriminative model\n'
        D.train(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += str(datetime.now()) + '\tTraining set result after re-training\t'
        log_message += score_template % D.evaluate(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tTest set result after re-training\t'
        log_message += score_template % D.evaluate(generated_test_data[:, :-1], generated_test_data[:, -1])
        with open(params.get('log path'), 'a') as f:
            f.write(log_message + '\n\n')


def learning_MalGAN(D_name='RF', data_fraction='0.1', diff_data='0'):
    params = {
        'G layers': [160, 256, 160],
        'noise dim': 10,
        'malware batch size': 128,
        'L2': 0.0,
        'learning rate': 0.001,
        'training set fraction': 0.75,
        'D layers': [160, 256, 1],
        'batch size': 256,
        'regularization D layers': [160, 256, 1],
        'regu coef': 0.0,
        'max epochs': 200,
        'max epochs no improvement': 25,
        'num trials': 1
    }
#    tag = '20171013_%sMalGan_%s_drebin_%s_G_layers_%s_noise_%d_mal_batch_%d_L2_%g_' \
#          'D_layers_%s_batch_%d_regu_D_layers_%s_coef_%g_lr_%g_epoch_%d_%d_trials_%d' % (
#        '' if diff_data is '0' else 'diff_data_',
#        D_name,
#        data_fraction,
#        '_'.join([str(layer) for layer in params['G layers']]),
#        params['noise dim'],
#        params['malware batch size'],
#        params['L2'],
#        '_'.join([str(layer) for layer in params['D layers']]),
#        params['batch size'],
#        '_'.join([str(layer) for layer in params['regularization D layers']]),
#        params['regu coef'],
#        params['learning rate'],
#        params['max epochs'],
#        params['max epochs no improvement'],
#        params['num trials']
#    )
    tag = '20171026_WGAN'
    dir_path = '../result/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('./', os.path.join(dir_path, 'code'))

    params['model path'] = dir_path + '/model'
    params['log path'] = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    if D_name is 'RF':
        D = RandomForest()
    elif D_name is 'GBDT':
        D = GBDT()
    elif D_name is 'LR':
        D = LR()
    elif D_name is 'DT':
        D = DT()
    elif D_name is 'NB':
        D = NB()
    elif D_name is 'SVM':
        D = SVM()
    elif D_name is 'MLP':
        D = MLP()
    elif D_name is 'KNN':
        D = KNN()
    else:
        D = VOTE()

    G = MalGAN(D, params)
    training_data = np.loadtxt('../data/API_truncation50_random_split_trainval_1gram_feature.csv',
                               delimiter=',', dtype=np.int32)
    test_data = np.loadtxt('../data/API_truncation50_random_split_test_1gram_feature.csv',
                           delimiter=',', dtype=np.int32)
    log_message = str(datetime.now()) + '\tnow using ' + D_name + ' as Discrimibator\n'
    log_message += str(datetime.now()) + '\tTraining discriminative model on original dataset\n'
    if diff_data is '0':
        D.train(training_data[:, :-1], training_data[:, -1])
    else:
        D.train(training_data[:len(training_data) / 2, :-1], training_data[:len(training_data) / 2, -1])
        training_data = training_data[len(training_data) / 2:, :]
    log_message += str(datetime.now()) + '\tTraining set result\t'
    log_message += score_template % D.evaluate(training_data[:, :-1], training_data[:, -1])
    log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    log_message += score_template % D.evaluate(test_data[:, :-1], test_data[:, -1])
    with open(params.get('log path'), 'a') as f:
        f.write(log_message + '\n')
    training_data_benign, training_data_malware = split_matrix(training_data)
    test_data_benign, test_data_malware = split_matrix(test_data)
    for i in range(1):
        log_message = str(datetime.now()) + '\tTraining generative model for the %d-th time\n' % (i,)
        G.train((training_data_malware[:, :-1], training_data_benign[:, :-1]))
        log_message += str(datetime.now()) + '\tGenerating examples\n'
        generated_training_malware, num_trials_training = G.sample(training_data_malware[:, :-1])
        generated_training_malware = np.concatenate((generated_training_malware,
                                                     training_data_malware[:, -1:]), axis=1)
        generated_training_data = np.concatenate((generated_training_malware, training_data_benign))
        generated_test_malware, num_trials_test = G.sample(test_data_malware[:, :-1])
        generated_test_malware = np.concatenate((generated_test_malware,
                                                 test_data_malware[:, -1:]), axis=1)
        generated_test_data = np.concatenate((generated_test_malware, test_data_benign))
        log_message += str(datetime.now()) + '\tMean number of trials for training and test set: %f, %f\n' % \
                                             (num_trials_training.mean(), num_trials_test.mean())
        log_message += str(datetime.now()) + '\tTraining set result before re-training\t'
        log_message += score_template % D.evaluate(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tTest set result before re-training\t'
        log_message += score_template % D.evaluate(generated_test_data[:, :-1], generated_test_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tRe-training discriminative model\n'
        D.train(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += str(datetime.now()) + '\tTraining set result after re-training\t'
        log_message += score_template % D.evaluate(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tTest set result after re-training\t'
        log_message += score_template % D.evaluate(generated_test_data[:, :-1], generated_test_data[:, -1])
        with open(params.get('log path'), 'a') as f:
            f.write(log_message + '\n\n')


def genearting_adversarial_examples():
    params = {
        'learning rate': 0.001,
        'training set fraction': 0.75,
        'D layers': [44942, 200, 200, 1],
        'batch size': 256,
        'max epochs': 1000,
        'max epochs no improvement': 10,
        'num trials': 1000,
    }
    tag = '20170909_AdvExam_drebin_D_layers_%s_batch_%d_lr_%g_epoch_%d_%d_trials_%d' % (
        '_'.join([str(layer) for layer in params['D layers']]),
        params['batch size'],
        params['learning rate'],
        params['max epochs'],
        params['max epochs no improvement'],
        params['num trials']
    )
    dir_path = '../model/' + tag
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(os.path.join(dir_path, 'code')):
        shutil.rmtree(os.path.join(dir_path, 'code'))
    shutil.copytree('.', os.path.join(dir_path, 'code'))

    params['model path'] = dir_path + '/model'
    params['log path'] = dir_path + '/log.txt'
    score_template = 'TPR %(TPR)f\tFPR %(FPR)f\tAccuracy %(Accuracy)f\tAUC %(AUC)f'
    D = RandomForrest()
    G = AdversarialExamples(D, params)
    training_data = np.loadtxt('../data/drebin/drebin_train_0.5.csv',
                               delimiter=',', dtype=np.int32)
    test_data = np.loadtxt('../data/drebin/drebin_test_0.5.csv',
                           delimiter=',', dtype=np.int32)
    log_message = str(datetime.now()) + '\tTraining discriminative model on original dataset\n'
    D.train(training_data[:, :-1], training_data[:, -1])
    log_message += str(datetime.now()) + '\tTraining set result\t'
    log_message += score_template % D.evaluate(training_data[:, :-1], training_data[:, -1])
    log_message += '\n' + str(datetime.now()) + '\tTest set result\t'
    log_message += score_template % D.evaluate(test_data[:, :-1], test_data[:, -1])
    with open(params.get('log path'), 'a') as f:
        f.write(log_message + '\n')
    training_data_benign, training_data_malware = split_matrix(training_data)
    test_data_benign, test_data_malware = split_matrix(test_data)
    for i in range(1):
        log_message = str(datetime.now()) + '\tTraining generative model for the %d-th time\n' % (i,)
        G.train((training_data_malware[:, :-1], training_data_benign[:, :-1]))
        log_message += str(datetime.now()) + '\tGenerating examples\n'
        generated_training_malware, num_trials_training = G.sample(training_data_malware[:, :-1])
        generated_training_malware = np.concatenate((generated_training_malware,
                                                     training_data_malware[:, -1:]), axis=1)
        generated_training_data = np.concatenate((generated_training_malware, training_data_benign))
        generated_test_malware, num_trials_test = G.sample(test_data_malware[:, :-1])
        generated_test_malware = np.concatenate((generated_test_malware,
                                                 test_data_malware[:, -1:]), axis=1)
        generated_test_data = np.concatenate((generated_test_malware, test_data_benign))
        log_message += str(datetime.now()) + '\tMean number of trials for training and test set: %f, %f\n' % \
                                             (num_trials_training.mean(), num_trials_test.mean())
        log_message += str(datetime.now()) + '\tTraining set result before re-training\t'
        log_message += score_template % D.evaluate(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tTest set result before re-training\t'
        log_message += score_template % D.evaluate(generated_test_data[:, :-1], generated_test_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tRe-training discriminative model\n'
        D.train(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += str(datetime.now()) + '\tTraining set result after re-training\t'
        log_message += score_template % D.evaluate(generated_training_data[:, :-1], generated_training_data[:, -1])
        log_message += '\n' + str(datetime.now()) + '\tTest set result after re-training\t'
        log_message += score_template % D.evaluate(generated_test_data[:, :-1], generated_test_data[:, -1])
        with open(params.get('log path'), 'a') as f:
            f.write(log_message + '\n\n')

if __name__ == '__main__':
    #learning_BernoulliGenerator()
    #learning_MalGAN(sys.argv[1], sys.argv[2], sys.argv[3])
    #print sys.argv[1], sys.argv[2], sys.argv[3]
    for D_name in ['RF', 'GBDT', 'LR', 'DT', 'NB', 'SVM', 'MLP', 'KNN', 'VOTE']:
        learning_MalGAN(D_name, '0.1', '1')
    #for D_name in ['LR', 'DT', 'SVM', 'MLP', 'VOTE']:
    #    for data_fraction in ['0.1', '0.2', '0.3', '0.4', '0.5', 'all']:
    #        learning_MalGAN(D_name, data_fraction, '1')
    #genearting_adversarial_examples()
