import tensorflow as tf
import numpy as np
import math
import random

class GAN(object):
    # The basic class of GAN
    def train(self, X):
        raise NotImplementedError("Abstract method")
    def sample(self, X):
        raise NotImplementedError("Abstract method")

class malGAN(GAN):
    # malGAN - using WGAN as GAN moded
    def __init__(self, D_blackbox, params):
        """ intialize malGAN model
        :param D: the discriminator object
        :param params: the dict used to train the generative neural networks
        :return: None
        """
        self.D_blackbox = D_blackbox
        self.params = params
        self._build_model()

    def _build_model(self):
        """ build malGAN model
        :return: None
        """
        def init_weight(dim_in, dim_out, name=None, stddev=1.0):
            return tf.Variable(tf.truncated_normal([dim_in, dim_out],
                                                   stddev=stddev/tf.sqrt(float(dim_in))), name=name)
        def init_bias(dim_out, name=None):
            return tf.Variable(tf.zeros([dim_out]), name=name)

        tf.reset_default_graph()

        # define G network
        G_Ws = []
        G_bs = []
        G_layers = self.params.get('G_layers',[160,256,160])
        Z_dim = self.params.get('Noise_dim', 10)
        X_malware = tf.placeholder(tf.int32, [None, G_layers[0]])
        Z = tf.placeholder(tf.int32, [None, Z_dim])
        G_layers[0] += Z_dim
        for i in range(0, len(G_layers) - 1):
            G_Ws.append(init_weight(G_layers[i], G_layers[i + 1], 'G_Ws_%d' % (i,)))
            G_bs.append(init_bias(G_layers[i + 1], 'G_bs_%d' % (i,)))
        def G(x_malware,z):
            G_out = tf.concat(axis=1, values=[tf.to_float(x_malware), z])
            for i in range(0, len(G_layers) - 2):
                G_out = tf.nn.relu(tf.matmul(G_out, G_Ws[i]) + G_bs[i])
            G_out = tf.nn.sigmoid(tf.matmul(G_out, G_Ws[G_layers[-1]]) + G_bs[G_layers[-1]])
            G_out = tf.maximum(tf.to_int32(tf.round(G_out)), x_malware)
            return G_out

        # define D networrk
        D_Ws = []
        D_bs = []
        D_layers = self.params.get('D_layers',[160,256,1])
        X_benign = tf.placeholder(tf.int32, [None, D_layers[0]])
        D_in = tf.placeholder(tf.int32, [None, D_layers[0]])
        for i in range(0, len(D_layers) - 1):
            D_Ws.append(init_weight(D_layers[i], D_layers[i + 1], 'D_Ws_%d' % (i,)))
            D_bs.append(init_bias(D_layers[i + 1], 'D_bs_%d' % (i,)))
        def D(x):
            D_out = x
            for i in range(0, len(D_layers) - 2):
                D_out = tf.nn.relu(tf.matmul(D_out, D_Ws[i]) + D_bs[i])
            D_out = tf.nn.sigmoid(tf.matmul(D_out, D_Ws[D_layers[-1]]) + D_bs[D_layers[-1]])
            return D_out

        # cal loss
        G_out = G(X_malware,Z)
        D_in = tf.concat(axis=0,values=[G_out,X_benign])
        D_out = D(D_in)
        D_out_black = tf.placeholder(tf.int32, [None,1])

        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out,labels=D_out_black))
        G_loss = -tf.reduce_mean(D_fake)

        D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                    .minimize(-D_loss, var_list=theta_D))
        G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                    .minimize(G_loss, var_list=theta_G))

    def