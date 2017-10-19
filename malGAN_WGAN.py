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

        def G(G_layers, )
    def