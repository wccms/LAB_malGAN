import tensorflow as tf
import math
import numpy as np
import random
import glob
import os

class Generator(object):
    """
    The basic class for generator
    """
    def train(self, X):
        """ train a generator according to X
        :param X: the data matrix, or the list of data matrix
        """
        raise NotImplementedError("Abstract method")

    def sample(self, X):
        """ generate samples for X
        :param X: the data matrix
        :return: the generated samples, optional [the number of trials to get a misclassified sample]
        """
        raise NotImplementedError("Abstract method")


class BernoulliGenerator(Generator):
    """
    Generate a binary vector according to Bernoulli distribution
    """
    def __init__(self, D, params):
        """
        :param D: the discriminator object
        :param params: the dict used to train the generative neural networks
        """
        self.D = D
        self.params = params
        self._build_model()

    def _build_model(self):
        """ build a tensorflow model
        :return:
        """
        def init_weight(dim_in, dim_out, name=None, stddev=1.0):
            return tf.Variable(tf.truncated_normal([dim_in, dim_out],
                                                   stddev=stddev/math.sqrt(float(dim_in))), name=name)

        def init_bias(dim_out, name=None):
            return tf.Variable(tf.zeros([dim_out]), name=name)

        Ws = []
        bs = []
        layers = self.params.get('layers', [1024, 1024])
        for i in range(0, len(layers) - 1):
            Ws.append(init_weight(layers[i], layers[i + 1], 'W_%d' % (i,)))
            bs.append(init_bias(layers[i + 1], 'b_%d' % (i,)))
        batch_size = self.params.get('batch size', 128)
        self.input_data = tf.placeholder(tf.int32, [batch_size, layers[0]])
        self.generated_data = tf.placeholder(tf.int32, [batch_size, layers[-1]])
        self.discriminator_result = tf.placeholder(tf.int32, [batch_size])
        self.probability = tf.to_float(self.input_data)
        for i in range(0, len(layers) - 1):
            self.probability = tf.nn.sigmoid(tf.matmul(self.probability, Ws[i]) + bs[i])
        self.loss = (tf.to_float(self.generated_data) *
                      tf.log(tf.clip_by_value(self.probability, 1e-10, 1.0))
                      + (1 - tf.to_float(self.generated_data)) *
                      tf.log(tf.clip_by_value(1 - self.probability, 1e-10, 1.0)))
        self.loss = tf.reduce_sum(self.loss * (1 - tf.to_float(self.input_data)), 1)
        self.pre_train_loss = tf.reduce_mean(self.loss)
        self.loss = self.loss * tf.to_float(1 - 2 * self.discriminator_result)
        self.loss = tf.reduce_mean(self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.params.get('max epochs', 100))
        self.pre_train_op = tf.train.AdamOptimizer(self.params.get('learning rate', 0.001)).\
            minimize(-self.pre_train_loss)
        opt = tf.train.AdamOptimizer(self.params.get('learning rate', 0.001))
        grads_and_vars = opt.compute_gradients(-self.loss)
        grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.sess.run(tf.global_variables_initializer())

    def _pre_train(self, X_malware, X_malware_val, X_benign, num_trials=2):
        X_malware_all = np.concatenate((X_malware, X_malware_val))
        X_malware_repeated = np.repeat(X_malware_all, num_trials, axis=0)
        for i in range(len(X_malware_repeated)):
            X_malware_repeated[i] += random.sample(X_benign, 1)[0]
        discriminator_result = self.D.predict(X_malware_repeated)
        train_valid_index = []
        val_valid_index = []
        for idx, result in enumerate(discriminator_result):
            if result == 0:
                if idx / num_trials < len(X_malware):
                    train_valid_index.append(idx)
                else:
                    val_valid_index.append(idx)

        random.shuffle(train_valid_index)
        if len(train_valid_index) > len(X_malware):
            train_valid_index = train_valid_index[:len(X_malware)]
        X_malware_output = X_malware_repeated[train_valid_index]
        train_valid_index = [idx / num_trials for idx in train_valid_index]
        X_malware_input = X_malware_all[train_valid_index]

        random.shuffle(val_valid_index)
        if len(val_valid_index) > len(X_malware_val):
            val_valid_index = val_valid_index[:len(X_malware_val)]
        X_malware_output_val = X_malware_repeated[val_valid_index]
        val_valid_index = [idx / num_trials for idx in val_valid_index]
        X_malware_input_val = X_malware_all[val_valid_index]

        batch_size = self.params.get('batch size', 128)
        best_val_loss = -1e10
        best_val_epoch = 0
        with open(self.params.get('log path'), 'a') as f:
            f.write('Pre-Training\n%d / %d samples generated from %d / %d malware from %d trials\n' %
                    (len(train_valid_index), len(val_valid_index), len(X_malware), len(X_malware_val), num_trials))
        for epoch in range(self.params.get('max epochs', 100)):
            train_loss = 0.0
            for start, end in zip(range(0, len(X_malware_input), batch_size),
                                  range(batch_size, len(X_malware_input) + 1, batch_size)):
                X_batch_input = X_malware_input[start: end]
                X_batch_output = X_malware_output[start: end]
                _, loss_value = self.sess.run([self.pre_train_op, self.pre_train_loss], feed_dict={
                    self.input_data: X_batch_input,
                    self.generated_data: X_batch_output
                })
                train_loss += loss_value
            train_loss /= len(X_malware_input) / batch_size
            self.saver.save(self.sess, self.params.get('model path') + '-pre-train', global_step=epoch)
            val_loss = 0.0
            for start, end in zip(range(0, len(X_malware_input_val), batch_size),
                                  range(batch_size, len(X_malware_input_val) + 1, batch_size)):
                X_batch_input = X_malware_input_val[start: end]
                X_batch_output = X_malware_output_val[start: end]
                loss_value = self.sess.run(self.pre_train_loss, feed_dict={
                    self.input_data: X_batch_input,
                    self.generated_data: X_batch_output
                })
                val_loss += loss_value
            val_loss /= len(X_malware_input_val) / batch_size
            with open(self.params.get('log path'), 'a') as f:
                f.write('Epoch %d: Loss Training = %g . Val = %g\n' % (epoch, train_loss, val_loss))
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
            if epoch - best_val_epoch >= self.params.get('max epochs no improvement', 10):
                self.saver.restore(self.sess, self.params.get('model path') + '-pre-train' + '-' + str(best_val_epoch))
                break


    def train(self, X):
        """ train a generator according to X
        :param X: the data matrix, or the list of data matrix(first element is malware, second element is benign)
        """
        batch_size = self.params.get('batch size', 128)
        X_benign = None
        if isinstance(X, tuple):
            X_benign = X[1]
            X = X[0]
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        num_training_samples = int(len(X) * self.params.get('training set fraction', 0.75))
        X_val = X[num_training_samples:]
        X = X[:num_training_samples]
        if X_benign is not None:
            self._pre_train(X, X_val, X_benign)
        best_val_tpr = 1.0
        best_val_epoch = 0
        for epoch in range(self.params.get('max epochs', 100)):
            train_loss = 0.0
            train_tpr = 0.0
            break_flag = False
            for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X) + 1, batch_size)):
                X_batch = X[start: end]
                prob_batch = self.sess.run(self.probability, feed_dict={self.input_data: X_batch})
                rand = np.random.rand(prob_batch.shape[0], prob_batch.shape[1])
                generated_X_batch = ((rand < 0.5) & X_batch).astype(np.int32)
                discriminator_result_X_batch = self.D.predict(generated_X_batch)
                _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.input_data: X_batch,
                    self.generated_data: generated_X_batch,
                    self.discriminator_result: discriminator_result_X_batch
                })
                train_loss += loss_value
                train_tpr += discriminator_result_X_batch.mean()
            train_loss /= len(X) / batch_size
            train_tpr /= len(X) / batch_size
            self.saver.save(self.sess, self.params.get('model path'), global_step=epoch)
            log_message = 'Epoch %d: Training loss = %g / TPR %g' % (epoch, train_loss, train_tpr)
            if X_val is not None:
                val_loss = 0.0
                val_tpr = 0.0
                for start, end in zip(range(0, len(X_val), batch_size), range(batch_size, len(X_val) + 1, batch_size)):
                    X_batch = X_val[start: end]
                    prob_batch = self.sess.run(self.probability, feed_dict={self.input_data: X_batch})
                    rand = np.random.rand(prob_batch.shape[0], prob_batch.shape[1])
                    generated_X_batch = ((rand < 0.5) & X_batch).astype(np.int32)
                    discriminator_result_X_batch = self.D.predict(generated_X_batch)
                    loss_value = self.sess.run(self.loss, feed_dict={
                        self.input_data: X_batch,
                        self.generated_data: generated_X_batch,
                        self.discriminator_result: discriminator_result_X_batch
                    })
                    val_loss += loss_value
                    val_tpr += discriminator_result_X_batch.mean()
                val_loss /= len(X_val) / batch_size
                val_tpr /= len(X_val) / batch_size
                if val_tpr < best_val_tpr:
                    best_val_tpr = val_tpr
                    best_val_epoch = epoch
                log_message += ' Val loss = %g / TPR %g' % (val_loss, val_tpr)
                if epoch - best_val_epoch >= self.params.get('max epochs no improvement', 10):
                    self.saver.restore(self.sess, self.params.get('model path') + '-' + str(best_val_epoch))
                    break_flag = True
            with open(self.params.get('log path'), 'a') as f:
                f.write(log_message + '\n')
            if break_flag:
                break

    def sample(self, X):
        batch_size = self.params.get('batch size', 128)
        num_samples = len(X)
        X = np.concatenate((X, X[:batch_size - 1]))
        generated_X = np.zeros_like(X)
        num_trials = np.ones((len(X),), dtype=np.int32) * self.params.get('num trials', 1000)
        for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X) + 1, batch_size)):
            X_batch = X[start: end]
            prob_batch = self.sess.run(self.probability, feed_dict={self.input_data: X_batch})
            succeed_flag = np.zeros((batch_size,), dtype=np.int32)
            for t in range(self.params.get('num trials', 1000)):
                rand = np.random.rand(prob_batch.shape[0], prob_batch.shape[1])
                generated_X_batch = ((rand < prob_batch) & X_batch).astype(np.int32)
                discriminator_result_X_batch = self.D.predict(generated_X_batch)
                for i in range(batch_size):
                    if succeed_flag[i] == 0:
                        if discriminator_result_X_batch[i] == 0:
                            succeed_flag[i] = 1
                            generated_X[start + i] = generated_X_batch[i]
                            num_trials[start + i] = t
                        else:
                            generated_X[start + i] = generated_X_batch[i]

        return generated_X[:num_samples], num_trials[:num_samples]


class MalGAN(Generator):
    """
    GAN for generating malware.
    """
    def __init__(self, D, params):
        """
        :param D: the discriminator object
        :param params: the dict used to train the generative neural networks
        """
        self.D = D
        self.params = params
        self._build_model()

    def _build_model(self):
        """ build a tensorflow model
        :return:
        """
        def init_weight(dim_in, dim_out, name=None, stddev=1.0):
            return tf.Variable(tf.truncated_normal([dim_in, dim_out],
                                                   stddev=stddev/math.sqrt(float(dim_in))), name=name)

        def init_bias(dim_out, name=None):
            return tf.Variable(tf.zeros([dim_out]), name=name)

        tf.reset_default_graph()
        # the generator
        malware_batch_size = self.params.get('malware batch size', 128)
        G_Ws = []
        G_bs = []
        G_layers = self.params.get('G layers', [1024, 1024])
        self.malware_input_data = tf.placeholder(tf.int32, [malware_batch_size, G_layers[0]])
        G_layers[0] += self.params.get('noise dim', 10)
        for i in range(0, len(G_layers) - 1):
            G_Ws.append(init_weight(G_layers[i], G_layers[i + 1], 'G_W_%d' % (i,)))
            G_bs.append(init_bias(G_layers[i + 1], 'G_b_%d' % (i,)))
        noise = tf.random_uniform([malware_batch_size, self.params.get('noise dim', 10)])
        probability = tf.concat(axis=1, values=[tf.to_float(self.malware_input_data), noise])
        for i in range(0, len(G_layers) - 1):
            probability = tf.nn.sigmoid(tf.matmul(probability, G_Ws[i]) + G_bs[i])
        self.generated_data = tf.maximum(tf.to_int32(tf.round(probability)), self.malware_input_data)

        # the discriminator
        D_Ws = []
        D_bs = []
        D_layers = self.params.get('D layers', [1024, 1024, 1])
        for i in range(0, len(D_layers) - 1):
            D_Ws.append(init_weight(D_layers[i], D_layers[i + 1], 'D_W_%d' % (i,)))
            D_bs.append(init_bias(D_layers[i + 1], 'D_b_%d' % (i,)))

        # the regularization discriminator
        regu_D_Ws = []
        regu_D_bs = []
        regu_D_layers = self.params.get('regularization D layers', [1024, 1024, 1])
        for i in range(0, len(regu_D_layers) - 1):
            regu_D_Ws.append(init_weight(regu_D_layers[i], regu_D_layers[i + 1], 'regu_D_W_%d' % (i,)))
            regu_D_bs.append(init_bias(regu_D_layers[i + 1], 'regu_D_b_%d' % (i,)))

        # D loss
        batch_size = self.params.get('batch size', 256)
        self.input_data = tf.placeholder(tf.int32, [batch_size, D_layers[0]])
        hidden = tf.to_float(self.input_data)
        for i in range(0, len(D_layers) - 1):
            hidden = tf.nn.sigmoid(tf.matmul(hidden, D_Ws[i]) + D_bs[i])
        hidden = hidden[:, 0]
        self.discriminator_result = tf.placeholder(tf.int32, [batch_size])
        # D_loss : now use WGAN
        self.D_loss = -(tf.to_float(self.discriminator_result) *
                      tf.log(tf.clip_by_value(hidden, 1e-10, 1.0))
                      + (1 - tf.to_float(self.discriminator_result)) *
                      tf.log(tf.clip_by_value(1 - hidden, 1e-10, 1.0)))
        self.D_loss = tf.reduce_mean(self.D_loss)

        #regu D loss
        hidden = tf.to_float(self.input_data)
        for i in range(0, len(regu_D_layers) - 1):
            hidden = tf.nn.sigmoid(tf.matmul(hidden, regu_D_Ws[i]) + regu_D_bs[i])
        hidden = hidden[:, 0]
        label = tf.constant([1.0] * malware_batch_size + [0.0] * (batch_size - malware_batch_size))
        self.regu_D_loss = -(label *
                      tf.log(tf.clip_by_value(hidden, 1e-10, 1.0))
                      + (1 - label) *
                      tf.log(tf.clip_by_value(1 - hidden, 1e-10, 1.0)))
        self.regu_D_loss = tf.reduce_mean(self.regu_D_loss)

        # G loss
        hidden = tf.maximum(probability, tf.to_float(self.malware_input_data))
        for i in range(0, len(D_layers) - 1):
            hidden = tf.nn.sigmoid(tf.matmul(hidden, D_Ws[i]) + D_bs[i])
        hidden = hidden[:, 0]

        hidden_regu = tf.maximum(probability, tf.to_float(self.malware_input_data))
        for i in range(0, len(regu_D_layers) - 1):
            hidden_regu = tf.nn.sigmoid(tf.matmul(hidden_regu, regu_D_Ws[i]) + regu_D_bs[i])
        hidden_regu = hidden_regu[:, 0]
        self.regu_D_tpr = tf.reduce_mean(tf.round(hidden_regu))
        hidden = (hidden + hidden_regu * self.params.get('regu coef', 0.1)) / (1.0 + self.params.get('regu coef', 0.1))
        # G_loss : now use WGAN
        #self.G_loss = -tf.log(tf.clip_by_value(hidden, 1e-10, 1.0))
        self.G_loss = -tf.clip_by_value(hidden, 1e-10, 1.0)
        self.G_loss = tf.reduce_mean(self.G_loss)

        self.G_L2 = tf.reduce_sum(probability) / malware_batch_size
        self.G_L2 *= self.params.get('L2', 0.1)
        perturbation = tf.to_float(tf.greater(tf.to_int32(tf.round(probability)), self.malware_input_data))
        self.G_perturbation = tf.reduce_sum(perturbation) / malware_batch_size

        # gradient clipping
        D_opt = tf.train.AdamOptimizer(self.params.get('learning rate', 0.001))
        D_grads_and_vars = D_opt.compute_gradients(self.D_loss, D_Ws + D_bs)
        D_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in D_grads_and_vars]
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        self.D_train_op = D_opt.apply_gradients(D_grads_and_vars)

        regu_D_opt = tf.train.AdamOptimizer(self.params.get('learning rate', 0.001))
        regu_D_grads_and_vars = regu_D_opt.compute_gradients(self.regu_D_loss, regu_D_Ws + regu_D_bs)
        regu_D_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in regu_D_grads_and_vars]
        self.regu_D_train_op = regu_D_opt.apply_gradients(regu_D_grads_and_vars)

        G_opt = tf.train.AdamOptimizer(self.params.get('learning rate', 0.001))
        G_grads_and_vars = G_opt.compute_gradients(-self.G_loss + self.G_L2, G_Ws + G_bs)
        G_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in G_grads_and_vars]
        self.G_train_op = G_opt.apply_gradients(G_grads_and_vars)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.params.get('max epochs', 100))
        self.sess.run(tf.global_variables_initializer())
        #######DEBUG#############
        self.Weights = G_Ws + G_bs + D_Ws + D_bs + regu_D_Ws + regu_D_bs
        #######DEBUG#############

    def train(self, X):
        """ train a generator according to X
        :param X: the tuple of data matrix, first element is malware, second element is benign
        """
        self.sess.run(tf.global_variables_initializer())
        malware_batch_size = self.params.get('malware batch size', 128)
        batch_size = self.params.get('batch size', 256)
        benign_batch_size = batch_size - malware_batch_size
        X_malware, X_benign = X
        index = np.arange(len(X_malware))
        np.random.shuffle(index)
        X_malware = X_malware[index]
        index = np.arange(len(X_benign))
        np.random.shuffle(index)
        X_benign = X_benign[index]
        num_training_samples = int(len(X_malware) * self.params.get('training set fraction', 0.75))
        X_malware_val = X_malware[num_training_samples:]
        X_malware = X_malware[:num_training_samples]
        num_training_samples = int(len(X_benign) * self.params.get('training set fraction', 0.75))
        X_benign_val = X_benign[num_training_samples:]
        X_benign = X_benign[:num_training_samples]
        best_val_tpr = 1.0
        best_val_epoch = 0
        for epoch in range(self.params.get('max epochs', 100)):
            # training set
            train_D_loss = 0.0
            train_regu_D_loss = 0.0
            train_G_loss = 0.0
            train_G_L2 = 0.0
            train_G_perturbation = 0.0
            train_G_tpr = 0.0
            train_regu_D_tpr = 0.0
            break_flag = False
            idx_benign = 0
            for idx_malware in range(0, len(X_malware) - malware_batch_size + 1, malware_batch_size):
                X_malware_batch = X_malware[idx_malware: idx_malware + malware_batch_size]
                if idx_benign + benign_batch_size > len(X_benign):
                    idx_benign = 0
                X_benign_batch = X_benign[idx_benign: idx_benign + benign_batch_size]
                idx_benign += benign_batch_size
                # train D
                generated_X_batch = self.sess.run(self.generated_data,
                                                  feed_dict={self.malware_input_data: X_malware_batch})
                X_batch = np.concatenate((generated_X_batch, X_benign_batch))
                discriminator_result_X_batch = self.D.predict(X_batch)
                _, D_loss_value = self.sess.run([self.D_train_op, self.D_loss], feed_dict={
                self.input_data: X_batch,
                self.discriminator_result: discriminator_result_X_batch
                })
                train_D_loss += D_loss_value
                # train regu D
                _, regu_D_loss_value = self.sess.run([self.regu_D_train_op, self.regu_D_loss], feed_dict={
                self.input_data: X_batch
                })
                train_regu_D_loss += regu_D_loss_value
                # train G
                _, G_loss_value, G_L2_value, G_perturbation_value, regu_D_tpr_value = self.sess.run(
                    [self.G_train_op, self.G_loss, self.G_L2, self.G_perturbation, self.regu_D_tpr], feed_dict={
                self.malware_input_data: X_malware_batch
                })
                train_G_loss += G_loss_value
                train_G_L2 += G_L2_value
                train_G_perturbation += G_perturbation_value
                train_G_tpr += discriminator_result_X_batch[:malware_batch_size].mean()
                train_regu_D_tpr += regu_D_tpr_value

            train_D_loss /= len(X_malware) / malware_batch_size
            train_regu_D_loss /= len(X_malware) / malware_batch_size
            train_G_loss /= len(X_malware) / malware_batch_size
            train_G_L2 /= len(X_malware) / malware_batch_size
            train_G_perturbation /= len(X_malware) / malware_batch_size
            train_G_tpr /= len(X_malware) / malware_batch_size
            train_regu_D_tpr /= len(X_malware) / malware_batch_size
            self.saver.save(self.sess, self.params.get('model path'), global_step=epoch)
            log_message = 'Epoch %d: Training loss = D %g regu %g / G %g L2 %g Pert %g TPR %g (regu %g) . ' % \
                          (epoch, train_D_loss, train_regu_D_loss,
                           train_G_loss, train_G_L2, train_G_perturbation, train_G_tpr, train_regu_D_tpr)
            # val set
            val_D_loss = 0.0
            val_regu_D_loss = 0.0
            val_G_loss = 0.0
            val_G_L2 = 0.0
            val_G_perturbation = 0.0
            val_G_tpr = 0.0
            val_regu_D_tpr = 0.0
            idx_benign = 0
            for idx_malware in range(0, len(X_malware_val) - malware_batch_size + 1, malware_batch_size):
                X_malware_batch = X_malware_val[idx_malware: idx_malware + malware_batch_size]
                if idx_benign + benign_batch_size > len(X_benign_val):
                    idx_benign = 0
                X_benign_batch = X_benign_val[idx_benign: idx_benign + benign_batch_size]
                idx_benign += benign_batch_size
                # val D
                generated_X_batch = self.sess.run(self.generated_data,
                                                  feed_dict={self.malware_input_data: X_malware_batch})
                X_batch = np.concatenate((generated_X_batch, X_benign_batch))
                discriminator_result_X_batch = self.D.predict(X_batch)
                D_loss_value = self.sess.run(self.D_loss, feed_dict={
                    self.input_data: X_batch,
                    self.discriminator_result: discriminator_result_X_batch
                })
                val_D_loss += D_loss_value
                # val regu D
                regu_D_loss_value = self.sess.run(self.regu_D_loss, feed_dict={
                self.input_data: X_batch
                })
                val_regu_D_loss += regu_D_loss_value
                # val G
                G_loss_value, G_L2_value, G_perturbation_value, regu_D_tpr_value = self.sess.run(
                    [self.G_loss, self.G_L2, self.G_perturbation, self.regu_D_tpr], feed_dict={
                self.malware_input_data: X_malware_batch
                })
                val_G_loss += G_loss_value
                val_G_L2 += G_L2_value
                val_G_perturbation += G_perturbation_value
                val_G_tpr += discriminator_result_X_batch[:malware_batch_size].mean()
                val_regu_D_tpr += regu_D_tpr_value

            val_D_loss /= len(X_malware_val) / malware_batch_size
            val_regu_D_loss /= len(X_malware_val) / malware_batch_size
            val_G_loss /= len(X_malware_val) / malware_batch_size
            val_G_L2 /= len(X_malware_val) / malware_batch_size
            val_G_perturbation /= len(X_malware_val) / malware_batch_size
            val_G_tpr /= len(X_malware_val) / malware_batch_size
            val_regu_D_tpr /= len(X_malware_val) / malware_batch_size
            log_message += 'Val loss = D %g regu %g / G %g L2 %g Pert %g TPR %g (regu %g)' % \
                           (val_D_loss, val_regu_D_loss,
                            val_G_loss, val_G_L2, val_G_perturbation, val_G_tpr, val_regu_D_tpr)
            if val_G_tpr < best_val_tpr:
                best_val_epoch = epoch
                best_val_tpr = val_G_tpr
            if epoch - best_val_epoch >= self.params.get('max epochs no improvement', 10):
                self.saver.restore(self.sess, self.params.get('model path') + '-' + str(best_val_epoch))
                break_flag = True
            #######DEBUG#############
            #for weight in self.Weights:
            #    log_message += '. %s: %g, %g; ' % (weight.name, tf.reduce_mean(tf.abs(weight)).eval(session=self.sess),
            #                            tf.reduce_max(tf.abs(weight)).eval(session=self.sess))
            #######DEBUG#############
            with open(self.params.get('log path'), 'a') as f:
                f.write(log_message + '\n')
            if break_flag:
                break
        for f in glob.glob(self.params.get('model path') + '*'):
            os.remove(f)

    def sample(self, X):
        malware_batch_size = self.params.get('malware batch size', 128)
        num_samples = len(X)
        X = np.concatenate((X, X[:malware_batch_size - 1]))
        num_trials = np.ones((len(X),), dtype=np.int32) * self.params.get('num trials', 1000)
        generated_X = np.zeros_like(X)
        for start, end in \
                zip(range(0, len(X), malware_batch_size), range(malware_batch_size, len(X) + 1, malware_batch_size)):
            X_malware_batch = X[start: end]
            succeed_flag = np.zeros((malware_batch_size,), dtype=np.int32)
            for t in range(self.params.get('num trials', 1000)):
                generated_X_batch = self.sess.run(self.generated_data,
                                                  feed_dict={self.malware_input_data: X_malware_batch})
                discriminator_result_X_batch = self.D.predict(generated_X_batch)
                for i in range(malware_batch_size):
                    if succeed_flag[i] == 0:
                        if discriminator_result_X_batch[i] == 0:
                            succeed_flag[i] = 1
                            generated_X[start + i] = generated_X_batch[i]
                            num_trials[start + i] = t
                        else:
                            generated_X[start + i] = generated_X_batch[i]
        return generated_X[:num_samples], num_trials[:num_samples]


class AdversarialExamples(Generator):
    """
    Generating adversarial examples using jacobian matrix
    """
    def __init__(self, D, params):
        """
        :param D: the discriminator object
        :param params: the dict used to train the generative neural networks
        """
        self.D = D
        self.params = params
        self._build_model()

    def _build_model(self):
        """ build a tensorflow model
        :return:
        """
        def init_weight(dim_in, dim_out, name=None, stddev=1.0):
            return tf.Variable(tf.truncated_normal([dim_in, dim_out],
                                                   stddev=stddev/math.sqrt(float(dim_in))), name=name)

        def init_bias(dim_out, name=None):
            return tf.Variable(tf.zeros([dim_out]), name=name)

        # the discriminator
        D_Ws = []
        D_bs = []
        D_layers = self.params.get('D layers', [1024, 1024, 1])
        for i in range(0, len(D_layers) - 1):
            D_Ws.append(init_weight(D_layers[i], D_layers[i + 1], 'D_W_%d' % (i,)))
            D_bs.append(init_bias(D_layers[i + 1], 'D_b_%d' % (i,)))

        # D loss
        batch_size = self.params.get('batch size', 256)
        self.input_data = tf.placeholder(tf.int32, [batch_size, D_layers[0]])
        float_input_data = tf.to_float(self.input_data)
        hidden = float_input_data
        for i in range(0, len(D_layers) - 1):
            hidden = tf.nn.sigmoid(tf.matmul(hidden, D_Ws[i]) + D_bs[i])
        hidden = hidden[:, 0]
        self.discriminator_result = tf.placeholder(tf.int32, [batch_size])
        self.D_loss = -(tf.to_float(self.discriminator_result) *
                      tf.log(tf.clip_by_value(hidden, 1e-10, 1.0))
                      + (1 - tf.to_float(self.discriminator_result)) *
                      tf.log(tf.clip_by_value(1 - hidden, 1e-10, 1.0)))
        self.D_loss = tf.reduce_mean(self.D_loss)


        # gradient clipping
        D_opt = tf.train.AdamOptimizer(self.params.get('learning rate', 0.001))
        D_grads_and_vars = D_opt.compute_gradients(self.D_loss, D_Ws + D_bs)
        D_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in D_grads_and_vars]
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        self.D_train_op = D_opt.apply_gradients(D_grads_and_vars)

        # gradient
        self.input_gradients = tf.gradients(tf.reduce_sum(hidden), float_input_data)[0]

        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.params.get('max epochs', 100))
        self.sess.run(tf.global_variables_initializer())

    def train(self, X):
        """ train a generator according to X
        :param X: the tuple of data matrix, first element is malware, second element is benign
        """
        self.benign_X = X[1]
        self.sess.run(tf.global_variables_initializer())
        batch_size = self.params.get('batch size', 256)

        X = np.concatenate(X)
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        y = self.D.predict(X)
        num_training_samples = int(len(X) * self.params.get('training set fraction', 0.75))
        X_val = X[num_training_samples:]
        y_val = y[num_training_samples:]
        X = X[:num_training_samples]
        y = y[:num_training_samples]

        best_val_loss = 1000.0
        best_val_epoch = 0
        for epoch in range(self.params.get('max epochs', 100)):
            # training set
            train_D_loss = 0.0
            break_flag = False
            for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X) + 1, batch_size)):
                _, D_loss_value = self.sess.run([self.D_train_op, self.D_loss], feed_dict={
                    self.input_data: X[start: end],
                    self.discriminator_result: y[start: end]
                })
                train_D_loss += D_loss_value

            train_D_loss /= len(X) / batch_size

            self.saver.save(self.sess, self.params.get('model path'), global_step=epoch)
            log_message = 'Epoch %d: Training loss = D %g . ' % (epoch, train_D_loss)
            # val set
            val_D_loss = 0.0
            for start, end in zip(range(0, len(X_val), batch_size), range(batch_size, len(X_val) + 1, batch_size)):
                D_loss_value = self.sess.run(self.D_loss, feed_dict={
                    self.input_data: X_val[start: end],
                    self.discriminator_result: y_val[start: end]
                })
                val_D_loss += D_loss_value

            val_D_loss /= len(X_val) / batch_size

            log_message += 'Val loss = D %g .' % (val_D_loss,)
            if val_D_loss < best_val_loss:
                best_val_epoch = epoch
                best_val_loss = val_D_loss
            if epoch - best_val_epoch >= self.params.get('max epochs no improvement', 10):
                self.saver.restore(self.sess, self.params.get('model path') + '-' + str(best_val_epoch))
                break_flag = True

            with open(self.params.get('log path'), 'a') as f:
                f.write(log_message + '\n')
            if break_flag:
                break
        for f in glob.glob(self.params.get('model path') + '*'):
            os.remove(f)

    def sample(self, X):
        batch_size = self.params.get('batch size', 256)
        num_samples = len(X)
        X = np.concatenate((X, X[:batch_size - 1]))
        generated_X = X
        num_trials = np.ones((len(X),), dtype=np.int32) * self.params.get('num trials', 1000)
        succeed_flag = np.zeros((len(X),), dtype=np.int32)
        for t in range(self.params.get('num trials', 1000)):
            print t
            for start, end in zip(range(0, len(X), batch_size), range(batch_size, len(X) + 1, batch_size)):
                X_batch = generated_X[start: end]
                #succeed_flag = np.zeros((batch_size,), dtype=np.int32)
                discriminator_result_X_batch = self.D.predict(X_batch)
                for i in range(batch_size):
                    if succeed_flag[start + i] == 0 and discriminator_result_X_batch[i] == 0:
                        succeed_flag[start + i] = 1
                        num_trials[start + i] = t
                input_gradients_value = self.sess.run(self.input_gradients, feed_dict={
                    self.input_data: X_batch
                })
                input_gradients_value -= np.amax(input_gradients_value)
                input_gradients_value *= 1 - X_batch
                index = np.argmin(input_gradients_value, axis=1)
                for i in range(batch_size):
                    if succeed_flag[start + i] == 0 and X_batch[i, index[i]] == 0:
                        X_batch[i, index[i]] = 1

                generated_X[start: end] = X_batch
            self.train((generated_X, self.benign_X))
        return generated_X[:num_samples], num_trials[:num_samples]
