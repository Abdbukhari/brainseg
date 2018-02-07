__author__ = 'Anson Leung'

import time
import numpy as np

import theano
import theano.tensor as T

import lasagne


class Trainer():
    """
    Class that supervises the training of a neural network.

    Attributes:
        net (Network object): the network to be trained
        ds_training (Dataset object): the dataset on which the network is trained
        cost_function (CostFunction object): the cost function of the training

        batch_size (int): number of training datapoints to include in a training batch
        n_train_batches (int): number of batches that the dataset contains

        ls_monitors (list of Monitor objects): each monitor tracks a particular statistic of the training
        ls_stopping_criteria (list of StoppingCriterion objects): stopping criteria that decide when to
            stop the training

        train_minibatch (function): function to train the network on a single minibatch
    """
    def __init__(self, net, ds_testing,  ds_val, ds_training, batch_size, learning_rate):
        print('Configure training ...')

        self.net = net
        self.ds_testing = ds_testing
        self.ds_val = ds_val
        self.ds_training = ds_training
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.n_train_batches = self.ds_training.n_data / self.batch_size

        self.train_minibatch = None

        self.best_dice = 0
        self.real_best_dice = 0
        self.best_net_param = None

        self.val_dices = []
        self.val_errs = []
        self.test_dices =[]
        self.test_errs = []
        

    # def record(self, epoch, epoch_minibatch, id_minibatch, force_record=False, update_stopping=True, verbose=True):
    #     """
    #     Record statistics about the training.
    #     Returns True is at least one value is recorded.
    #     """
    #     updated_monitors = []  # memorize monitors that record a new value

    #     for i, monitor in enumerate(self.ls_monitors):
    #         has_monitored = monitor.record(epoch, epoch_minibatch, id_minibatch, force_record, update_stopping, verbose)
    #         if has_monitored:
    #             updated_monitors.append(i)

    #     if verbose and updated_monitors:
    #         print("    minibatch {}/{}:".format(epoch_minibatch, self.n_train_batches))
    #         for i in updated_monitors:
    #             print("        {}".format(self.ls_monitors[i].str_value_from_position(-1)))

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]


    def train(self):
        print("Train the network ...")

        start_time = time.clock()

        freq_display_batch = max(self.n_train_batches / 4, 1)  # Frequency for printing the batch id
        epoch_id = minibatch_id = 0

        # Record statistics before training really starts
        # self.record(epoch_id, 0, minibatch_id)

        stop = False
        patience = 5
        patience_increase = 10
        maxEpoch = 300
        while not stop:
            starting_epoch_time = time.clock()
            epoch_id += 1
            print("Epoch {}".format(epoch_id))
            if epoch_id >= 20:
                if epoch_id%20 == 0:
                    self.learning_rate /= 2

            train_loss = 0
            train_err = 0
            train_dice = 0
            train_batches = 0
            train_common_classes = np.zeros((self.net.n_out-1, 3))

            for batch in self.iterate_minibatches(self.ds_training.inputs, self.ds_training.outputs, self.batch_size, shuffle=True):
                train_batches += 1

                if train_batches%freq_display_batch == 0:
                    print("    minibatch {}/{}".format(train_batches, self.n_train_batches))

                inputs, targets = batch
                loss, err, pred = self.net.train_fn(inputs, targets, self.learning_rate)

                # dice = np.sum(dice)/(self.net.n_out-1)
                # print(dice)
                train_loss += loss
                train_err += err
                train_common_classes += self.net.count_common_classes(np.argmax(pred,axis=1), np.argmax(targets,axis=1), self.net.n_out)

            train_dice = np.mean(self.net.compute_dice_from_counts(train_common_classes))

            val_loss = 0
            val_err = 0
            val_dice = 0
            val_batches = 0
            val_common_classes = np.zeros((self.net.n_out-1, 3))

            for batch in self.iterate_minibatches(self.ds_val.inputs, self.ds_val.outputs, self.batch_size, shuffle=False):
                val_batches += 1

                inputs, targets = batch
                loss, err, pred = self.net.val_fn(inputs, targets)

                # dice = np.sum(dice)/(self.net.n_out-1)
                val_loss += loss
                val_err += err
                val_common_classes += self.net.count_common_classes(np.argmax(pred,axis=1), np.argmax(targets,axis=1), self.net.n_out)

            val_dice = np.mean(self.net.compute_dice_from_counts(val_common_classes))

            test_loss = 0
            test_err = 0
            test_dice = 0
            test_batches = 0
            test_common_classes = np.zeros((self.net.n_out-1, 3))

            for batch in self.iterate_minibatches(self.ds_testing.inputs, self.ds_testing.outputs, self.batch_size, shuffle=False):
                test_batches += 1

                inputs, targets = batch
                loss, err, pred = self.net.val_fn(inputs, targets)

                # dice = np.sum(dice)/(self.net.n_out-1)
                test_loss += loss
                test_err += err
                test_common_classes += self.net.count_common_classes(np.argmax(pred,axis=1), np.argmax(targets,axis=1), self.net.n_out)

            test_dice = np.mean(self.net.compute_dice_from_counts(test_common_classes))

            if patience <= epoch_id:
                stop = True
            else:
                stop = False
                if test_dice*0.99 > self.best_dice:
                    self.best_dice = test_dice
                    patience += patience_increase
                    print("         patience increased")
                if test_dice > self.real_best_dice:
                    self.real_best_dice = test_dice
                    self.best_net_param = lasagne.layers.get_all_param_values(self.net.net)
                    print("         best parameters updated")


            print("   epoch {} finished after {} seconds".format(epoch_id, time.clock() - starting_epoch_time))
            print("   training error:\t\t{:.6f}".format(train_err / train_batches))
            print("   validation error:\t\t{:.6f}".format(val_err / val_batches))
            print("   testing error:\t\t{:.6f}".format(test_err / test_batches))
            print("   training dice coefficient:\t{:.6f}".format(train_dice))
            print("   validation dice coefficient:\t{:.6f}".format(val_dice))
            print("   testing dice coefficient:\t{:.6f}".format(test_dice))

            self.val_dices.append(val_dice)
            self.val_errs.append(val_err / val_batches)
            self.test_dices.append(test_dice)
            self.test_errs.append(test_err / test_batches)



        end_time = time.clock()
        print("Training ran for {} minutes".format((end_time - start_time) / 60.))

        if test_dice > self.real_best_dice:
            self.real_best_dice = test_dice
            self.best_net_param = lasagne.layers.get_all_param_values(self.net.net)
            print("         best parameters updated")

        lasagne.layers.set_all_param_values(self.net.net, self.best_net_param)
        print("Test the best param network")

        val_loss = 0
        val_err = 0
        val_dice = 0
        val_batches = 0
        val_common_classes = np.zeros((self.net.n_out-1, 3))

        for batch in self.iterate_minibatches(self.ds_val.inputs, self.ds_val.outputs, self.batch_size, shuffle=False):
            val_batches += 1

            inputs, targets = batch
            loss, err, pred = self.net.val_fn(inputs, targets)

            # dice = np.sum(dice)/(self.net.n_out-1)
            val_loss += loss
            val_err += err
            val_common_classes += self.net.count_common_classes(np.argmax(pred,axis=1), np.argmax(targets,axis=1), self.net.n_out)

        val_dice = np.mean(self.net.compute_dice_from_counts(val_common_classes))

        test_loss = 0
        test_err = 0
        test_dice = 0
        test_batches = 0
        test_common_classes = np.zeros((self.net.n_out-1, 3))

        for batch in self.iterate_minibatches(self.ds_testing.inputs, self.ds_testing.outputs, self.batch_size, shuffle=False):
            test_batches += 1

            inputs, targets = batch
            loss, err, pred = self.net.val_fn(inputs, targets)

            # dice = np.sum(dice)/(self.net.n_out-1)
            test_loss += loss
            test_err += err
            test_common_classes += self.net.count_common_classes(np.argmax(pred,axis=1), np.argmax(targets,axis=1), self.net.n_out)

        test_dice = np.mean(self.net.compute_dice_from_counts(test_common_classes))

        print("   validation error:\t\t{:.6f}".format(val_err / val_batches))
        print("   testing error:\t\t{:.6f}".format(test_err / test_batches))
        print("   validation dice coefficient:\t{:.6f}".format(val_dice))
        print("   testing dice coefficient:\t{:.6f}".format(test_dice))


