__author__ = 'Anson Leung'

import numpy as np
import theano
import theano.tensor as T
from spynet.utils.utilities import error_rate

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, ConcatLayer, SliceLayer, ReshapeLayer, \
MaxPool2DLayer, FlattenLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, ExpressionLayer, PadLayer,\
DropoutLayer, batch_norm, get_all_layers, Pool2DLayer
from lasagne.regularization import regularize_layer_params, l2, l1

from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer, MaxPool3DDNNLayer as MaxPool3DLayer

from lasagne.nonlinearities import rectify as relu, softmax


class Network():
    def __init__(self, n_out, input_var, target_var):
        self.n_out = n_out
        self.input_var = input_var
        self.target_var = target_var
        self.net = None

        self.build_net()

        prediction = lasagne.layers.get_output(self.net)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean(dtype=theano.config.floatX)
        # l2_penalty = regularize_layer_params(self.net, l2)
        # loss = loss + l2_penalty

        # dice = self.compute_dice_symb(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1), n_out)

        err =  T.mean(T.neq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

        learning_rate = T.fscalar(name='learning_rate')
        params = lasagne.layers.get_all_params(self.net, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)


        test_prediction = lasagne.layers.get_output(self.net, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean(dtype=theano.config.floatX)

        # test_dice = self.compute_dice_symb(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1), n_out)

        test_err = T.mean(T.neq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)


        self.train_fn = theano.function([input_var, target_var, learning_rate], [loss, err, prediction], updates=updates)

        self.val_fn = theano.function([input_var, target_var], [test_loss, test_err, test_prediction])

    def build_net(self):
        print("Build the net...")
        self.build_net_virtual()
        print("Network strcture:\n" + str(self.net_structure()))

    def buid_net_virtual(self):
        raise NotImplementedError

    def compute_dice_symb(self, vec_pred, vec_true, n_classes_max):
        """
        Compute the DICE score between two segmentations in theano
        """
        vec_pred = vec_pred.dimshuffle((0, 'x'))
        vec_true = vec_true.dimshuffle((0, 'x'))

        max_count = T.max([T.max(vec_pred), T.max(vec_true)])

        classes = theano.shared(np.arange(1, n_classes_max))  # Start from 1 because we don't consider 0
        classes = classes.dimshuffle(('x', 0))

        binary_pred = T.cast(T.eq(vec_pred, classes), theano.config.floatX)
        binary_true = T.cast(T.eq(vec_true, classes), theano.config.floatX)
        binary_common = binary_pred * binary_true

        binary_pred_sum = T.sum(binary_pred, axis=0)
        binary_true_sum = T.sum(binary_true, axis=0)
        binary_common_sum = T.sum(binary_common, axis=0)
        no_zero = binary_common_sum.nonzero()

        return T.sum(2 * binary_common_sum[no_zero] / (binary_pred_sum[no_zero] + binary_true_sum[no_zero]), dtype=theano.config.floatX)/max_count

    def count_common_classes(self, pred, true, n_classes):
        """
        pred and true are one-dimensional vectors of integers
        For each class c, compute three values:
            - the number of elements of classes c in pred
            - the number of elements of classes c in true
            - the number of common elements of classes c in pred and true
        """
        counts = np.zeros((n_classes,3))

        for c in range(1, n_classes):  # Start from 1 because we don't consider 0
            class_pred = pred == c
            class_true = true == c

            counts[c, 0] = np.sum(class_pred)
            counts[c, 1] = np.sum(class_true)
            counts[c, 2] = np.sum(class_true * class_pred)

        return counts[1:, :]

    def compute_dice_from_counts(self, counts):
        """
        Compute the DICE score from function count_common_classes
        """
        dices = 2 * np.asarray(counts[:, 2], dtype=float) / (counts[:, 0] + counts[:, 1])
        return dices[~np.isnan(dices)]

    def compute_dice(self, pred, true, n_classes):
        """
        Compute the DICE score between two vectors pred and true
        """
        counts = self.count_common_classes(pred, true, n_classes)
        dices = 2 * np.asarray(counts[:, 2], dtype=float) / (counts[:, 0] + counts[:, 1])

        return dices[~np.isnan(dices)]

    def net_structure(self):
        return [l.__class__.__name__ for l in get_all_layers(self.net)]

    def generate_testing_function(self):
        """
        Generate a C-compiled function that can be used to compute the output of the network from an input batch
        Args:
            batch_size (int): the input of the returned function will be a batch of batch_size elements
        Returns:
            (function): function that returns the output of the network for a given input batch
        """
        # in_batch = T.matrix('inputs')  # Minibatch input matrix
        y_pred = lasagne.layers.get_output(self.net, deterministic=True)
        return theano.function([self.input_var], y_pred)

    def predict_from_generator(self, batches_generator, scaler, pred_functions=None, raw=False):
        """
        Returns the predictions of the batches of voxels, features and targets yielded by the batches_generator
        """
        if pred_functions is None:
            pred_functions = {}
        ls_vx = []
        ls_pred = []
        id_batch = 0
        for vx_batch, patch_batch, tg_batch in batches_generator:
            id_batch += 1

            batch_size_current = len(vx_batch)
            if batch_size_current not in pred_functions:
                pred_functions[batch_size_current] = self.generate_testing_function()

            if scaler is not None:
                scaler.scale(patch_batch)

            pred_raw = pred_functions[batch_size_current](patch_batch)

            pred = np.argmax(pred_raw, axis=1)
            err = error_rate(pred, np.argmax(tg_batch, axis=1))
            # dice  = self.compute_dice(pred, np.argmax(tg_batch,axis=1), self.n_out)
            print("     {"+str(err)+"}")

            ls_vx.append(vx_batch)
            if raw:
                ls_pred.append(pred_raw)
            else:
                ls_pred.append(pred)

        # Count the number of voxels
        n_vx = 0
        for vx in ls_vx:
            n_vx += vx.shape[0]

        # Aggregate the data
        vx_all = np.zeros((n_vx, 3), dtype=int)
        if raw:
            pred_all = np.zeros((n_vx, 135), dtype=float)
        else:
            pred_all = np.zeros((n_vx,), dtype=int)
        idx = 0
        for vx, pred in zip(ls_vx, ls_pred):
            next_idx = idx+vx.shape[0]
            vx_all[idx:next_idx] = vx
            pred_all[idx:next_idx] = pred
            idx = next_idx

        return vx_all, pred_all

class ConvNet_Dropout(Network):
    def __init__(self, n_out, input_var, target_var, patch_width, patch_width_comp, patch_width_3d, n_centroids):
        self.patch_width = patch_width
        self.patch_width_comp = patch_width_comp
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.split_idx = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.split_idx = np.cumsum(self.split_idx)
        Network.__init__(self, n_out, input_var, target_var)

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class ConvNet(Network):
    def __init__(self, n_out, input_var, target_var, patch_width, patch_width_comp, patch_width_3d, n_centroids):
        self.patch_width = patch_width
        self.patch_width_comp = patch_width_comp
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.split_idx = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.split_idx = np.cumsum(self.split_idx)
        Network.__init__(self, n_out, input_var, target_var)

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class VGGNet(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, pad="same")
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, pad="same")
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, pad="same")
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, pad="same")
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, pad="same")
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, pad="same")
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNet(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNet_Lg(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=50, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNetComp_Lg(Network):
    def __init__(self, n_out, input_var, target_var, patch_width_comp, patch_width_3d, n_centroids):
        self.patch_width_comp = patch_width_comp
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.split_idx = [0] + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.split_idx = np.cumsum(self.split_idx)
        Network.__init__(self, n_out, input_var, target_var)

    def build_net_virtual(self):
        vector_size = self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width_comp, self.patch_width_comp))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 3:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=50, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 4:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNet_Dropout(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNet_Multidropout(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = DropoutLayer(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = DropoutLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            if i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = DropoutLayer(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = DropoutLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                networks["patch_3d"] = DropoutLayer(networks["patch_3d"])
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        # net = DropoutLayer(merged_net)

        net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class ResNet(ConvNet):
    def projectionA(self, l_inp):
        n_filters = l_inp.output_shape[1]*2
        l = ExpressionLayer(l_inp, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], ceildiv(s[2], 2), ceildiv(s[3], 2)))
        l = PadLayer(l, [n_filters//4,0,0], batch_ndim=1)
        return l

    def projectionB(self, l_inp):
        # twice normal channels when projecting!
        n_filters = l_inp.output_shape[1]*2 
        l = Conv2DLayer(l_inp, num_filters=n_filters, filter_size=(1, 1),
                 stride=(2, 2), nonlinearity=None, pad='"same"', b=None)
        l = BatchNormLayer(l)
        return l

    # helper function to handle filters/strides when increasing dims
    def filters_increase_dims(self, l, increase_dims):
        in_num_filters = l.output_shape[1]
        if increase_dims:
            first_stride = (2, 2)
            out_num_filters = in_num_filters*2
        else:
            first_stride = (1, 1)
            out_num_filters = in_num_filters
 
        return out_num_filters, first_stride

    def res_block_v1(self, l_inp, nonlinearity=relu, increase_dim=False):
        projection = self.projectionB
        # first figure filters/strides
        n_filters, first_stride = self.filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
        l = Conv2DLayer(l_inp, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='"same"',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = BatchNormLayer(l)
        l = NonlinearityLayer(l, nonlinearity=nonlinearity)
        l = Conv2DLayer(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='"same"',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = BatchNormLayer(l)
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = ElemwiseSumLayer([l, p])
        l = NonlinearityLayer(l, nonlinearity=nonlinearity)
        return l

    def bottleneck_block_fast(self, l_inp, nonlinearity=relu, increase_dim=False):
        projection = self.projectionB
        # first figure filters/strides
        n_filters, last_stride = self.filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4 # as per bottleneck architecture used in paper
        scaled_filters = n_filters/scale
        l = Conv2DLayer(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='"same"',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = BatchNormLayer(l)
        l = NonlinearityLayer(l, nonlinearity=nonlinearity)
        l = Conv2DLayer(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='"same"',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = BatchNormLayer(l)
        l = NonlinearityLayer(l, nonlinearity=nonlinearity)
        l = Conv2DLayer(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=last_stride, nonlinearity=None, pad='"same"',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = ElemwiseSumLayer([l, p])
        l = NonlinearityLayer(l, nonlinearity=nonlinearity)
        return l

    def blockstack(self, l, n, res_block, nonlinearity=relu):
        for _ in range(n):
            l = res_block(l, nonlinearity=nonlinearity)
        return l

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}

        res_block = self.bottleneck_block_fast

        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                num_filters = 8
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))

                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=num_filters, stride=(1, 1), filter_size=(3, 3), nonlinearity=None, pad='"same"')
                networks["patch_2d_"+str(i)] = BatchNormLayer(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = NonlinearityLayer(networks["patch_2d_"+str(i)], nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))

                networks["patch_2d_"+str(i)] = self.blockstack(networks["patch_2d_"+str(i)], 3, res_block)
                networks["patch_2d_"+str(i)] = res_block(networks["patch_2d_"+str(i)], increase_dim=True)

                networks["patch_2d_"+str(i)] = self.blockstack(networks["patch_2d_"+str(i)], 2, res_block)

                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                networks["patch_3d"] = DenseLayer(networks["patch_3d"], num_units=1000, nonlinearity=relu)
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        # net = DenseLayer(merged_net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(merged_net, num_units=1000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

        print("ResNet structure: conv --> 3 res_block --> 3 res_block (7 layers in total)")

class GoogLeNet(ConvNet):
    def build_inception_module(self, input_layer, nfilters):
        # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
        net = {}
        net['pool'] = MaxPool2DLayer(input_layer, pool_size=3, stride=1, pad=1)
        net['pool_proj'] = Conv2DLayer(
            net['pool'], nfilters[0], 1, flip_filters=False)

        net['1x1'] = Conv2DLayer(input_layer, nfilters[1], 1, flip_filters=False)

        net['3x3_reduce'] = Conv2DLayer(
            input_layer, nfilters[2], 1, flip_filters=False)
        net['3x3'] = Conv2DLayer(
            net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

        net['5x5_reduce'] = Conv2DLayer(
            input_layer, nfilters[4], 1, flip_filters=False)
        net['5x5'] = Conv2DLayer(
            net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

        net['output'] = ConcatLayer([
            net['1x1'],
            net['3x3'],
            net['5x5'],
            net['pool_proj'],
            ])

        return net['output']

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = self.build_inception_module(networks["patch_2d_"+str(i)], [10, 10, 10, 20, 10, 20])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(3,3), stride=2)
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNet_SmCompFilter(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeUniform())
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeUniform())
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeUniform())
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeUniform())
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu, W=lasagne.init.HeUniform(gain='relu'))
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeUniform(gain='relu'))
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeUniform(gain='relu'))
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class Conv3DNet_HeNorm(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_3d"] = batch_norm(networks["patch_3d"])
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))

class Conv3DNet_NoCentroid(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_3d"] = batch_norm(networks["patch_3d"])
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))

class ConvNet_VerySmall(Network):
    def __init__(self, n_out, input_var, target_var, patch_width, patch_width_comp, patch_width_sm, n_centroids):
        self.patch_width = patch_width
        self.patch_width_comp = patch_width_comp
        self.patch_width_sm = patch_width_sm
        self.n_centroids = n_centroids

        self.split_idx = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_sm**2]*3 + [n_centroids]
        self.split_idx = np.cumsum(self.split_idx)
        Network.__init__(self, n_out, input_var, target_var)

    def build_inception_module(self, input_layer, nfilters):
        # nfilters: (1x1, 3x3)
        net = {}
        net['1x1'] = batch_norm(Conv2DLayer(
            input_layer, nfilters[0], 1, flip_filters=False, W=lasagne.init.HeNormal(gain='relu')))
        net['3x3'] = batch_norm(Conv2DLayer(
            input_layer, nfilters[1], 3, pad=1, flip_filters=False, W=lasagne.init.HeNormal(gain='relu')))

        net['output'] = ConcatLayer([
            net['1x1'],
            net['3x3'],
            ])

        return net['output']

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_sm**2*3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width_comp, self.patch_width_comp))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 6 and i < 9:
                networks["patch_2d_sm_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_sm_"+str(i)] = ReshapeLayer(networks["patch_2d_sm_"+str(i)], shape=([0], 1, self.patch_width_sm, self.patch_width_sm))
                print("patch_2d_sm_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_sm_"+str(i)].output_shape))
                networks["patch_2d_sm_"+str(i)] = self.build_inception_module(networks["patch_2d_sm_"+str(i)], [20,20])
                networks["patch_2d_sm_"+str(i)] = MaxPool2DLayer(networks["patch_2d_sm_"+str(i)], pool_size=(2,2))
                print("patch_2d_sm_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_sm_"+str(i)].output_shape))
                networks["patch_2d_sm_"+str(i)] = FlattenLayer(networks["patch_2d_sm_"+str(i)])
                print("patch_2d_sm_"+str(i)+" output_shape: "+str(networks["patch_2d_sm_"+str(i)].output_shape))
            elif i == 9:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))

class ConvNet_No3D(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            # elif i == 6:
            #     networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
            #     networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
            #     print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
            #     networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
            #     networks["patch_3d"] = batch_norm(networks["patch_3d"])
            #     networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
            #     print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
            #     networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
            #     print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))

class Conv3DNet_LgFilterSmCompFilter(ConvNet):
    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(9,9), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(5,5), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=50, filter_size=(3,3), nonlinearity=relu)
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu)
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu)
        self.net = DenseLayer(net, num_units=self.n_out, nonlinearity=softmax)

class SmallInception(ConvNet):
    def build_inception_module(self, input_layer, nfilters):
        # nfilters: (3x3, 5x5)
        net = {}
        net['3x3'] = batch_norm(Conv2DLayer(
            input_layer, nfilters[0], 3, pad=1, flip_filters=False, W=lasagne.init.HeNormal(gain='relu')))

        net['5x5'] = batch_norm(Conv2DLayer(
            input_layer, nfilters[1], 5, pad=2, flip_filters=False, W=lasagne.init.HeNormal(gain='relu')))

        net['output'] = ConcatLayer([
            net['3x3'],
            net['5x5'],
            ])

        return net['output']

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for i in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[i], self.split_idx[i+1])
            if i < 3:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = self.build_inception_module(networks["patch_2d_"+str(i)], [10,50])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i >= 3 and i < 6:
                networks["patch_2d_"+str(i)] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_2d_"+str(i)] = ReshapeLayer(networks["patch_2d_"+str(i)], shape=([0], 1, self.patch_width, self.patch_width))
                print("patch_2d_"+str(i)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = Conv2DLayer(networks["patch_2d_"+str(i)], num_filters=20, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_2d_"+str(i)] = batch_norm(networks["patch_2d_"+str(i)])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                networks["patch_2d_"+str(i)] = self.build_inception_module(networks["patch_2d_"+str(i)], [50,10])
                networks["patch_2d_"+str(i)] = MaxPool2DLayer(networks["patch_2d_"+str(i)], pool_size=(2,2))
                print("patch_2d_"+str(i)+" output_shape before flattening: "+str(networks["patch_2d_"+str(i)].output_shape))
                networks["patch_2d_"+str(i)] = FlattenLayer(networks["patch_2d_"+str(i)])
                print("patch_2d_"+str(i)+" output_shape: "+str(networks["patch_2d_"+str(i)].output_shape))
            elif i == 6:
                networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
                networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
                print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
                networks["patch_3d"] = batch_norm(networks["patch_3d"])
                networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
                print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
                networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
                print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
            elif i == 7:
                networks["centroids"] = SliceLayer(input_layer, indices=s, axis=1)
                print("centroids output_shape: "+str(networks["centroids"].output_shape))

        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))

class Inceptionv4(ConvNet):

    def conv_block(self, input_layer, num_filters, filter_size, stride, nonlinearity,  W=lasagne.init.HeNormal(gain='relu'), pad="valid"):
        x = Conv2DLayer(input_layer, num_filters=num_filters, filter_size=filter_size, stride=stride, nonlinearity=nonlinearity, pad = pad,  W=lasagne.init.HeNormal(gain='relu'))
       ## x = T.nnet.relu(x)
	#x = 0.5*(x+T.abs_(x))
	x = BatchNormLayer(x)
        return x

    ## build_inceptionC(..., (5,5))

    def build_stem(self, input_layer, indices, patch_iter):
        networks = {}
        networks["patch_2d_"+str(patch_iter)] = SliceLayer(input_layer, indices=indices, axis=1)
        networks["patch_2d_"+str(patch_iter)] = ReshapeLayer(networks["patch_2d_"+str(patch_iter)], shape=([0], 1, self.patch_width, self.patch_width))
        print("patch_2d_"+str(patch_iter)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(patch_iter)].output_shape))

        #input to 3 conv layers  -- Decreased stride to 1.
        networks["patch_2d_"+str(patch_iter)] = self.conv_block(input_layer=networks["patch_2d_"+str(patch_iter)], num_filters=32, filter_size=(3,3), stride=2, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks["patch_2d_"+str(patch_iter)] = self.conv_block(input_layer=networks["patch_2d_"+str(patch_iter)], num_filters=32, filter_size=(3,3), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks["conv_2d_"+str(patch_iter)] = self.conv_block(input_layer=networks["patch_2d_"+str(patch_iter)], num_filters=64, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
       
        #input to maxpool+conv
        networks["patch_2d(1)_"+str(patch_iter)] = MaxPool2DLayer(networks["conv_2d_"+str(patch_iter)], pool_size=(3,3), stride=1)
        networks["patch_2d(2)_"+str(patch_iter)] = Conv2DLayer(networks["conv_2d_"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("Path_2d(2) size: " + str( networks["patch_2d(2)_" + str(patch_iter)].output_shape)) 
        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["patch_2d(2)_"+str(patch_iter)], networks["patch_2d(1)_"+str(patch_iter)]])
       
        #ConvA
        networks['outputA'+str(patch_iter)] = self.conv_block(input_layer=networks["output"+str(patch_iter)], num_filters=64, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("OutputA shape 1 " + str(networks['outputA' + str(patch_iter)].output_shape))

        networks['outputA'+str(patch_iter)] = self.conv_block(input_layer=networks["outputA"+str(patch_iter)], num_filters=64, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("OutputA shape 2 " + str(networks['outputA' + str(patch_iter)].output_shape))
        networks['outputA'+str(patch_iter)] = self.conv_block(input_layer=networks["outputA"+str(patch_iter)], num_filters=64, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("OutputA shape 3 " + str(networks['outputA' + str(patch_iter)].output_shape))
        networks['outputA'+str(patch_iter)] = Conv2DLayer(networks["outputA"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1,nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
       
        #ConvB
        networks['outputB'+str(patch_iter)] = self.conv_block(input_layer=networks["output"+str(patch_iter)], num_filters=64, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['outputB'+str(patch_iter)] = Conv2DLayer(networks["outputB"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #concatenateAB
        networks['output'+str(patch_iter)] = ConcatLayer([networks["outputA"+str(patch_iter)], networks['outputB'+ str(patch_iter)]])
       
        #Maxpool+conv
        networks['outputA'+str(patch_iter)] = MaxPool2DLayer(networks["output"+str(patch_iter)], pool_size=(3,3), stride=2)
        networks['outputB'+str(patch_iter)] = Conv2DLayer(networks["output"+str(patch_iter)], num_filters=192, filter_size=(3,3), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["outputA"+str(patch_iter)], networks["outputB"+str(patch_iter)]])
        print("Output shape  " + str(networks['output' + str(patch_iter)].output_shape))


        return networks['output'+str(patch_iter)]

    def build_inceptionA(self, stem_output, patch_iter):
        networks = {}
        
        #inceptionMoDA
        #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=stem_output, num_filters=64, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch2
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=stem_output, num_filters=64, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_2"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_2 shape: " + str(networks['branch_2' + str(patch_iter)].output_shape))

        #branch3
        networks['branch_3'+str(patch_iter)] = self.conv_block(input_layer=stem_output, num_filters=96, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_3 shape: " + str(networks['branch_3' + str(patch_iter)].output_shape))
        
        #branch4
        networks['branch_4'+str(patch_iter)] = Pool2DLayer(stem_output, pool_size=(3,3), stride=1, mode='average_inc_pad')
        print("branch_4 shape: " + str(networks['branch_4' + str(patch_iter)].output_shape))
        networks['branch_4'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_4"+str(patch_iter)], num_filters=96, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
    
        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_2"+str(patch_iter)], networks["branch_3"+str(patch_iter)], networks["branch_4"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_reductionA(self, outputModA, patch_iter):
        networks = {}
        
         #reductionA 35x35->17x17
         #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputModA, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu')) 
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=224, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch2
        networks['branch_2'+str(patch_iter)] = Conv2DLayer(outputModA, num_filters=384, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch3
        networks['branch_3'+str(patch_iter)] = MaxPool2DLayer(outputModA, pool_size=(3,3), stride=2)
        
        #concatentate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_2"+str(patch_iter)], networks["branch_3"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_inceptionB(self, outputRedA, patch_iter):
        networks = {}

        #inceptionMoDB
        #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputRedA, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=192, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=224, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=224, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch2
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=outputRedA, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_2"+str(patch_iter)], num_filters=224, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_2'+str(patch_iter)] = Conv2DLayer(networks["branch_2"+str(patch_iter)], num_filters=256, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch3
        networks['branch_3'+str(patch_iter)] = Conv2DLayer(outputRedA, num_filters=384, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch4
        networks['branch_4'+str(patch_iter)] = Pool2DLayer(outputRedA, pool_size=(3,3), stride=1, pad=(0,0), mode='average_inc_pad')
        networks['branch_4'+str(patch_iter)] = Conv2DLayer(networks["branch_4"+str(patch_iter)], num_filters=128, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_2"+str(patch_iter)], networks["branch_3"+str(patch_iter)], networks["branch_4"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_reductionB(self, outputModB, patch_iter):
        networks = {}
        
        #reductionB 17x17->8x8
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=256, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu')) 
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=320, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=320, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch2
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_2'+str(patch_iter)] = Conv2DLayer(networks["branch_2"+str(patch_iter)], num_filters=192, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch3
        networks['branch_3'+str(patch_iter)] = MaxPool2DLayer(outputModB, pool_size=(3,3), stride=2)
        
        #concatentate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_2"+str(patch_iter)], networks["branch_3"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    
    def build_inceptionC(self, outputModB, patch_iter):
        networks = {}
        
        #inceptionMoDC
        #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=384, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=448, filter_size=(1,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=512, filter_size=(3,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_11'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=filter_size_for_valid_padded_convs, stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_12'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(3,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch2
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=384, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_21'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_2"+str(patch_iter)], num_filters=256, filter_size=(3,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_22'+str(patch_iter)] = Conv2DLayer(networks["branch_2"+str(patch_iter)], num_filters=256, filter_size=(1,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch3
        networks['branch_3'+str(patch_iter)] = Conv2DLayer(outputModB, num_filters=256, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch4
        networks['branch_4'+str(patch_iter)] = Pool2DLayer(outputModB, pool_size=(3,3), stride=1, pad="same", mode='average_inc_pad')
        networks['branch_4'+str(patch_iter)] = Conv2DLayer(networks["branch_4"+str(patch_iter)], num_filters=256, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_11"+str(i)], networks["branch_12"+str(patch_iter)], networks["branch_21"+str(patch_iter)], networks["branch_22"+str(patch_iter)], networks["branch_3"+str(patch_iter)], networks["branch_4"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_averagepool(self, outputModC, patch_iter):
   
    #averagepool
        networks = {}
        networks["output"+str(patch_iter)] = Pool2DLayer(outputModC, pool_size=(8,8), pad="valid", mode='average_inc_pad')
        networks['output'+str(patch_iter)] = DropoutLayer(networks["output"+str(i)], p=0.8)
        print("output"+str(patch_iter)+" output_shape before flattening: "+str(networks["output"+str(i)].output_shape))
        networks["output"+str(patch_iter)] = FlattenLayer(networks["output"+str(i)])
        print("output"+str(patch_iter)+" output_shape: "+str(networks["output"+str(i)].output_shape))

        return networks['output'+str(patch_iter)]
    
    def build_3d_patch_layer(self, input_layer, indices):
        networks = {}
        networks["patch_3d"] = SliceLayer(input_layer, indices=s, axis=1)
        networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
        print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
        networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks["patch_3d"] = batch_norm(networks["patch_3d"])
        networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
        print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
        networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
        print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
        
        return networks["patch_3d"]

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for patch_iter in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[patch_iter], self.split_idx[patch_iter+1])
            if patch_iter < 6:
                networks['stem_layer'+str(patch_iter)] = self.build_stem(input_layer, s, patch_iter)
                #str(i)
                inceptionA_out = None
                for j in xrange(0, 4):
                    if j == 0:  #  conv_block -> self.conv_block build_inceptionA -> self.build_inceptionA
                        inceptionA_out = self.build_inceptionA(networks['stem_layer'+str(patch_iter)], patch_iter)
                    else:
                        inceptionA_out = self.build_inceptionA(inceptionA_out, patch_iter)
                networks['inception_A'+str(patch_iter)] = self.build_reductionA(inceptionA_out, patch_iter)
                
                inceptionB_out = None
                for j in xrange(0, 7):
                    if j == 0:
                        inceptionB_out = self.build_inceptionB(networks['inception_A'+str(patch_iter)], patch_iter)
                    else:
                        inceptionB_out = self.build_inceptionB(inceptionB_out, patch_iter)
                networks['inception_B'+str(patch_iter)] = self.build_reductionB(inceptionB_out, patch_iter)
                
                inceptionC_out = None
                for j in xrange(0, 3):
                    if j == 0:
                        inceptionC_out = self.build_inceptionC(networks['inception_B'+str(patch_iter)], patch_iter)
                    else:
                        inceptionC_out = self.build_inceptionC(inceptionC_out, patch_iter)
                networks['inception_C'+str(patch_iter)] = self.build_averagepool(inceptionC_out, patch_iter)
            elif patch_iter == 6:
                networks['patch_3d'] =self.build_3d_patch_layer(input_layer)
        networks_list = []
        for key, value in networks.iteritems():
            networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))
  
                
                
                
class Inceptionv4Simple(ConvNet):

    def conv_block(self, input_layer, num_filters, filter_size, stride, nonlinearity,  W=lasagne.init.HeNormal(gain='relu'), pad="valid"):
        x = Conv2DLayer(input_layer, num_filters=num_filters, filter_size=filter_size, stride=stride, nonlinearity=nonlinearity, pad = pad,  W=lasagne.init.HeNormal(gain='relu'))
        x = NonlinearityLayer(x, nonlinearity=relu)
	x = BatchNormLayer(x)
        ##x = NonlinearityLayer(x, nonlinearity=relu)
        return x

    ## build_inceptionC(..., (5,5))

    def build_stem(self, input_layer, indices, patch_iter):
        networks = {}
        networks["patch_2d_"+str(patch_iter)] = SliceLayer(input_layer, indices=indices, axis=1)
        networks["patch_2d_"+str(patch_iter)] = ReshapeLayer(networks["patch_2d_"+str(patch_iter)], shape=([0], 1, self.patch_width, self.patch_width))
        print("patch_2d_"+str(patch_iter)+" output_shape after reshaping: "+str(networks["patch_2d_"+str(patch_iter)].output_shape))

        #input to 3 conv layers  -- Decreased stride to 1.
        networks["patch_2d_"+str(patch_iter)] = self.conv_block(input_layer=networks["patch_2d_"+str(patch_iter)], num_filters=32, filter_size=(3,3), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks["conv_2d_"+str(patch_iter)] = self.conv_block(input_layer=networks["patch_2d_"+str(patch_iter)], num_filters=64, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
       
        #input to maxpool+conv
        networks["patch_2d(1)_"+str(patch_iter)] = Conv2DLayer(networks["conv_2d_"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks["patch_2d(2)_"+str(patch_iter)] = MaxPool2DLayer(networks["conv_2d_"+str(patch_iter)], pool_size=(3,3), stride=1)
        print("Path_2d(2) size: " + str( networks["patch_2d(2)_" + str(patch_iter)].output_shape)) 

        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["patch_2d(1)_"+str(patch_iter)], networks["patch_2d(2)_"+str(patch_iter)]])
        print("Output shape  " + str(networks['output' + str(patch_iter)].output_shape))

        return networks['output'+str(patch_iter)]

    def build_inceptionA(self, stem_output, patch_iter):
        networks = {}
        
        #inceptionMoDA
        #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=stem_output, num_filters=64, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        
        #branch2
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=stem_output, num_filters=64, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_2"+str(patch_iter)], num_filters=96, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
       # print("branch_2 shape: " + str(networks['branch_2' + str(patch_iter)].output_shape))

        #branch3
        #networks['branch_3'+str(patch_iter)] = self.conv_block(input_layer=stem_output, num_filters=96, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
       # print("branch_3 shape: " + str(networks['branch_3' + str(patch_iter)].output_shape))
        
        #branch4
        networks['branch_4'+str(patch_iter)] = Pool2DLayer(stem_output, pool_size=(3,3), stride=1, mode='average_inc_pad', pad=(1, 1))
        print("branch_4 shape: " + str(networks['branch_4' + str(patch_iter)].output_shape))

        networks['branch_4'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_4"+str(patch_iter)], num_filters=96, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
    
        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_2"+str(patch_iter)], networks["branch_4"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_reductionA(self, outputModA, patch_iter):
        networks = {}
        
         #reductionA 35x35->17x17
         #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputModA, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu')) 
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=224, filter_size=(3,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_1 redA shape: " + str(networks['branch_1' + str(patch_iter)].output_shape))

        #branch2
        networks['branch_2'+str(patch_iter)] = Conv2DLayer(outputModA, num_filters=384, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_2 redA shape: " + str(networks['branch_2' + str(patch_iter)].output_shape))

        #branch3
       # networks['branch_3'+str(patch_iter)] = MaxPool2DLayer(outputModA, pool_size=(3,3), stride=2)
       # print("branch_3 redA shape: " + str(networks['branch_3' + str(patch_iter)].output_shape))

        #concatentate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_2"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_inceptionB(self, outputRedA, patch_iter):
        networks = {}

        #inceptionMoDB
        #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputRedA, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=192, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=224, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=224, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_1 B shape: " + str(networks['branch_1' + str(patch_iter)].output_shape))

        #branch2
      #  networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=outputRedA, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
      #  networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_2"+str(patch_iter)], num_filters=224, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
     #   networks['branch_2'+str(patch_iter)] = Conv2DLayer(networks["branch_2"+str(patch_iter)], num_filters=256, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
     #   print("branch_2 B shape: " + str(networks['branch_2' + str(patch_iter)].output_shape))

        #branch3
        networks['branch_3'+str(patch_iter)] = Conv2DLayer(outputRedA, num_filters=384, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_3 B shape: " + str(networks['branch_3' + str(patch_iter)].output_shape))

        #branch4
        networks['branch_4'+str(patch_iter)] = Pool2DLayer(outputRedA, pool_size=(3,3), stride=1, pad=(1,1), mode='average_inc_pad')
        networks['branch_4'+str(patch_iter)] = Conv2DLayer(networks["branch_4"+str(patch_iter)], num_filters=128, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_4 B shape: " + str(networks['branch_4' + str(patch_iter)].output_shape))

        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_3"+str(patch_iter)], networks["branch_4"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_reductionB(self, outputModB, patch_iter):
        networks = {}
        
        #reductionB 17x17->8x8
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=256, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu')) 
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(1,7), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=320, filter_size=(7,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=320, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_1 redB shape: " + str(networks['branch_1' + str(patch_iter)].output_shape))

        #branch2
      #  networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=192, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
      #  networks['branch_2'+str(patch_iter)] = Conv2DLayer(networks["branch_2"+str(patch_iter)], num_filters=192, filter_size=(3,3), stride=2, pad="valid", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        #print("branch_2 redB shape: " + str(networks['branch_2' + str(patch_iter)].output_shape))

        #branch3
        networks['branch_3'+str(patch_iter)] = MaxPool2DLayer(outputModB, pool_size=(3,3), stride=2)
        print("branch_3 redB shape: " + str(networks['branch_3' + str(patch_iter)].output_shape))

        #concatentate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_1"+str(patch_iter)], networks["branch_3"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    
    def build_inceptionC(self, outputModB, patch_iter):
        networks = {}
        
        #inceptionMoDC
        #branch1
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=384, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=448, filter_size=(1,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_1'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=512, filter_size=(3,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_11'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(1,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks['branch_12'+str(patch_iter)] = Conv2DLayer(networks["branch_1"+str(patch_iter)], num_filters=256, filter_size=(3,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_11 C shape: " + str(networks['branch_11' + str(patch_iter)].output_shape))
        print("branch_12 C shape: " + str(networks['branch_12' + str(patch_iter)].output_shape))

        #branch2
       # networks['branch_2'+str(patch_iter)] = self.conv_block(input_layer=outputModB, num_filters=384, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
      #  networks['branch_21'+str(patch_iter)] = self.conv_block(input_layer=networks["branch_2"+str(patch_iter)], num_filters=256, filter_size=(3,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
      #  networks['branch_22'+str(patch_iter)] = Conv2DLayer(networks["branch_2"+str(patch_iter)], num_filters=256, filter_size=(1,3), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
      #  print("branch_22 C shape: " + str(networks['branch_22' + str(patch_iter)].output_shape))

        #branch3
        networks['branch_3'+str(patch_iter)] = Conv2DLayer(outputModB, num_filters=256, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_3 C shape: " + str(networks['branch_3' + str(patch_iter)].output_shape))

        #branch4
        networks['branch_4'+str(patch_iter)] = Pool2DLayer(outputModB, pool_size=(3,3), stride=1, pad=(1,1), mode='average_inc_pad')
        networks['branch_4'+str(patch_iter)] = Conv2DLayer(networks["branch_4"+str(patch_iter)], num_filters=256, filter_size=(1,1), stride=1, pad="same", nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        print("branch_4 C  shape: " + str(networks['branch_4' + str(patch_iter)].output_shape))

        #concatenate
        networks['output'+str(patch_iter)] = ConcatLayer([networks["branch_11"+str(patch_iter)], networks["branch_12"+str(patch_iter)], networks["branch_3"+str(patch_iter)], networks["branch_4"+str(patch_iter)]])

        return networks['output'+str(patch_iter)]

    def build_averagepool(self, outputModC, patch_iter):
   
    #averagepool
        networks = {}
        networks["output"+str(patch_iter)] = Pool2DLayer(outputModC, pool_size=(8,8), pad=(7,7), mode='average_inc_pad')
        print("output_pool_layer  shape: " + str(networks['output' + str(patch_iter)].output_shape))

        networks['output'+str(patch_iter)] = DropoutLayer(networks["output"+str(patch_iter)], p=0.8)
        print("output"+str(patch_iter)+" output_shape before flattening: "+str(networks["output"+str(patch_iter)].output_shape))
        networks["output"+str(patch_iter)] = FlattenLayer(networks["output"+str(patch_iter)])
        print("output"+str(patch_iter)+" output_shape: "+str(networks["output"+str(patch_iter)].output_shape))

        return networks['output'+str(patch_iter)]
    
    def build_3d_patch_layer(self, input_layer, indices):
        networks = {}
        networks["patch_3d"] = SliceLayer(input_layer, indices=indices, axis=1)
        networks["patch_3d"] = ReshapeLayer(networks["patch_3d"], shape=([0], 1, self.patch_width_3d, self.patch_width_3d, self.patch_width_3d))
        print("patch_3d output_shape after reshaping: "+str(networks["patch_3d"].output_shape))
       # networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(3,3,3), stride=2, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
       # networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(3,3,3), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        networks["patch_3d"] = Conv3DLayer(networks["patch_3d"], num_filters=20, filter_size=(5,5,5), stride=1, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))

        print("patch_3d_conv shape: " +str(networks["patch_3d"].output_shape))
        networks["patch_3d"] = batch_norm(networks["patch_3d"])
        networks["patch_3d"] = MaxPool3DLayer(networks["patch_3d"], pool_size=2)
        print("patch_3d output_shape before flattening: "+str(networks["patch_3d"].output_shape))
        networks["patch_3d"] = FlattenLayer(networks["patch_3d"])
        print("patch_3d output_shape: "+str(networks["patch_3d"].output_shape))
        
        return networks["patch_3d"]

    def build_net_virtual(self):
        vector_size = self.patch_width**2*3 + self.patch_width_comp**2*3 + self.patch_width_3d**3 + self.n_centroids
        input_layer = InputLayer(shape=(None, vector_size), input_var=self.input_var)

        networks = {}
        for patch_iter in xrange(len(self.split_idx) - 1):
            s = slice(self.split_idx[patch_iter], self.split_idx[patch_iter+1])
            if patch_iter < 6:
                networks['stem_layer'+str(patch_iter)] = self.build_stem(input_layer, s, patch_iter)
                #str(i)
                inceptionA_out = None
                for j in xrange(0, 1):
                    if j == 0:  #  conv_block -> self.conv_block build_inceptionA -> self.build_inceptionA
                        inceptionA_out = self.build_inceptionA(networks['stem_layer'+str(patch_iter)], patch_iter)
                    else:
                        inceptionA_out = self.build_inceptionA(inceptionA_out, patch_iter)
                networks['inception_A'+str(patch_iter)] = self.build_reductionA(inceptionA_out, patch_iter)
                
                inceptionB_out = None
                for j in xrange(0, 1):
                    if j == 0:
                        inceptionB_out = self.build_inceptionB(networks['inception_A'+str(patch_iter)], patch_iter)
                    else:
                        inceptionB_out = self.build_inceptionB(inceptionB_out, patch_iter)
                networks['inception_B'+str(patch_iter)] = self.build_reductionB(inceptionB_out, patch_iter)
                
                inceptionC_out = None
                for j in xrange(0, 1):
                    if j == 0:
                        inceptionC_out = self.build_inceptionC(networks['inception_B'+str(patch_iter)], patch_iter)
                    else:
                        inceptionC_out = self.build_inceptionC(inceptionC_out, patch_iter)
                networks['inception_C'+str(patch_iter)] = self.build_averagepool(inceptionC_out, patch_iter)
            elif patch_iter == 6:
               networks['patch_3d'] = self.build_3d_patch_layer(input_layer, s)
        networks_list = []
        for key, value in networks.iteritems():
            if 'inception_C' in key or 'patch_3d' in key:
                networks_list.append(value)

        merged_net = ConcatLayer(networks_list, axis=1)

        print("merged_net output shape: "+str(merged_net.output_shape))

        net = DropoutLayer(merged_net)

        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        net = DenseLayer(net, num_units=3000, nonlinearity=relu, W=lasagne.init.HeNormal(gain='relu'))
        net = batch_norm(net)
        self.net = batch_norm(DenseLayer(net, num_units=self.n_out, nonlinearity=softmax, W=lasagne.init.HeNormal(gain='relu')))
  
                
                
                
