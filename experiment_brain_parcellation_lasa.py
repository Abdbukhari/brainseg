__author__ = 'adeb'

import sys
import datetime
from shutil import copy2
import inspect
import PIL
import pickle
from spynet.utils.utilities import analyse_classes
from data_brain_parcellation import DatasetBrainParcellation
from network_brain_parcellation import *
from spynet.models.network import *
from spynet.models.neuron_type import *
from spynet.data.dataset import *
from spynet.training.trainer import *
from spynet.training.monitor import *
from spynet.training.parameters_selector import *
from spynet.training.stopping_criterion import *
from spynet.training.cost_function import *
from spynet.training.learning_update import *
from spynet.experiment import Experiment
from spynet.utils.utilities import tile_raster_images

import numpy as np

import lasagne
from lasagne_training.network import *
from lasagne_training.trainer import *

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save

import theano


class ExperimentBrain(Experiment):
    """
    Main experiment to train a network on a dataset
    """
    def __init__(self, exp_name, data_path):
        Experiment.__init__(self, exp_name, data_path)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):
        ###### Create the datasets

        # aa = CostNegLLWeighted(np.array([0.9, 0.1]))
        # e = theano.function(inputs=[], outputs=aa.test())
        # print e()

        ## Load the data
        training_data_path = self.data_path + "train.h5"
        ds_training = DatasetBrainParcellation()
        ds_training.read(training_data_path)

        ds_training, ds_validation = ds_training.split_dataset_proportions([0.95, 0.05])

        testing_data_path = self.data_path + "test.h5"
        ds_testing = DatasetBrainParcellation()
        ds_testing.read(testing_data_path)

        ## Display data sample
        # image = PIL.Image.fromarray(tile_raster_images(X=ds_training.inputs[0:50],
        #                                                img_shape=(29, 29), tile_shape=(5, 10),
        #                                                tile_spacing=(1, 1)))
        # image.save(self.path + "filters_corruption_30.png")

        ## Few stats about the targets
        classes, proportion_class = analyse_classes(np.argmax(ds_training.outputs, axis=1), "Training data:")
        print(classes)
        ## Scale some part of the data
        print("Scaling")
        s = Scaler([slice(-134, None, None)])
        s.compute_parameters(ds_training.inputs)
        s.scale(ds_training.inputs)
        s.scale(ds_validation.inputs)
        s.scale(ds_testing.inputs)
        pickle.dump(s, open(self.path + "s.scaler", "wb"))

        ###### Create the network
        input_var = T.matrix('inputs')
        target_var = T.matrix('targets')
        # net = ConvNet(135, input_var, target_var, 29, 29, 13, 134)
        # net = ResNet(135, input_var, target_var, 29, 29, 13, 134)
        # net = VGGNet(135, input_var, target_var, 29, 29, 13, 134)
        # net = Conv3DNet_Multidropout(135, input_var, target_var, 29, 29, 13, 134)
        # net = Conv3DNetComp_Lg(135, input_var, target_var, 29, 29, 134)
        # net = GoogLeNet(135, input_var, target_var, 29, 29, 13, 134)
        # net = Conv3DNet_SmCompFilter(135, input_var, target_var, 29, 29, 13, 134)
        # net = Conv3DNet_HeNorm(135, input_var, target_var, 29, 29, 13, 134)
        # net = SmallInception(135, input_var, target_var, 29, 29, 13, 134)
        # net = Conv3DNet_NoCentroid(135, input_var, target_var, 29, 29, 13, 134)
        # net = ConvNet_VerySmall(135, input_var, target_var, 29, 29, 13, 134)
        #try:
        net = Inceptionv4Simple(135, input_var, target_var, 29, 29, 13, 134)
       # except Exception as e:
           # print("Program terminated at: " + datetime.datetime.now())
          #  print(e)

        # print net.net
        learning_rate = 0.045
        # learning_rate = 0.0001

        # Create stopping criteria and add them to the trainer
        max_epoch = 15
        # early_stopping = EarlyStopping(err_validation, 10, 0.99, 5)

        # Create the trainer object
        batch_size = 50
        t = Trainer(net, ds_testing, ds_validation, ds_training, batch_size, learning_rate)

        ###### Train the network

        t.train()

        ###### Plot the records

        # pred = np.argmax(t.net.predict(ds_testing.inputs, 10000), axis=1)
        # d = compute_dice(pred, np.argmax(ds_testing.outputs, axis=1), 134)
        # print "Dice test: {}".format(np.mean(d))
        # print "Error rate test: {}".format(error_rate(np.argmax(ds_testing.outputs, axis=1), pred))

        save_records_plot(self.path, [t.val_errs, t.test_errs], ["validation", "test", "Error"], "upper right")
        save_records_plot(self.path, [t.val_dices, t.test_dices], ["validation", "test", "Dice coefficient"], "lower right")

        ###### Save the network
        np.savez(self.path+'model.npz', *t.best_net_param)
        print("network model saved")
        # net.save_parameters(self.path + "net.net")

def save_records_plot(file_path, stats, names, legend_loc="upper right"):
    """
    Save a plot of a list of monitors' history.
    Args:
        file_path (string): the folder path where to save the plot
        ls_monitors: the list of statistics to plot
        name: name of file to be saved
        n_train_batches: the total number of training batches
    """

    lines = ["--", "-", "-.",":"]
    linecycler = cycle(lines)

    plt.figure()
    for i, s in enumerate(stats):
        X = range(1,len(s)+1)
        Y = s
        a, b = zip(*sorted(zip(X, Y)))
        plt.plot(a, b, next(linecycler), label=names[i])

    plt.xlabel('Training epoch')
    plt.ylabel(names[len(names)-1])
    plt.legend(loc=legend_loc)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=10)
    plt.savefig(file_path + names[len(names)-1] + ".png")
    tikz_save(file_path + names[len(names)-1] + ".tikz", figureheight = '\\figureheighttik', figurewidth = '\\figurewidthtik')


if __name__ == '__main__':

    if (len(sys.argv) == 0):
        exp_name = "paper_ultimate_conv"
    elif (not len(sys.argv) == 2):
        print("You cannot enter more than one option")
        sys.exit(2)
    else:
        exp_name = sys.argv[1]
    # data_path = "./datasets/"+exp_name+"/"
    data_path = "./datasets/paper_ultimate_conv/"

    exp = ExperimentBrain(exp_name, data_path)
    exp.run()
