__author__ = 'adeb'

import os
import imp
import time
import glob
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import nibabel
import theano
import theano.tensor as T
import scipy.io

from spynet.utils.utilities import create_img_from_pred, compute_dice_symb, compute_dice, error_rate
import spynet.training.trainer as trainer
from spynet.models.network import *
from network_brain_parcellation import *
from spynet.data.utils_3d.pick_voxel import *
from spynet.data.utils_3d.pick_patch import *
from spynet.data.utils_3d.pick_target import *
from data_brain_parcellation import DatasetBrainParcellation, DataGeneratorBrain, list_miccai_files, RegionCentroids
from spynet.utils.utilities import open_h5file
from ensemble import Ensemble


if __name__ == '__main__':
    """
    Compute the segmentations of the testing brains with the trained networks (with approximation of the centroids)
    """

    experiment_path = "./experiments/ensemble/"
    data_path = "./datasets/paper_ultimate_conv/"
    cf_data = imp.load_source("cf_data", data_path + "cfg_testing_data_creation.py")

    nets = []
    net_paths = glob.glob(experiment_path + "*/")
    # Load the networks
    for net_path in net_paths:
        net = NetworkConvDropout()
        net.init(29, 29, 13, 134, 135)
        net.load_parameters(open_h5file(net_path + "net.net"))
        n_out = net.n_out
        nets.append(net)

    ensemble_net = Ensemble(nets)


    # Load the scaler
    scaler = pickle.load(open(net_paths[0] + "s.scaler", "rb"))

    # Files on which to evaluate the network
    file_list = list_miccai_files(**{"mode": "folder", "path": "./datasets/miccai/2/"})
    n_files = len(file_list)

    # Options for the generation of the dataset
    # The generation/evaluation of the dataset has to be split into batches as a whole brain does not fit into memory
    batch_size = 100000
    select_region = SelectWholeBrain()
    extract_vx = ExtractVoxelAll(1)
    pick_vx = PickVoxel(select_region, extract_vx)
    pick_patch = create_pick_features(cf_data)
    pick_tg = create_pick_target(cf_data)

    # Create the data generator
    data_gen = DataGeneratorBrain()
    data_gen.init_from(file_list, pick_vx, pick_patch, pick_tg)

    # Evaluate the centroids
    net_wo_centroids_path = "./experiments/noCentroid/"
    net_wo_centroids = NetworkWithoutCentroidConv()
    net_wo_centroids.init(29, 13, 135)
    net_wo_centroids.load_parameters(open_h5file(net_wo_centroids_path + "net.net"))
    ds_testing = DatasetBrainParcellation()
    ds_testing.read(data_path + "train.h5")
    pred_wo_centroids = np.argmax(net_wo_centroids.predict(ds_testing.inputs, 1000), axis=1)
    region_centroids = RegionCentroids(134)
    region_centroids.update_barycentres(ds_testing.vx, pred_wo_centroids)

    # Generate and evaluate the dataset
    dices = np.zeros((n_files, 134))
    errors = np.zeros((n_files,))

    pred_functions = {}
    dices_mean = []
    for atlas_id in xrange(n_files):
    # for atlas_id in xrange(1):
        start_time = time.clock()

        print "Atlas: {}".format(atlas_id)

        # brain_batches = data_gen.generate_single_atlas(atlas_id, None, region_centroids, batch_size, True)

        # vx_all, pred_all = net.predict_from_generator(brain_batches, scaler, pred_functions)
        vx_all, pred_all = ensemble_net.predict(data_gen, atlas_id, None, region_centroids, batch_size, scaler, pred_functions, True)

        # Construct the predicted image
        img_true = data_gen.atlases[atlas_id][1]
        img_pred = create_img_from_pred(vx_all, pred_all, img_true.shape)

        # Compute the dice coefficient and the error
        non_zo = img_pred.nonzero() or img_true.nonzero()
        pred = img_pred[non_zo]
        true = img_true[non_zo]

        dice_regions = compute_dice(pred, true, n_out)
        err_global = error_rate(pred, true)

        dices_all, errs = ensemble_net.stat_of_all_models(img_true, n_out)

        end_time = time.clock()
        print("\n============================================================\n")
        print "It took {} seconds to evaluate the whole brain.".format(end_time - start_time)
        print "The mean dices of each individual model are as follow:"
        for i, dice in enumerate(dices_all):
            print("Model " + str(i+1) + ": " + str(dice) )
        print "The error rates of each individual model are as follow:"
        for i, err in enumerate(errs):
            print("Model " + str(i+1) + ": " + str(err) )
        print "The overall mean dice is {}".format(dice_regions.mean())
        print "The overall error rate is {}".format(err_global)

        # Save the results
        errors[atlas_id] = err_global
        dices[atlas_id, :] = dice_regions
        dices_mean.append(dice_regions.mean())

        # Diff Image
        img_diff = (img_pred == img_true).astype(np.uint8)
        img_diff += 1
        img_diff[img_pred == 0] = 0

        # Save the 3D images
        affine = data_gen.atlases[atlas_id][2]
        mri_file = data_gen.files[atlas_id][0]
        mri = data_gen.atlases[atlas_id][0]
        basename = os.path.splitext(os.path.basename(mri_file))[0]
        img_pred_nifti = nibabel.Nifti1Image(img_pred, affine)
        img_mri_nifti = nibabel.Nifti1Image(mri, affine)
        img_true_nifti = nibabel.Nifti1Image(img_true, affine)
        img_diff_nifti = nibabel.Nifti1Image(img_diff, affine)
        nibabel.save(img_pred_nifti, experiment_path + basename + "_pred.nii")
        nibabel.save(img_mri_nifti, experiment_path + basename + "_mri.nii")
        nibabel.save(img_true_nifti, experiment_path + basename + "_true.nii")
        nibabel.save(img_diff_nifti, experiment_path + basename + "_diff.nii")

    scipy.io.savemat(experiment_path + "dice_brains.mat", mdict={'arr': dices})
    scipy.io.savemat(experiment_path + "error_brains.mat", mdict={'arr': errors})

    dices_mean = np.array(dices_mean)
    print("The mean dice coefficient = " + str(dices_mean.mean()) + " +/-" + str(dices_mean.std()))

