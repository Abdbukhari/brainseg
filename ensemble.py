from spynet.models.network import *
from spynet.utils.utilities import create_img_from_pred, compute_dice_symb, compute_dice, error_rate
import copy

class Ensemble(object):
    '''
    	Class that ensemble several neural network model
    '''
    def __init__(self, nets):
        self.name = self.__class__.__name__

        self.nets = nets

    def predict(self, data_gen, atlas_id, n_points, region_centroid, batch_size, scaler, pred_functions, verbose=False):
        pred_raws = []

        for i, net in enumerate(self.nets):
            print("\n============================================================\n")
            print("Model " + str(i+1) + " starts evaluating the whole brain\n")
            brain_batches = data_gen.generate_single_atlas(atlas_id, n_points, region_centroid, batch_size, verbose)
            pred_functions_c = copy.copy(pred_functions)
            vx_all, pred_raw = net.predict_from_generator(brain_batches, scaler, pred_functions_c, True)
            pred_raws.append(pred_raw)

        pred_raws = np.array(pred_raws)

        self.pred_raws = pred_raws
        self.vx_all = vx_all

        pred_all = self.pred_raws.sum(axis=0).argmax(axis=1)

        return vx_all, pred_all

    def stat_of_all_models(self, img_true, n_out):
        '''
            return two lists of dice coefficient and error of all models in the ensemble
        '''
        dices = []
        errs = []
        count = 1
        for pred_raw in self.pred_raws:
            # Compute img_pred
            pred_all = np.argmax(pred_raw, axis=1)
            img_pred = create_img_from_pred(self.vx_all, pred_all, img_true.shape)

            # Compute the dice coefficient and the error
            non_zo = img_pred.nonzero() or img_true.nonzero()
            pred = img_pred[non_zo]
            true = img_true[non_zo]
            dice_regions = compute_dice(pred, true, n_out)
            dices.append(dice_regions.mean())
            err_global = error_rate(pred, true)
            errs.append(err_global)
            count = count + 1

        return dices, errs






