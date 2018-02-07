__author__ = 'adeb'

from spynet.models.layer_block import *
import spynet.models.neuron_type as neuron_type
from spynet.models.layer import *
from spynet.models.network import Network


class NetworkOnePatchConv(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None

    def init(self, patch_width, n_out):
        Network.init_common(self, patch_width**2, n_out)

        self.patch_width = patch_width

        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))


        # Layer 1
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        # Layer 2
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = n_kern1 * filter_map_height2 * filter_map_with2
        n_out2 = 1000
        block2 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

        # Layer 3
        block3 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

        self.ls_layers = convert_blocks_into_feed_forward_layers([block0, block1, block2, block3])

        self.update_params()

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)


class NetworkOnePatchConvCentroids(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None

    def init(self, patch_width, n_centroids, n_out):
        Network.init_common(self, patch_width**2 + n_centroids, n_out)

        self.patch_width = patch_width
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures([0, interval, self.n_in]))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))
        block1 = LayerBlockIdentity()
        self.ls_layers.append(LayerOfBlocks([block0, block1]))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockIdentity()
        self.ls_layers.append(LayerOfBlocks([block0, block1]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = n_kern1 * filter_map_height2 * filter_map_with2 + n_centroids
        n_out2 = 1000
        block2 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block2]))

        # Layer 5
        n_out3 = 1000
        block2 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block2]))

        # Layer 6
        block3 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block3]))

        self.update_params()

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)





class NetworkCentroids(Network):
    def __init__(self):
        Network.__init__(self)

    def init(self, n_in, n_out):
        Network.init_common(self, n_in, n_out)

        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        block0 = LayerBlockGaussianNoise()

        # Layer 1
        n_neurons_1 = 200
        block1 = LayerBlockFullyConnected(neuron_relu, self.n_in, n_neurons_1)

        # Layer 2
        n_neurons_2 = 200
        block2 = LayerBlockFullyConnected(neuron_relu, n_neurons_1, n_neurons_2)

        # Layer 3
        block3 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_neurons_2, self.n_out)

        self.ls_layers = convert_blocks_into_feed_forward_layers([block0, block1, block2, block3])

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        pass

    def load_parameters_virtual(self, h5file):
        self.init(self.n_in, self.n_out)


class NetworkThreePatchesMLP(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.n_patches = 3

    def init(self, patch_width, n_out):
        Network.init_common(self, self.n_patches * patch_width**2, n_out)

        self.patch_width = patch_width

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, self.n_in + 1, interval)))

        # Layer 1
        n_out_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1)
        self.ls_layers.append(LayerOfBlocks([block0]*3))

        # Layer 2
        n_out_2 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_out_1, n_out_2)
        block1 = LayerBlockFullyConnected(neuron_relu, n_out_1, n_out_2)
        block2 = LayerBlockFullyConnected(neuron_relu, n_out_1, n_out_2)
        self.ls_layers.append(LayerOfBlocks([block0, block1, block2]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        n_in2 = n_out_2 * self.n_patches
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.update_params()

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)


class NetworkThreePatchesMLPPriors(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.n_patches = 3

    def init(self, patch_width, priors, n_out):
        Network.init_common(self, self.n_patches * patch_width**2, n_out)

        self.patch_width = patch_width

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, self.n_in + 1, interval)))

        # Layer 1
        n_out_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1)
        self.ls_layers.append(LayerOfBlocks([block0]*3))

        # Layer 2
        n_out_2 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_out_1, n_out_2)
        block1 = LayerBlockFullyConnected(neuron_relu, n_out_1, n_out_2)
        block2 = LayerBlockFullyConnected(neuron_relu, n_out_1, n_out_2)
        self.ls_layers.append(LayerOfBlocks([block0, block1, block2]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        n_in2 = n_out_2 * self.n_patches
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        self.ls_layers.append(LayerOfBlocks([LayerBlockMultiplication(priors)]))

        # Layer 7
        self.ls_layers.append(LayerOfBlocks([LayerBlockNormalization()]))

        self.update_params()

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)


class NetworkThreePatchesConv(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.n_patches = 3

    def init(self, patch_width, n_out):
        Network.init_common(self, self.n_patches * patch_width**2, n_out)

        self.patch_width = patch_width

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, self.n_in + 1, interval)))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        self.ls_layers.append(LayerOfBlocks([block0]*3))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = self.n_patches * n_kern1 * filter_map_height2 * filter_map_with2
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.update_params()

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)


class NetworkThreePatchesConvCentroids(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.n_patches = 3

    def init(self, patch_width, n_centroids, n_out):
        Network.init_common(self, self.n_patches * patch_width**2 + n_centroids, n_out)

        self.patch_width = patch_width

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, self.n_in + 1, interval) + [self.n_in]))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block1 = LayerBlockIdentity()
        self.ls_layers.append(LayerOfBlocks([block0]*3 + [block1]))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        block3 = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = self.n_patches * n_kern1 * filter_map_height2 * filter_map_with2 + n_centroids
        n_out2 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.update_params()

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, 134, self.n_out)


class Network6PatchesMLP(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.n_centroids = None

    def init(self, patch_width, n_out):
        Network.init_common(self, 6 * patch_width**2, n_out)

        self.patch_width = patch_width

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, 6*interval+1, interval)))

        # Layer 1
        n_out_1_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_2 = 1000
        block1 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block1, block1, block1]))

        # Layer 2
        n_out_2_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block1 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block2 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)

        n_out_2_2 = 1000
        block3 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block4 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block5 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        n_in2 = n_out_2_1 * 3 + n_out_2_2 * 3
        n_out2 = 4000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)


class Network6PatchesConv(Network):
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.in_height = None

    def init(self, patch_width, n_out):
        Network.init_common(self, 6 * patch_width*patch_width, n_out)

        self.patch_width = patch_width

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, self.n_in + interval, interval)))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block1, block1, block1]))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        block3 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block4 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block5 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = 6 * n_kern1 * filter_map_height2 * filter_map_with2
        n_out2 = 4000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.init(self.patch_width, self.n_out)

class NetworkWithoutCentroidConv(Network):
    '''
        A network without centriod but has dropout
    '''
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.in_height = None

    def init(self, patch_width, patch_width_3d, n_out):
        Network.init_common(self, 6 * patch_width*patch_width + patch_width_3d**3, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        splits = [0] + [patch_width**2]*3 + [patch_width**2]*3 + [self.patch_width_3d**3]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))
        # interval = patch_width ** 2
        # self.ls_layers.append(LayerDivideFeatures(range(0, self.n_in + interval, interval)))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block1, block1, block1, block_3d]))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        block3 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block4 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block5 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        n_out_2_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, n_out_1_3, n_out_2_3)

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())


        # Layer 4
        dropout_rate = 0.4
        self.ls_layers.append(LayerDropout(0.4))

        # Layer 5
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = 3 * n_kern1 * filter_map_height2 * filter_map_with2 + \
                3 * n_kern1 * filter_map_height2 * filter_map_with2 + n_out_2_3
        n_out2 = 3000

        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        # self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.patch_width_3d = 13
        self.init(self.patch_width, self.patch_width_3d, self.n_out)



# class Network3DPatchConv(Network):
#     """
#     3D convnet
#     """
#     def __init__(self):
#         Network.__init__(self)
#         self.patch_width = None
#         self.patch_width = None
#         self.patch_width = None

#     def init(self, patch_width, n_out):
#         Network.init_common(self, patch_width**3, n_out)

#         self.patch_width = patch_width

#         neuron_relu = neuron_type.NeuronRELU()

#         # Layer 0
#         filter_map_0_shape = np.array([patch_width, patch_width, patch_width], dtype=int)
#         filter_0_shape = np.array([2, 2, 2], dtype=int)
#         pool_0_shape = np.array([2, 2, 2], dtype=int)
#         n_kern0 = 20
#         block0 = LayerBlockConvPool3D(neuron_relu,
#                                       1, tuple(filter_map_0_shape),
#                                       n_kern0, tuple(filter_0_shape),
#                                       poolsize=tuple(pool_0_shape))

#         # Layer 1
#         filter_map_1_shape = (filter_map_0_shape - filter_0_shape + 1) / pool_0_shape
#         filter_1_shape = np.array([2, 2, 2], dtype=int)
#         pool_1_shape = np.array([2, 2, 2], dtype=int)
#         n_kern1 = 50
#         block1 = LayerBlockConvPool3D(neuron_relu,
#                                       n_kern0, tuple(filter_map_1_shape),
#                                       n_kern1, tuple(filter_1_shape),
#                                       poolsize=tuple(pool_1_shape))

#         # Layer 2
#         filter_map_2_shape = (filter_map_1_shape - filter_1_shape + 1) / pool_1_shape
#         n_in2 = n_kern1 * np.prod(filter_map_2_shape)
#         n_out2 = 1000
#         block2 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

#         # Layer 3
#         block3 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

#         self.ls_layers = convert_blocks_into_feed_forward_layers([block0, block1, block2, block3])

#         self.ls_params = []
#         for l in self.ls_layers:
#             self.ls_params += l.params

#     def save_parameters_virtual(self, h5file):
#         h5file.attrs['patch_width'] = self.patch_width

#     def load_parameters_virtual(self, h5file):
#         self.patch_width = int(h5file.attrs["patch_width"])
#         self.init(self.patch_width, self.n_out)


class NetworkUltimateMLP(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 6 * patch_width**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, 6*interval+1, interval)
                                                  + [6*interval + self.patch_width_3d**3]
                                                  + [self.n_in]))

        # Layer 1
        n_out_1_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_2 = 1000
        block1 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block1, block1, block1, block_3d, block_centroids]))

        # Layer 2
        n_out_2_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block1 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block2 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)

        n_out_2_2 = 1000
        block3 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block4 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block5 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)

        n_out_2_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, n_out_1_3, n_out_2_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d, block_centroids]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        n_in2 = n_out_2_1 * 3 + n_out_2_2 * 3 + n_out_2_3 + n_centroids
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.init(self.patch_width, self.patch_width_3d, self.n_centroids, self.n_out)


class NetworkUltimateMLPWithoutCentroids(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_3d, n_out):
        Network.init_common(self, 6 * patch_width**2 + patch_width_3d**3, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, 6*interval+1, interval)
                                                  + [6*interval + self.patch_width_3d**3]
                                                  + [self.n_in]))

        # Layer 1
        n_out_1_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_2 = 1000
        block1 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block1, block1, block1, block_3d]))

        # Layer 2
        n_out_2_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block1 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block2 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)

        n_out_2_2 = 1000
        block3 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block4 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block5 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)

        n_out_2_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, n_out_1_3, n_out_2_3)

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        n_in2 = n_out_2_1 * 3 + n_out_2_2 * 3 + n_out_2_3
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.init(self.patch_width, self.patch_width_3d, self.n_out)


class NetworkUltimateMLPWithoutCentroidsWo3D(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_3d, n_out):
        Network.init_common(self, 6 * patch_width**2 + patch_width_3d**3, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        interval = patch_width ** 2
        self.ls_layers.append(LayerDivideFeatures(range(0, 6*interval+1, interval)
                                                  + [6*interval + self.patch_width_3d**3]
                                                  + [self.n_in]))

        # Layer 1
        n_out_1_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_2 = 1000
        block1 = LayerBlockFullyConnected(neuron_relu, patch_width**2, n_out_1_1)

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block1, block1, block1, block_3d]))

        # Layer 2
        n_out_2_1 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block1 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)
        block2 = LayerBlockFullyConnected(neuron_relu, n_out_1_1, n_out_2_1)

        n_out_2_2 = 1000
        block3 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block4 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)
        block5 = LayerBlockFullyConnected(neuron_relu, n_out_1_2, n_out_2_2)

        n_out_2_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, n_out_1_3, n_out_2_3)

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        n_in2 = n_out_2_1 * 3 + n_out_2_2 * 3 + n_out_2_3
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.init(self.patch_width, self.patch_width_3d, self.n_out)


class NetworkDeeperConv(Network):
    """
    3D convnet with optional n_conv_layer
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out, n_conv_layer=2):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.n_conv_layer = n_conv_layer
        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Layers N

        kernel_height = 5
        kernel_width = 5
        pool_size_height = 2
        pool_size_width = 2
        filter_map_height = patch_width
        filter_map_width = patch_width
        n_kern = 1
        n_out_3 = patch_width_3d**3
        for ly in range(n_conv_layer):
            if not ly == 0:
                filter_map_height = (filter_map_height - kernel_height + 1) / pool_size_height
                filter_map_width = (filter_map_width - kernel_width + 1) / pool_size_width

            if ly == 1:
                pool_size_height = 1
                pool_size_width = 1
            else:   
                pool_size_height = 2
                pool_size_width = 2

            prev_n_kern = n_kern
            n_kern = (ly+1)*20
            block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(prev_n_kern, filter_map_height, filter_map_width),
                                      flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width),
                                      poolsize=(pool_size_height, pool_size_width))

            n_in_3 = n_out_3
            n_out_3 = 1000
            block_3d = LayerBlockFullyConnected(neuron_relu, n_in_3, n_out_3)

            block_centroids = LayerBlockIdentity()
            self.ls_layers.append(LayerOfBlocks([block0, block0, block0, block0, block0, block0, block_3d, block_centroids]))

        # Layer N+1
        self.ls_layers.append(LayerMergeFeatures())

        # Layer N+2
        filter_map_height = (filter_map_height - kernel_height + 1) / pool_size_height
        filter_map_with = (filter_map_width - kernel_width + 1) / pool_size_width
        # filter_map_height_comp = (filter_map_height_comp - kernel_height + 1) / pool_size_height
        # filter_map_with_comp = (filter_map_width_comp - kernel_width + 1) / pool_size_width
        n_in2 = 3 * n_kern * filter_map_height * filter_map_with + \
                3 * n_kern * filter_map_height * filter_map_with + n_out_3 + n_centroids
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer N+3
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer N+4
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.init(self.patch_width, 29, self.patch_width_3d, self.n_centroids, self.n_out)

class NetworkConvDropout(Network):
    """
    3D convnet with Dropout
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block3 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block4 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block5 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d, block_centroids]))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        filter_map_height1_comp = (patch_width_comp - kernel_height0 + 1) / pool_size_height0
        filter_map_width1_comp = (patch_width_comp - kernel_width0 + 1) / pool_size_width0
        block3 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1_comp, filter_map_width1_comp),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block4 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1_comp, filter_map_width1_comp),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block5 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1_comp, filter_map_width1_comp),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        n_out_2_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, n_out_1_3, n_out_2_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d, block_centroids]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        dropout_rate = 0.4
        self.ls_layers.append(LayerDropout(dropout_rate))

        # Layer 5
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        filter_map_height2_comp = (filter_map_height1_comp - kernel_height1 + 1) / pool_size_height1
        filter_map_with2_comp = (filter_map_width1_comp - kernel_width1 + 1) / pool_size_width1
        n_in2 = 3 * n_kern1 * filter_map_height2 * filter_map_with2 + \
                3 * n_kern1 * filter_map_height2_comp * filter_map_with2_comp + n_out_2_3 + n_centroids
        n_out2 = 3000

        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 7
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids
        h5file.attrs['n_out'] = self.n_out

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.n_out = int(h5file.attrs["n_out"])
        self.init(self.patch_width, self.patch_width, self.patch_width_3d, self.n_centroids, self.n_out)


class ResNet(Network):
    """
    ResNet
    """

    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    # helper function to handle filters/strides when increasing dims
    def filters_increase_dims(self, in_block, increase_dims):
        if increase_dims:
            first_stride = (2, 2)
            in_filters = in_block.filter_shape[0]*2
        else:
            first_stride = (1, 1)
            in_filters = in_block.filter_shape[0]

        return first_stride, in_filters


    def projectionB(self, neuron_type, in_block):
    # twice normal channels when projecting!
        n_filters = in_block.filter_shape[0]*2
        block = LayerBlockPadConv2D(neuron_type, in_block.output_shape, (n_filters, 1,1), (2,2), (0,0))
        return block

    def res_block_v1(self, block_centroids, in_blocks, neuron_type, increase_dims=False):

        first_stride, num_filters = self.filters_increase_dims(in_blocks[0], increase_dims)
        pads = (in_blocks[0].filter_shape[2]//2, in_blocks[0].filter_shape[3]//2)
        ls_layers = []
        # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
        convBlocks0 = []
        for _ in range(6):
            convBlock0 = LayerBlockPadConv2D(neuron_type, in_blocks[0].output_shape, (num_filters,in_blocks[0].filter_shape[0],3,3), first_stride, pads)
            convBlocks0.append(convBlock0)
        if increase_dims:
            blocks = convBlocks0 + [LayerBlockFullyConnected(neuron_type, 1000, 1000), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))
        else:
            blocks = convBlocks0 + [LayerBlockIdentity(), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))

        # batch normalization
        BNs = []
        for _ in range(6):
            BNs.append(LayerBlockNormalization())
        blocks = BNs + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        # nonlinear activation
        nonlins = []
        for _ in range(6):
            nonlins.append(LayerBlockNonLinear(neuron_type))
        blocks = nonlins + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        convBlocks1 = []
        for i in range(6):
            convBlock1 = LayerBlockPadConv2D(neuron_type, convBlock0.output_shape, (num_filters,convBlock0.filter_shape[0],3,3), (1,1), (1,1))
            convBlocks1.append(convBlock1)
        blocks = convBlocks1 + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        # batch normalization
        BNs = []
        for _ in range(6):
            BNs.append(LayerBlockNormalization())
        blocks = BNs + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        sumBlocks = []
        if increase_dims:
            for i in range(6):
                sumBlocks.append(LayerBlockSumProjA(in_blocks[i]))
            blocks = sumBlocks + [LayerBlockIdentity(), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))
        else:
            for i in range(6):
                sumBlocks.append(LayerBlockSum(in_blocks[i]))
            blocks = sumBlocks + [LayerBlockIdentity(), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))

        # nonlinear activation
        nonlins = []
        for i in range(6):
            nonlins.append(LayerBlockNonLinear(neuron_type, True, in_blocks[i], convBlock1.filter_shape, convBlock1.output_shape))
        blocks = nonlins + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        return ls_layers

    def bottleneck_block_fast(self, block_centroids, in_blocks, neuron_type, increase_dims=False):
        last_stride, num_filters = self.filters_increase_dims(in_blocks[0], increase_dims)

        ls_layers = []
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin

        scale = 4
        scaled_filters = num_filters/4

        convBlocks0 = []
        for _ in range(6):
            convBlock0 = LayerBlockPadConv2D(neuron_type, in_blocks[0].output_shape, (scaled_filters,in_blocks[0].filter_shape[0],1,1), (1,1), (0,0))
            convBlocks0.append(convBlock0)
        if increase_dims:
            blocks = convBlocks0 + [LayerBlockFullyConnected(neuron_type, 1000, 1000), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))
        else:
            blocks = convBlocks0 + [LayerBlockIdentity(), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))

        # batch normalization
        BNs = []
        for _ in range(6):
            BNs.append(LayerBlockNormalization())
        blocks = BNs + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        # nonlinear activation
        nonlins = []
        for _ in range(6):
            nonlins.append(LayerBlockNonLinear(neuron_type))
        blocks = nonlins + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        convBlocks1 = []
        for i in range(6):
            convBlock1 = LayerBlockPadConv2D(neuron_type, convBlock0.output_shape, (scaled_filters,convBlock0.filter_shape[0],3,3), (1,1), (1,1))
            convBlocks1.append(convBlock1)
        blocks = convBlocks1 + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        # batch normalization
        BNs = []
        for _ in range(6):
            BNs.append(LayerBlockNormalization())
        blocks = BNs + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        # nonlinear activation
        nonlins = []
        for _ in range(6):
            nonlins.append(LayerBlockNonLinear(neuron_type))
        blocks = nonlins + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))


        pads = (convBlock1.filter_shape[2]//2, convBlock1.filter_shape[3]//2)
        convBlocks2 = []
        for i in range(6):
            convBlock2 = LayerBlockPadConv2D(neuron_type, convBlock1.output_shape, (num_filters,convBlock1.filter_shape[0],1,1), last_stride, pads)
            convBlocks2.append(convBlock2)
        blocks = convBlocks2 + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        sumBlocks = []
        if increase_dims:
            for i in range(6):
                sumBlocks.append(LayerBlockSumProjA(in_blocks[i]))
            blocks = sumBlocks + [LayerBlockIdentity(), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))
        else:
            for i in range(6):
                sumBlocks.append(LayerBlockSum(in_blocks[i]))
            blocks = sumBlocks + [LayerBlockIdentity(), block_centroids]
            ls_layers.append(LayerOfBlocks(blocks))

        # nonlinear activation
        nonlins = []
        for i in range(6):
            nonlins.append(LayerBlockNonLinear(neuron_type, True, in_blocks[i], convBlock2.filter_shape, convBlock2.output_shape))
        blocks = nonlins + [LayerBlockIdentity(), block_centroids]
        ls_layers.append(LayerOfBlocks(blocks))

        return ls_layers



    def blockstack(self, ls_layers, n, neuron_type, block_centroids, in_blocks):
        res_block = self.bottleneck_block_fast
        for _ in range(n):
            res_layers = res_block(block_centroids, in_blocks, neuron_type)
            ls_layers.extend(res_layers)
        return ls_layers


    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Layer 1
        kernel_height0 = 3
        kernel_width0 = 3
        # n_kern0 = 32
        n_kern0 = 8
        block0 = LayerBlockConv2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      save_output=True)

        block1 = LayerBlockConv2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      save_output=True)

        block2 = LayerBlockConv2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      save_output=True)

        block3 = LayerBlockConv2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      save_output=True)

        block4 = LayerBlockConv2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      save_output=True)

        block5 = LayerBlockConv2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      save_output=True)

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        block_centroids = LayerBlockIdentity()

        layer_block = [block0, block1, block2, block3, block4, block5, block_3d, block_centroids]

        self.ls_layers.append(LayerOfBlocks(layer_block))

        self.ls_layers = self.blockstack(self.ls_layers, 1, neuron_relu, block_centroids, layer_block[:6])
        res_block = self.bottleneck_block_fast(block_centroids, self.ls_layers[len(self.ls_layers)-1].ls_layer_blocks[:6], neuron_relu, True)
        self.ls_layers.extend(res_block)

        self.ls_layers = self.blockstack(self.ls_layers, 1, neuron_relu, block_centroids, self.ls_layers[len(self.ls_layers)-1].ls_layer_blocks[:6])
        res_block = self.bottleneck_block_fast(block_centroids, self.ls_layers[len(self.ls_layers)-1].ls_layer_blocks[:6], neuron_relu, True)
        self.ls_layers.extend(res_block)

        self.ls_layers = self.blockstack(self.ls_layers, 1, neuron_relu, block_centroids, self.ls_layers[len(self.ls_layers)-1].ls_layer_blocks[:6])
        res_block = self.bottleneck_block_fast(block_centroids, self.ls_layers[len(self.ls_layers)-1].ls_layer_blocks[:6], neuron_relu, True)
        self.ls_layers.extend(res_block)

        self.ls_layers = self.blockstack(self.ls_layers, 1, neuron_relu, block_centroids, self.ls_layers[len(self.ls_layers)-1].ls_layer_blocks[:6])

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        last_layer_blocks = self.ls_layers[len(self.ls_layers)-2].ls_layer_blocks
        filter_map_height = last_layer_blocks[0].output_shape[1]
        filter_map_with = last_layer_blocks[0].output_shape[2]
        filter_map_height_comp = last_layer_blocks[3].output_shape[1]
        filter_map_with_comp = last_layer_blocks[3].output_shape[2]
        n_in2 = 3 * last_layer_blocks[0].filter_shape[0] * filter_map_height * filter_map_with + \
                3 * last_layer_blocks[3].filter_shape[0] * filter_map_height_comp * filter_map_with_comp + n_out_1_3 + n_centroids
        n_out2 = 1000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids
        h5file.attrs['n_out'] = self.n_out

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.n_out = int(h5file.attrs["n_out"])
        self.init(self.patch_width, self.patch_width, self.patch_width_3d, self.n_centroids, self.n_out)




class NetworkUltimateConv(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Layer 1
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block3 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block4 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        block5 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_width_comp, patch_width_comp),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        n_out_1_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_1_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d, block_centroids]))

        # Layer 2
        filter_map_height1 = (patch_width - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block2 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        filter_map_height1_comp = (patch_width_comp - kernel_height0 + 1) / pool_size_height0
        filter_map_width1_comp = (patch_width_comp - kernel_width0 + 1) / pool_size_width0
        block3 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1_comp, filter_map_width1_comp),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block4 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1_comp, filter_map_width1_comp),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))
        block5 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1_comp, filter_map_width1_comp),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        n_out_2_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, n_out_1_3, n_out_2_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block0, block1, block2, block3, block4, block5, block_3d, block_centroids]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        filter_map_height2_comp = (filter_map_height1_comp - kernel_height1 + 1) / pool_size_height1
        filter_map_with2_comp = (filter_map_width1_comp - kernel_width1 + 1) / pool_size_width1
        n_in2 = 3 * n_kern1 * filter_map_height2 * filter_map_with2 + \
                3 * n_kern1 * filter_map_height2_comp * filter_map_with2_comp + n_out_2_3 + n_centroids
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.init(self.patch_width, 29, self.patch_width_3d, self.n_centroids, self.n_out)


class NetworkConvVGG(Network):
    """
    3D convnet VGG-alike
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def conv_group_factory(self, input_shape, kernel_shape, prev_n_kern, n_kern, n_conv, pool_size_shape, prev_n_out_3, n_out_3, neuron):
        ls_layers = []

        input_height = input_shape[0]
        input_width = input_shape[1]
        kernel_height = kernel_shape[0]
        kernel_width = kernel_shape[1]

        for i in range(n_conv-1):
            if i > 0:
                input_height = input_height - kernel_height + 1
                input_width = input_width - kernel_width + 1
                prev_n_kern = n_kern
                prev_n_out_3 = n_out_3

            block_2Ds = []
            for j in range(6):
                block_2D = LayerBlockConv2D(neuron,
                                          in_shape=(prev_n_kern, input_height, input_width),
                                          flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width))
                block_2Ds.append(block_2D)

            block_3d = LayerBlockFullyConnected(neuron, prev_n_out_3, n_out_3)

            block_centroids = LayerBlockIdentity()

            blocks = block_2Ds + [block_3d, block_centroids]

            ls_layers.append(LayerOfBlocks(blocks))

        input_height = input_height - kernel_height + 1
        input_width = input_width - kernel_width + 1

        block_2Ds = []
        for j in range(6):
            block_2D = LayerBlockConvPool2D(neuron,
                                      in_shape=(prev_n_kern, input_height, input_width),
                                      flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width),
                                      poolsize=pool_size_shape)
            block_2Ds.append(block_2D)

        block_3d = LayerBlockFullyConnected(neuron, prev_n_out_3, n_out_3)

        block_centroids = LayerBlockIdentity()

        blocks = block_2Ds + [block_3d, block_centroids]

        ls_layers.append(LayerOfBlocks(blocks))

        block_2D_output_shape = ((input_height - kernel_height + 1)/pool_size_shape[0],
                                    (input_width - kernel_width + 1)/pool_size_shape[1])
        return (ls_layers, block_2D_output_shape)

    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Conv group 0
        input_shape = (patch_width, patch_width)
        prev_n_kern = 1
        n_kern = 30
        n_conv = 3
        kernel_shape = (3,3)
        pool_size_shape = (2,2)
        prev_n_out_3=patch_width_3d**3
        n_out_3 = 1000

        conv_group = self.conv_group_factory(input_shape=input_shape,
                                        kernel_shape=kernel_shape,
                                        prev_n_kern=prev_n_kern,
                                        n_kern=n_kern,
                                        n_conv=n_conv,
                                        pool_size_shape=pool_size_shape,
                                        prev_n_out_3=prev_n_out_3,
                                        n_out_3=n_out_3,
                                        neuron=neuron_relu)

        self.ls_layers.extend(conv_group[0])

        # Conv group 1
        input_shape = conv_group[1]
        prev_n_kern = n_kern
        n_kern = 60
        n_conv = 3
        kernel_shape = (3,3)
        pool_size_shape = (2,2)
        prev_n_out_3 = n_out_3
        n_out_3 = 1000

        conv_group = self.conv_group_factory(input_shape=input_shape,
                                        kernel_shape=kernel_shape,
                                        prev_n_kern=prev_n_kern,
                                        n_kern=n_kern,
                                        n_conv=n_conv,
                                        pool_size_shape=pool_size_shape,
                                        prev_n_out_3=prev_n_out_3,
                                        n_out_3=n_out_3,
                                        neuron=neuron_relu)

        self.ls_layers.extend(conv_group[0])

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height = conv_group[1][0]
        filter_map_width = conv_group[1][1]
        n_in2 = 3 * n_kern * filter_map_height * filter_map_width + \
                3 * n_kern * filter_map_height * filter_map_width + n_out_3 + n_centroids
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.init(self.patch_width, 29, self.patch_width_3d, self.n_centroids, self.n_out)

class NetworkConvVGGDropout(Network):
    """
    3D convnet VGG-alike
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def conv_group_factory(self, input_shape, kernel_shape, prev_n_kern, n_kern, n_conv, pool_size_shape, prev_n_out_3, n_out_3, neuron):
        ls_layers = []

        input_height = input_shape[0]
        input_width = input_shape[1]
        kernel_height = kernel_shape[0]
        kernel_width = kernel_shape[1]

        for i in range(n_conv-1):
            if i > 0:
                input_height = input_height - kernel_height + 1
                input_width = input_width - kernel_width + 1
                prev_n_kern = n_kern
                prev_n_out_3 = n_out_3

            block_2Ds = []
            for j in range(6):
                block_2D = LayerBlockConv2D(neuron,
                                          in_shape=(prev_n_kern, input_height, input_width),
                                          flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width))
                block_2Ds.append(block_2D)

            block_3d = LayerBlockFullyConnected(neuron, prev_n_out_3, n_out_3)

            block_centroids = LayerBlockIdentity()

            blocks = block_2Ds + [block_3d, block_centroids]

            ls_layers.append(LayerOfBlocks(blocks))

        input_height = input_height - kernel_height + 1
        input_width = input_width - kernel_width + 1

        block_2Ds = []
        for j in range(6):
            block_2D = LayerBlockConvPool2D(neuron,
                                      in_shape=(prev_n_kern, input_height, input_width),
                                      flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width),
                                      poolsize=pool_size_shape)
            block_2Ds.append(block_2D)

        block_3d = LayerBlockFullyConnected(neuron, prev_n_out_3, n_out_3)

        block_centroids = LayerBlockIdentity()

        blocks = block_2Ds + [block_3d, block_centroids]

        ls_layers.append(LayerOfBlocks(blocks))

        block_2D_output_shape = ((input_height - kernel_height + 1)/pool_size_shape[0],
                                    (input_width - kernel_width + 1)/pool_size_shape[1])
        return (ls_layers, block_2D_output_shape)

    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Conv group 0
        input_shape = (patch_width, patch_width)
        prev_n_kern = 1
        n_kern = 30
        n_conv = 3
        kernel_shape = (3,3)
        pool_size_shape = (2,2)
        prev_n_out_3=patch_width_3d**3
        n_out_3 = 1000

        conv_group = self.conv_group_factory(input_shape=input_shape,
                                        kernel_shape=kernel_shape,
                                        prev_n_kern=prev_n_kern,
                                        n_kern=n_kern,
                                        n_conv=n_conv,
                                        pool_size_shape=pool_size_shape,
                                        prev_n_out_3=prev_n_out_3,
                                        n_out_3=n_out_3,
                                        neuron=neuron_relu)

        self.ls_layers.extend(conv_group[0])

        # Conv group 1
        input_shape = conv_group[1]
        prev_n_kern = n_kern
        n_kern = 60
        n_conv = 3
        kernel_shape = (3,3)
        pool_size_shape = (2,2)
        prev_n_out_3 = n_out_3
        n_out_3 = 1000

        conv_group = self.conv_group_factory(input_shape=input_shape,
                                        kernel_shape=kernel_shape,
                                        prev_n_kern=prev_n_kern,
                                        n_kern=n_kern,
                                        n_conv=n_conv,
                                        pool_size_shape=pool_size_shape,
                                        prev_n_out_3=prev_n_out_3,
                                        n_out_3=n_out_3,
                                        neuron=neuron_relu)

        self.ls_layers.extend(conv_group[0])

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Dropout Layer
        dropout_rate = 0.4
        self.ls_layers.append(LayerDropout(dropout_rate))

        # Layer 4
        filter_map_height = conv_group[1][0]
        filter_map_width = conv_group[1][1]
        n_in2 = 3 * n_kern * filter_map_height * filter_map_width + \
                3 * n_kern * filter_map_height * filter_map_width + n_out_3 + n_centroids
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.init(self.patch_width, 29, self.patch_width_3d, self.n_centroids, self.n_out)

class NetworkConvTest(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.patch_width = None
        self.patch_width_3d = None
        self.n_centroids = None

    def init(self, patch_width, patch_width_comp, patch_width_3d, n_centroids, n_out):
        Network.init_common(self, 3 * patch_width**2 + 3 * patch_width_comp**2 + patch_width_3d**3 + n_centroids, n_out)

        self.patch_width = patch_width
        self.patch_width_3d = patch_width_3d
        self.n_centroids = n_centroids

        self.ls_layers = []
        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0

        splits = [0] + [patch_width**2]*3 + [patch_width_comp**2]*3 + [self.patch_width_3d**3] + [n_centroids]
        self.ls_layers.append(LayerDivideFeatures(np.cumsum(splits)))

        # Layer 1
        filter_map_height = patch_width
        filter_map_width = patch_width
        kernel_height = 5
        kernel_width = 5
        pool_size_height = 2
        pool_size_width = 2
        prev_n_kern = 1
        n_kern = 20
        block_2D = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(prev_n_kern, filter_map_height, filter_map_width),
                                      flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width),
                                      poolsize=(pool_size_height, pool_size_width))

        n_out_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, patch_width_3d**3, n_out_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block_2D, block_2D, block_2D, block_2D, block_2D, block_2D, block_3d, block_centroids]))

        # Layer 2
        filter_map_height = (filter_map_height - kernel_height + 1) / pool_size_height
        filter_map_width = (filter_map_width - kernel_width + 1) / pool_size_width
        kernel_height = 5
        kernel_width = 5
        pool_size_height = 2
        pool_size_width = 2
        prev_n_kern = n_kern
        n_kern = 50
        block_2D = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(prev_n_kern, filter_map_height, filter_map_width),
                                      flt_shape=(n_kern, prev_n_kern, kernel_height, kernel_width),
                                      poolsize=(pool_size_height, pool_size_width))

        prev_n_out_3 = n_out_3
        n_out_3 = 1000
        block_3d = LayerBlockFullyConnected(neuron_relu, prev_n_out_3, n_out_3)

        block_centroids = LayerBlockIdentity()

        self.ls_layers.append(LayerOfBlocks([block_2D, block_2D, block_2D, block_2D, block_2D, block_2D, block_3d, block_centroids]))

        # Layer 3
        self.ls_layers.append(LayerMergeFeatures())

        # Layer 4
        filter_map_height = (filter_map_height - kernel_height + 1) / pool_size_height
        filter_map_width = (filter_map_width - kernel_width + 1) / pool_size_width
        n_in2 = 3 * n_kern * filter_map_height * filter_map_width + \
                3 * n_kern * filter_map_height * filter_map_width + n_out_3 + n_centroids
        n_out2 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 5
        n_out3 = 3000
        block0 = LayerBlockFullyConnected(neuron_relu, n_in=n_out2, n_out=n_out3)
        self.ls_layers.append(LayerOfBlocks([block0]))

        # Layer 6
        block0 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out3, n_out=self.n_out)
        self.ls_layers.append(LayerOfBlocks([block0]))

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['patch_width'] = self.patch_width
        h5file.attrs['patch_width_3d'] = self.patch_width_3d
        h5file.attrs['n_centroids'] = self.n_centroids

    def load_parameters_virtual(self, h5file):
        self.patch_width = int(h5file.attrs["patch_width"])
        self.patch_width_3d = int(h5file.attrs["patch_width_3d"])
        self.n_centroids = int(h5file.attrs["n_centroids"])
        self.init(self.patch_width, 29, self.patch_width_3d, self.n_centroids, self.n_out)



# class NetworkTwin(Network):
#     """
#     3D convnet
#     """
#     def __init__(self, net_no_centoids, net_centroids):
#         Network.__init__(self)
#         self.net_no_centoids = net_no_centoids
#         self.net_centroids = net_centroids
#
#     def init(self):
#         pass
#
#     def predict(self, in_numpy_array, batch_size_limit):
#         """
#         User-friendly function to return the outputs of provided inputs without worrying about batch_size.
#         Args:
#             in_numpy_array (2D array): dataset in which rows are datapoints
#             batch_size_limit (int): limit size of a batch (should be what the GPU memory can support (or the RAM))
#         Returns:
#             pred (2D array): outputs of the network for the given inputs
#         """
#         n_inputs = in_numpy_array.shape[0]
#         out_pred = np.zeros((n_inputs, self.n_out), dtype=np.float32)  # Will store the output predictions
#         batch_size = min(batch_size_limit, n_inputs)
#         pred_fun = self.generate_testing_function(batch_size)
#
#         n_batches, n_rest = divmod(n_inputs, batch_size)
#
#         print "--------------------"
#         for b in xrange(n_batches):
#             sys.stdout.write("\r        Prediction: {}%".format(100*b/n_batches))
#             sys.stdout.flush()
#             id0 = b*batch_size
#             id1 = id0 + batch_size
#             out_pred[id0:id1] = pred_fun(in_numpy_array[id0:id1])
#
#         if n_rest > 0:
#             pred_fun_res = self.generate_testing_function(n_rest)
#             out_pred[n_batches*batch_size:] = pred_fun_res(in_numpy_array[n_batches*batch_size:])
#
#         return out_pred
#
#     def predict_from_generator(self, brain_batches, scaler, pred_functions=None):
#         if pred_functions is None:
#             pred_functions = {}
#         ls_vx = []
#         ls_pred = []
#         id_batch = 0
#         for vx_batch, patch_batch, tg_batch in brain_batches:
#             id_batch += 1
#
#             batch_size_current = len(vx_batch)
#             if batch_size_current not in pred_functions:
#                 pred_functions[batch_size_current] = self.generate_testing_function(batch_size_current)
#
#             if scaler is not None:
#                 scaler.scale(patch_batch)
#
#             pred_raw = pred_functions[batch_size_current](patch_batch)
#
#             pred = np.argmax(pred_raw, axis=1)
#             err = error_rate(pred, np.argmax(tg_batch, axis=1))
#             print "     {}".format(err)
#
#             ls_vx.append(vx_batch)
#             ls_pred.append(pred)
#
#         return ls_vx, ls_pred