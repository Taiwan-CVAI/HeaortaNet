import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.image import sobel_edges
from ai4med.operations import initializer, normalization, dropout, upsample
from ai4med.common.data_format import DataFormat
from ai4med.common.constants import ActivationFunc
import keras
import keras.backend as K

class HeaortaNet(object):
    __module__ = __name__
    __qualname__ = 'HeaortaNet'

    def __init__(self, inputs, num_classes, training, blocks_down='1,2,2,4', blocks_up='1,1,1', init_filters=8, use_batch_norm=False, use_group_norm=True, use_group_normG=8, reg_weight=0.0, dropout_prob=0.0, final_activation=ActivationFunc.SOFTMAX, use_vae=False, dtype=tf.float32, data_format=DataFormat.CHANNELS_FIRST):
        print('HeaortaNet init, num_classes:', num_classes, 'data_format:', data_format)
        print('HeaortaNet net_config:')
        print('num_classes      =>', num_classes)
        print('data_format      =>', data_format)
        print('blocks_down      =>', blocks_down)
        print('blocks_up        =>', blocks_up)
        print('init_filters     =>', init_filters)
        print('use_batchnorm    =>', use_batch_norm)
        print('use_groupnorm    =>', use_group_norm)
        print('use_groupnormG   =>', use_group_normG)
        print('reg_weight       =>', reg_weight)
        print('dropout_prob     =>', dropout_prob)
        print('final_activation =>', final_activation)
        print('use_vae          =>', use_vae)
        print('-----------------')
        self.input = inputs
        self.nb_classes = num_classes
        self.training = training
        self.dtype = dtype
        self.data_format = data_format
        self.blocks_down = list(map(int, blocks_down.split(',')))
        self.blocks_up = list(map(int, blocks_up.split(',')))
        self.init_filters = init_filters
        self.use_batchnorm = use_batch_norm
        self.use_groupnorm = use_group_norm
        self.use_groupnormG = use_group_normG
        self.reg_weight = reg_weight
        self.dropout_prob = dropout_prob
        self.use_bias = not self.use_batchnorm and not self.use_groupnorm
        self.final_activation = final_activation
        self.use_vae = use_vae

    def _is3D(self, x):
        return self._getLen(x) == 5

    def _getShape(self, x):
        return x.get_shape().as_list()

    def _getLen(self, x):
        return len(x.get_shape())

    def _isChannelsFirst(self, data_format):
        return data_format == 'channels_first'

    def _channelsAxis(self, data_format):
        if self._isChannelsFirst(data_format):
            return 1
        else:
            return -1

    def _getNumberOfChannels(self, x, data_format):
        return x.get_shape()[self._channelsAxis(data_format)].value

    def _kernel_initializer(self):
        return initializer.he_normal(None, self.dtype)

    def _regularizer(self):
        if self.reg_weight > 0:
            return regularizers.l2(self.reg_weight)

    def _conv(self, inputs, filters, kernel_size, strides, padding='same', dilation_rate=1, use_bias=None):
        with tf.name_scope('conv'):
            if use_bias is None:
                use_bias = self.use_bias
            data_format = self.data_format
            if self._is3D(inputs):
                inputs = tf.layers.conv3d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                      padding=padding,
                      data_format=data_format,
                      use_bias=use_bias,
                      kernel_initializer=(self._kernel_initializer()),
                      kernel_regularizer=(self._regularizer()))
            else:
                inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                      padding=padding,
                      data_format=data_format,
                      use_bias=use_bias,
                      kernel_initializer=(self._kernel_initializer()),
                      kernel_regularizer=(self._regularizer()))
        return inputs

    def _conv_transpose(self, inputs, filters, kernel_size, strides, padding='same', use_bias=None):
        with tf.name_scope('_conv_transpose'):
            if use_bias is None:
                use_bias = self.use_bias
            else:
                data_format = self.data_format
                if self._is3D(inputs):
                    inputs = tf.layers.conv3d_transpose(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      data_format=data_format,
                      use_bias=use_bias,
                      kernel_initializer=(self._kernel_initializer()),
                      kernel_regularizer=(self._regularizer()))
                else:
                    inputs = tf.layers.conv2d_transpose(inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      data_format=data_format,
                      use_bias=use_bias,
                      kernel_initializer=(self._kernel_initializer()),
                      kernel_regularizer=(self._regularizer()))
        return inputs

    def _batch_norm_relu(self, inputs, scope='batch_norm_scope'):
        with tf.name_scope('_batch_norm_relu'):
            data_format = self.data_format
            if self.use_batchnorm:
                inputs = tf.layers.batch_normalization(inputs=inputs, axis=(self._channelsAxis(data_format)), training=(self.training),
                  fused=True)
            else:
                if self.use_groupnorm:
                    inputs = normalization.group_norm(inputs, data_format=(self.data_format), G=(self.use_groupnormG), scope=scope)
            inputs = tf.nn.relu(inputs)
        return inputs

    def model(self):
        """Define neural network layers of ResNet based encoder and decoder.

        Layers connect with other modules to construct TensorFlow graph.

        Args:
            None

        Returns:
            Prediction results

        """
        if not self.use_vae:
            output = self._modelOnly()
        else:
            output = self._modelWithVae()
        return output

    def _modelOnly(self):
        inputs = self.input
        blocks_down = self.blocks_down
        blocks_up = self.blocks_up
        data_format = self.data_format
        is_training = self.training
        init_filters = self.init_filters
        use_bias = self.use_bias
        print('HeaortaNet inputs ', self._getShape(inputs), 'use_bias', use_bias, 'filters', init_filters, 'blocks_down ', blocks_down, 'blocks_up', blocks_up)
        chAxis = self._channelsAxis(data_format)
        skips = []
        filters = init_filters
        with tf.variable_scope('InitialConv'):
            inputs = self._conv(inputs, filters=filters, kernel_size=3, strides=1)
        if self.dropout_prob > 0:
            inputs = dropout.spatial_dropout(inputs, rate=(self.dropout_prob), training=is_training, data_format=(self.data_format))
        for fidx, num_blocks in enumerate(blocks_down):
            for i in range(num_blocks):
                s = inputs
                with tf.variable_scope('Down0_{}_{}'.format(fidx, i)):
                    inputs = self._batch_norm_relu(inputs)
                    inputs = self._conv(inputs, filters=filters, kernel_size=3, strides=1)
                with tf.variable_scope('Down1_{}_{}'.format(fidx, i)):
                    inputs = self._batch_norm_relu(inputs)
                    inputs = self._conv(inputs, filters=filters, kernel_size=3, strides=1)
                inputs += s
                print('LayerDown [', fidx, ':', i, '] shape', self._getShape(inputs))
            print(fidx, num_blocks, len(blocks_down))
            if fidx < len(blocks_down) - 1:
                skips.append(inputs)
                filters *= 2
                with tf.variable_scope('DownscaleConvOne_{}_{}'.format(fidx, i)):
                    print('down')
                    inputs = self._conv(inputs, filters=filters, kernel_size=3, strides=2)

        self.encoderEndpoint = inputs
        revSkips = list(reversed(skips))
        for fidx, num_blocks in enumerate(blocks_up):
            filters //= 2
            with tf.variable_scope('UpscaleConvOne_{}'.format(fidx)):
                inputs = self._conv(inputs, filters=filters, kernel_size=1, strides=1)
                inputs = upsample.upsample_semilinear(inputs, data_format=(self.data_format), upsample_factor=2)
            with tf.variable_scope('UpscaleAtten_{}'.format(fidx)):
                pass
            inputs += revSkips[fidx]
            for i in range(num_blocks):
                s = inputs
                f = self._getNumberOfChannels(inputs, self.data_format)
                with tf.variable_scope('Up0_{}_{}'.format(fidx, i)):
                    inputs = self._batch_norm_relu(inputs)
                    inputs = self._conv(inputs, filters=f, kernel_size=3, strides=1)
                with tf.variable_scope('Up1_{}_{}'.format(fidx, i)):
                    inputs = self._batch_norm_relu(inputs)
                    inputs = self._conv(inputs, filters=f, kernel_size=3, strides=1)
                inputs += s
                print('LayerUp [', fidx, ':', i, '] shape', self._getShape(inputs))

        with tf.variable_scope('Final'):
            inputs = self._batch_norm_relu(inputs)
            inputs = self._conv(inputs, filters=(self.nb_classes), kernel_size=1, strides=1, use_bias=True)
            print('Final activation', self.final_activation)
            if str.lower(self.final_activation) == 'softmax':
                inputs = tf.nn.softmax(inputs, axis=chAxis, name='softmax')
            else:
                if str.lower(self.final_activation) == 'sigmoid':
                    inputs = tf.nn.sigmoid(inputs, name='sigmoid')
                else:
                    if str.lower(self.final_activation) == 'linear':
                        pass
                    else:
                        raise ValueError('Unsupported final_activation, it must of one (softmax, sigmoid or linear), but provided:' + self.final_activation)
        return inputs

    def _modelWithVae(self):
        outputs = self._modelOnly()
        blocks_down = self.blocks_down
        blocks_up = self.blocks_up
        init_filters = self.init_filters
        inputs = self.encoderEndpoint
        filters = init_filters * 2 ** (len(blocks_down) - 1)
        print('_modelWithVae filters', filters)
        print('modelResnet20 blocks_down ', blocks_down, 'blocks_up', blocks_up)
        vaeLoss = 0
        n_z = 128
        with tf.variable_scope('batchnormVAE0'):
            inputs = self._batch_norm_relu(inputs)
        inputs = tf.nn.leaky_relu(self._conv(inputs, filters=16, kernel_size=3, strides=2, use_bias=True))
        sh = self._getShape(inputs)
        inputs = tf.reshape(inputs, [-1, sh[1] * sh[2] * sh[3] * sh[4]])
        z_mean = tf.layers.dense(inputs, n_z)
        vaeEstimateStd = True
        if vaeEstimateStd:
            z_sigma = tf.layers.dense(inputs, n_z)
            z_sigma = 1e-06 + tf.nn.softplus(z_sigma)
            vaeLoss = 0.5 * tf.reduce_mean(tf.square(z_mean) + tf.square(z_sigma) - tf.log(1e-08 + tf.square(z_sigma))) - 1
            inputs = tf.cond(self.training, lambda : z_mean + z_sigma * tf.random_normal(shape=(tf.shape(z_mean))), lambda : z_mean)
        else:
            inputs = tf.cond(self.training, lambda : z_mean + self.vaeStd * tf.random_normal(shape=(tf.shape(z_mean))), lambda : z_mean)
            vaeLoss = tf.reduce_mean(tf.square(z_mean))
        print('VAE mode 2, mean.shape', self._getShape(inputs))
        inputs = tf.nn.leaky_relu(tf.reshape(tf.layers.dense(inputs, sh[1] * sh[2] * sh[3] * sh[4]), [-1, sh[1], sh[2], sh[3], sh[4]]))
        inputs = self._conv(inputs, filters=filters, kernel_size=1, strides=1, use_bias=True)
        inputs = upsample.upsample_semilinear(inputs, data_format=(self.data_format), upsample_factor=2)
        print('VAE mode 2, inputs.shape', self._getShape(inputs))
        self.vaeLoss = vaeLoss
        for fidx, num_blocks in enumerate(blocks_up):
            filters //= 2
            with tf.variable_scope('VAEUpscaleConvOne_{}'.format(fidx)):
                inputs = self._conv(inputs, filters=filters, kernel_size=1, strides=1)
                inputs = upsample.upsample_semilinear(inputs, data_format=(self.data_format), upsample_factor=2)
            for i in range(num_blocks):
                s = inputs
                f = self._getNumberOfChannels(inputs, self.data_format)
                with tf.variable_scope('VAEUp0_{}_{}'.format(fidx, i)):
                    inputs = self._batch_norm_relu(inputs)
                    inputs = self._conv(inputs, filters=f, kernel_size=3, strides=1)
                with tf.variable_scope('VAEUp1_{}_{}'.format(fidx, i)):
                    inputs = self._batch_norm_relu(inputs)
                    inputs = self._conv(inputs, filters=f, kernel_size=3, strides=1)
                inputs += s
                print('VAELayerUp [', fidx, ':', i, '] shape', self._getShape(inputs))

        with tf.variable_scope('VAEFinal'):
            inputs = self._batch_norm_relu(inputs)
            inputs = self._conv(inputs, filters=(self._getNumberOfChannels(self.input, self.data_format)), kernel_size=1, strides=1,
              use_bias=True)
        self.autoEncoder = inputs
        self.reconstructionLoss = tf.losses.mean_squared_error(inputs, self.input)
        return outputs

    def loss(self):
        if self.use_vae:
            return self.vaeLoss + 0.1 * self.reconstructionLoss
        else:
            return 0
