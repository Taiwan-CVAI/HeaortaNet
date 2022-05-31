import tensorflow as tf
from ai4med.common.data_format import DataFormat
from ai4med.common.constants import ActivationFunc
from custom_models import heaortanet
from ai4med.common.build_ctx import BuildContext
from ai4med.common.graph_component import GraphComponent
from ai4med.components.models.model import Model

class HeaortaNet(Model):
    __module__ = __name__
    __qualname__ = 'SegResnet'

    def __init__(self, num_classes, blocks_down='1,2,2,4', blocks_up='1,1,1', init_filters=8, use_batch_norm=False, use_group_norm=True, use_group_normG=8, reg_weight=0.0, dropout_prob=0.0, final_activation=ActivationFunc.SOFTMAX, use_vae=False, dtype=tf.float32, data_format=DataFormat.CHANNELS_FIRST):
        Model.__init__(self)
        self.num_classes = num_classes
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.init_filters = init_filters
        self.use_batch_norm = use_batch_norm
        self.use_group_norm = use_group_norm
        self.use_group_normG = use_group_normG
        self.reg_weight = reg_weight
        self.dropout_prob = dropout_prob
        self.final_activation = final_activation
        self.use_vae = use_vae
        self.dtype = dtype
        self.data_format = data_format
        self.model = None

    def get_predictions(self, inputs, is_training, build_ctx):
        self.model = heaortanet.HeaortaNet(inputs=inputs,
          num_classes=(self.num_classes),
          training=is_training,
          blocks_down=(self.blocks_down),
          blocks_up=(self.blocks_up),
          init_filters=(self.init_filters),
          use_batch_norm=(self.use_batch_norm),
          use_group_norm=(self.use_group_norm),
          use_group_normG=(self.use_group_normG),
          reg_weight=(self.reg_weight),
          dropout_prob=(self.dropout_prob),
          final_activation=(self.final_activation),
          use_vae=(self.use_vae),
          dtype=(self.dtype),
          data_format=(self.data_format))
        return self.model.model()

    def get_loss(self):
        """Get the additional loss function in AHNet model.

        Args:
            None

        Returns:
            Loss function

        """
        return self.model.loss()

