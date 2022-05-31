from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance
from ai4med.components.losses.loss import Loss

def dice_loss(predictions,
              targets,
              data_format='channels_first',
              skip_background=False,
              squared_pred=False,
              jaccard=False,
              smooth=1e-5,
              top_smooth=0.0,
              is_onehot_targets=False):
    """Compute average Dice loss between two tensors.

    5D tensors (for 3D images) or 4D tensors (for 2D images).

    Args:
        predictions (Tensor): Tensor of Predicted segmentation output (e.g NxCxHxWxD)
        targets (Tensor): Tensor of True segmentation values. Usually has 1 channel dimension (e.g. Nx1xHxWxD),
                        where each element is an index indicating class label.
                        Alternatively it can be a one-hot-encoded tensor of the shape NxCxHxWxD,
                        where each channel is  binary (or float in interval 0..1) indicating
                        the probability of the corresponding class label
        data_format (str): channels_first (default) or channels_last
        skip_background (bool): skip dice computation on the first channel of the predicted output or not
        squared_pred (bool): use squared versions of targets and predictions in the denominator or not
        jaccard (bool): compute Jaccard Index (soft IoU) instead of dice or not
        smooth (float): denominator constant to avoid zero division (default 1e-5)
        top_smooth (float): experimental, nominator constant to avoid zero final loss when targets are all zeros
        is_onehot_targets (bool): the targets are already One-Hot-encoded or not

    Returns:
        tensor of one minus average dice loss

    """

    is_channels_first = (data_format == 'channels_first')
    ch_axis = 1 if is_channels_first else -1

    n_channels_pred = predictions.get_shape()[ch_axis].value
    n_channels_targ = targets.get_shape()[ch_axis].value
    n_len = len(predictions.get_shape())

    print('dice_loss targets', targets.get_shape().as_list(),
          'predictions', predictions.get_shape().as_list(),
          'targets.dtype', targets.dtype,
          'predictions.dtype', predictions.dtype)

    print('dice_loss is_channels_first:', is_channels_first,
          'skip_background:', skip_background,
          'is_onehot_targets', is_onehot_targets)

    # Sanity checks
    if skip_background and n_channels_pred == 1:
        raise ValueError("There is only 1 single channel in the predicted output, and skip_zero is True")
    if skip_background and n_channels_targ == 1 and is_onehot_targets:
        raise ValueError("There is only 1 single channel in the true output (and it is is_onehot_true), "
                         "and skip_zero is True")
    if is_onehot_targets and n_channels_targ != n_channels_pred:
        raise ValueError("Number of channels in target {} and pred outputs {} "
                         "must be equal to use is_onehot_true == True".format(
                            n_channels_targ, n_channels_pred))

    # End sanity checks
    if not is_onehot_targets:
        # if not one-hot representation already
        # remove singleton (channel) dimension for true labels
        targets = tf.cast(tf.squeeze(targets, axis=ch_axis), tf.int32)
        targets = tf.one_hot(targets, depth=n_channels_pred, axis=ch_axis,
                             dtype=tf.float32, name="loss_dice_targets_onehot")

    if skip_background:
        # if skipping background, removing first channel
        targets = targets[:, 1:] if is_channels_first else targets[..., 1:]
        predictions = predictions[:, 1:] if is_channels_first else predictions[..., 1:]

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, n_len)) if is_channels_first else list(range(1, n_len - 1))

    intersection = tf.reduce_sum(targets * predictions, axis=reduce_axis)

    if squared_pred:
        # technically we don't need this square for binary true values
        # (but in cases where true is probability/float, we still need to square
        targets = tf.square(targets)
        predictions = tf.square(predictions)

    y_true_o = tf.reduce_sum(targets, axis=reduce_axis)
    y_pred_o = tf.reduce_sum(predictions, axis=reduce_axis)

    denominator = y_true_o + y_pred_o

    if jaccard:
        denominator -= intersection

    f = (2.0 * intersection + top_smooth) / (denominator + smooth)

    # # If only compute dice for present label, mask out data-label that are not present
    # if only_present:
    #     dice_mask = tf.logical_not(tf.equal(label_sum, 0))
    #     dice = tf.boolean_mask(dice, dice_mask)

    f = tf.reduce_mean(f)  # final reduce_mean across batches and channels

    return 1 - f


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)
    
def boundary_loss(predictions, targets):
    '''
    for channel-first only
    '''
    loss = 0
    for c in range(predictions.shape[1]):
        y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                         inp=[targets[:, c, ...]],
                                         Tout=tf.float32)
        multipled = predictions[:, c, ...] * y_true_dist_map
        loss += K.mean(multipled)
    return loss

class BoundaryDiceLoss(Loss):
    """Compute average Dice loss between two tensors.

    5D tensors (for 3D images) or 4D tensors (for 2D images).

    Args:
        data_format (str): channels_first (default) or channels_last
        skip_background (bool): skip dice computation on the first channel of the predicted output or not
        squared_pred (bool): use squared versions of targets and predictions in the denominator or not
        jaccard (bool): compute Jaccard Index (soft IoU) instead of dice or not
        smooth (float): denominator constant to avoid zero division (default 1e-5)
        top_smooth (float): experimental, nominator constant to avoid zero final loss when targets are all zeros
        is_onehot_targets (bool): the targets are already One-Hot-encoded or not

    Returns:
        tensor of one minus average dice loss

    """

    def __init__(self,
                 alpha=0.02,
                 data_format='channels_first',
                 skip_background=False,
                 squared_pred=False,
                 jaccard=False,
                 smooth=1e-5,
                 top_smooth=0.0,
                 is_onehot_targets=False):
        Loss.__init__(self)
        self.alpha = alpha
        self.data_format = data_format
        self.skip_background = skip_background
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth = smooth
        self.top_smooth = top_smooth
        self.is_onehot_targets = is_onehot_targets

    def get_loss(self, predictions, targets, build_ctx=None):
        """Compute dice loss for tf variable

        Args:
            predictions (Tensor): outputs of the network
            targets (Tensor): target integer labels
            build_ctx: specified graph context

        Returns:
            tensor of dice loss

        """
        loss = self.alpha*boundary_loss(predictions, targets)+\
               dice_loss(predictions, targets,
                         data_format=self.data_format,
                         skip_background=self.skip_background,
                         squared_pred=self.squared_pred,
                         jaccard=self.jaccard,
                         smooth=self.smooth,
                         top_smooth=self.top_smooth,
                         is_onehot_targets=self.is_onehot_targets)
        return loss
