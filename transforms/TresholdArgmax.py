import numpy as np

# get common stuff you need
# from ai4med.common.constants import ImageProperty
# from ai4med.common.medical_image import MedicalImage
# from ai4med.common.shape_format import StdShapeFormat
from ai4med.common.transform_ctx import TransformContext

# get anything you need in libs
# from ai4med.libs.transforms.intensity_range_scaler import IntensityRangeScaler

# it is recommended to extend this base class to develop your own transformation
from ai4med.components.transforms.multi_field_transformer import MultiFieldTransformer


class TresholdArgmax(MultiFieldTransformer):
    """A template transform"""

    def __init__(self, fields, threshold = 0.5, dtype=np.float32):
        MultiFieldTransformer.__init__(self, fields)
        self.dtype = dtype
        self.threshold = threshold
        self.name = "TresholdArgmax"


    def transform(self, transform_ctx: TransformContext ):
        for field in self.fields:
            img = transform_ctx.get_image(field)  
            ##img is a MedicalImage object
            if not img.get_shape_format().is_3d():
                next
            if not img.get_shape_format().is_channeled():
                # not multichannel image=> apply threshold only.
                next
            data = img.get_data()
            print("data shape: {}".format(data.shape))
            # do something with the img
            channel_axis = img.get_shape_format().get_channel_axis()
            
            argMaxed_data = np.argmax( data, axis = channel_axis)
            argMaxed_data = argMaxed_data + 1
            less_than_threshold_data = data < self.threshold
            argMaxed_data[np.where(np.all(less_than_threshold_data, axis=channel_axis))] = 0
            argMaxed_data = np.expand_dims(argMaxed_data, axis=0)
            print('Applying {} on {}'.format(self.name, field))
            img.set_data(argMaxed_data, img.get_shape_format())
            transform_ctx.set_image(field, img)
        return transform_ctx

