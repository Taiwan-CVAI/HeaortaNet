from ai4med.common.medical_image import MedicalImage
from ai4med.common.transform_ctx import TransformContext
from ai4med.components.transforms.multi_field_transformer import MultiFieldTransformer
import numpy as np

class PadCropWithFixedSize(MultiFieldTransformer):

    def __init__(self, fields, size):
        '''
        size: size of expedted pad or crop region, ex: [224, 224, 144]
        '''
        MultiFieldTransformer.__init__(self, fields)
        self.size = size

    def transform(self, transform_ctx):
        size = self.size
        for field in self.fields:
            # get the MedicalImage using field
            img = transform_ctx.get_image(field)
            # get numpy array of the image
            img_arr = img.get_data()
#             if field==Dek.IMAGE:
#                 img_arr = (img_arr-img_arr.min())/(img_arr.max()-img_arr.min())
#             else:
#                 img_arr = img_arr[:-1]
            spatial_axis = img.get_shape_format().get_spatial_axis()
            channel_axis = img.get_shape_format().get_channel_axis()
            # works when channel first only
            new_img = np.zeros([img_arr.shape[channel_axis]]+size)
            ori_size = list(img_arr.shape[spatial_axis[0]:spatial_axis[1]+1])
            ori_size = [(s//2)*2 for s in ori_size]
            img_arr = img_arr[:, :ori_size[0], :ori_size[1], :ori_size[2]]
            new_img[:, 0 if size[0]//2-ori_size[0]//2<0 else size[0]//2-ori_size[0]//2:size[0]//2+ori_size[0]//2,
                    0 if size[1]//2-ori_size[1]//2<0 else size[1]//2-ori_size[1]//2:size[1]//2+ori_size[1]//2,
                    0 if size[2]//2-ori_size[2]//2<0 else size[2]//2-ori_size[2]//2:size[2]//2+ori_size[2]//2] = \
                    img_arr[:, 0 if ori_size[0]//2-size[0]//2<0 else ori_size[0]//2-size[0]//2:ori_size[0]//2+size[0]//2,
                            0 if ori_size[1]//2-size[1]//2<0 else ori_size[1]//2-size[1]//2:ori_size[1]//2+size[1]//2,
                            0 if ori_size[2]//2-size[2]//2<0 else ori_size[2]//2-size[2]//2:ori_size[2]//2+size[2]//2]

            # create a new MedicalImage use new_image() method
            # which will carry over the properties of the original image
            result_img = img.new_image(new_img, img.get_shape_format())

            # set the image back in transform_ctx
            transform_ctx.set_image(field, result_img)
        return transform_ctx

    def is_deterministic(self):
        """ This is not a deterministic transform.

        Returns:
            False (bool)
        """
        return False
