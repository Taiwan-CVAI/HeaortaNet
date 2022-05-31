from ai4med.common.medical_image import MedicalImage
from ai4med.common.transform_ctx import TransformContext
from ai4med.components.transforms.multi_field_transformer import MultiFieldTransformer
from ai4med.common.constants import DataElementKey as Dek
import numpy as np

class CutOut(MultiFieldTransformer):

    def __init__(self, fields, size):
        '''
        For channel first and 3D image only
        size: size of expedted hole, ex: [10, 10, 10]
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
            if field == Dek.IMAGE:
                h, w, d = img_arr.shape[1:]
                # random center
                a = np.random.randint(h)
                b = np.random.randint(w)
                c = np.random.randint(d)
                a1 = np.clip(a - size[0] // 2, 0, h)
                a2 = np.clip(a + size[0] // 2, 0, h)
                b1 = np.clip(b - size[1] // 2, 0, w)
                b2 = np.clip(b + size[1] // 2, 0, w)
                c1 = np.clip(c - size[2] // 2, 0, d)
                c2 = np.clip(c + size[2] // 2, 0, d)
            # cut out the hole
                new_img = img_arr
                new_img[:, a1:a2, b1:b2, c1:c2] = np.random.beta(.5, .5, [a2-a1, b2-b1, c2-c1])-.5
            else:
                new_img = img_arr
                new_img[:, a1:a2, b1:b2, c1:c2] = 0
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
