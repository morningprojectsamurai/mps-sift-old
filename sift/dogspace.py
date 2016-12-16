from collections import OrderedDict
from sift.scalespace import ScaleSpace, ImgSize


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'
__status__ = 'development'
__version__ = '0.0.1'
__date__ = '16 December 2016'


class DogSpace(ScaleSpace):
    def __init__(self, img, sigma=1.6, s=2, min_img_size=ImgSize(16, 16)):
        super(DogSpace, self).__init__(img, sigma, s, min_img_size)
        self.dog_octaves = []
        self._apply_dog()
    
    def _apply_dog(self):
        for octave in self.octaves:
            dog_octave = OrderedDict()
            prev_scale, prev_img = None, None
            for scale, img in octave.items():
                if prev_scale is None and prev_img is None:
                    prev_scale, prev_img = scale, img
                else:
                    dog_octave[prev_scale] = img - prev_img
                    prev_scale, prev_img = scale, img
            self.dog_octaves.append(dog_octave)

