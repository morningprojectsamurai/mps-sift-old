from collections import OrderedDict
import numpy as np
from scipy.ndimage import gaussian_filter


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'
__status__ = 'development'
__version__ = '0.0.1'
__date__ = '16 December 2016'


class ImgSize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @property
    def row(self):
        return self.height

    @property
    def col(self):
        return self.width

    @property
    def shape(self):
        return (self.row, self.col)

    def __eq__(self, other):
        if isinstance(other, (tuple, list, np.ndarray)):
            return other == self.shape
        elif isinstance(other, ImgSize):
            return other.shape == self.shape
        else:
            raise NotImplemented

    def __lt__(self, other):
        if isinstance(other, (tuple, list, np.ndarray)):
            return self.row < other[0] and self.col < other[1]
        elif isinstance(other, ImgSize):
            return self.row < other.row and self.col < other.col
        else:
            raise NotImplemented

    def __gt__(self, other):
        if isinstance(other, (tuple, list, np.ndarray)):
            return self.row > other[0] and self.col > other[1]
        elif isinstance(other, ImgSize):
            return self.row > other.row and self.col > other.col
        else:
            raise NotImplemented
        

class ScaleSpace:
    def __init__(self, img, sigma=1.6, s=2, min_img_size=ImgSize(16, 16)):
        """Constructor
        
        sigma: base for calculating scales
        s: number of images in the stack (s + 3 images in the stack)
        min_img_size: minimum image size in the space. ImgSize instance.
        """
        self._sigma = sigma
        self._s = s
        self._min_img_size = min_img_size
        self.octaves = []

        self._generate(img)

    @property
    def k(self):
        """Factor to generate bullered images.
        """
        return np.power(2, 1/self._s)
    
    def _generate(self, img):
        assert(isinstance(img, np.ndarray))
        scale, orig_img = 0.0, img
        while orig_img.shape > self._min_img_size:
            octave = OrderedDict()
            octave[scale] = orig_img
            for i in range(self._s + 2):
                scale = np.power(self.k, i) * self._sigma
                octave[scale] = gaussian_filter(orig_img, scale)
            assert(len(octave) == self._s + 2 + 1)
            self.octaves.append(octave)
            scale, orig_img = 0.0, list(octave.items())[-2][1][::2, ::2]

    def get_value(self, p):
        o, s, r, c = p
        return self.octaves[o][s][r, c]
            
    def next_scale(self, o, s):
        scales = list(self.octaves[o].keys())
        return scales[scales.index(s) + 1]
