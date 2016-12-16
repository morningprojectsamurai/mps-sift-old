import numpy as np
from scipy.misc import imread, imsave
from sift.dogspace import DogSpace


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'
__status__ = 'development'
__version__ = '0.0.1'
__date__ = '16 December 2016'


class TestDogSpace:
    def setUp(self):
        self.test_img = imread('img/lenna.png', True)
        assert(self.test_img.shape[0] == 512)
        assert(self.test_img.shape[1] == 512)

    def tearDown(self):
        pass

    def test_init(self):
        dogspace = DogSpace(self.test_img)
        for o, octave in enumerate(dogspace.dog_octaves):
            for scale, img in octave.items():
                imsave('img/dogspace/%s_%s.jpg' % (o, scale), img)
