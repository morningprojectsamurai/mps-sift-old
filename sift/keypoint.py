import numpy as np


__author__ = 'Junya Kaneko <jyuneko@hotmail.com>'
__status__ = 'development'
__version__ = '0.0.1'
__date__ = '16 December 2016'


class KeypointDetector:
    def __init__(self, dogspace, low_contrast_th=0.03, r=10.0):
        self._dogspace = dogspace
        self._low_contrast_th = low_contrast_th
        self._r = r
        self._extrema = []

        self._find_local_extrema()
        self._eliminate_low_contrast()
        self._eliminate_edge_response()

    def _find_local_extrema(self):
        for o, octave in enumerate(self._dogspace.dog_octaves):
            scales = list(octave.keys())
            for i in range(1, len(scales) - 1):
                prev_img = octave[scales[i-1]]
                img = octave[scales[i]]
                next_img = octave[scales[i+1]]
                for r in range(1, img.shape[0] - 1):
                    for c in range(1, img.shape[1] - 1):
                        if (np.max(img[r-1:r+2, c-1:c+2]) == img[r, c] and
                            np.max(prev_img[r-1:r+2, c-1:c+2]) < img[r, c] and
                            np.max(next_img[r-1:r+2, c-1:c+2]) < img[r, c]) or \
                            (np.min(img[r-1:r+2, c-1:c+2]) == img[r, c] and
                             np.min(prev_img[r-1:r+2, c-1:c+2]) > img[r, c] and
                             np.min(next_img[r-1:r+2, c-1:c+2]) > img[r, c]):
                            self._extrema.append((o, scales[i], r, c))

    def _eliminate_low_contrast(self):
        for i, p in enumerate(self._extrema):
            if np.abs(self._dogspace.get_value(p)) < self._low_contrast_th:
                self._extrema.pop(i)

    def _eliminate_edge_response(self):
        for i, p in enumerate(self._extrema):
            o, s, r, c = p
            Drr = self._dogspace.octaves[o][s][r - 1, c] \
                  - 2 * self._dogspace.octaves[o][s][r, c] \
                  + self._dogspace.octaves[o][s][r + 1, c]
            Dcc = self._dogspace.octaves[o][s][r, c - 1] \
                  - 2 * self._dogspace.octaves[o][s][r, c] \
                  + self._dogspace.octaves[o][s][r, c + 1]
            Drc = self._dogspace.octaves[o][s][r + 1, c + 1] \
                  - self._dogspace.octaves[o][s][r, c + 1] \
                  - self._dogspace.octaves[o][s][r + 1, c] \
                  + self._dogspace.octaves[o][s][r, c]
            value = np.power(Drr + Dcc, 2) / (Drr * Dcc - np.power(Drc, 2)) 
            if value < np.power(self._r + 1, 2) / self._r:
                self._extrema.pop(i)
            

class Descriptor:
    pass
