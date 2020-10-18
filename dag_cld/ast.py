from logging import getLogger

import numpy as np

from astropy.io import fits
from astropy import stats
from skimage import color
from skimage import feature

from sep import Background

from cv2 import resize as cvresize

from matplotlib import pyplot as plt

class Image:
    def __init__(self, file, logger=None):
        self.logger = logger or getLogger("dummy")
        self.file = file
        self.array = self.open()
        self.__original = self.copy()

    def header(self, field="*"):
        ret = []
        try:
            hdu = fits.open(self.file, mode='readonly')
            header = hdu[0].header
            hdu.close()

            if field == "?":
                return header

            ret = [[i, header[i]] for i in header if not i == ""]

            if isinstance(field, str):
                if field == "*":
                    return ret
                else:
                    return header[field]
            else:
                raise ValueError("A string bust be given as header card")

        except Exception as e:
            self.logger.error(e)


    def __str__(self):
        s = f"A {self.array.ndim}D, with shape of {self.array.shape}, array"
        s += f"\nSize: {self.array.size}"
        s += f"\nMax: {self.array.max()}"
        s += f"\nMin: {self.array.min()}"
        s += f"\nAverage: {self.array.mean()}"

        return s

    def shape(self, array=None):
        try:
            if array is not None:
                return array.shape

            return self.array.shape
        except Exception as e:
            self.logger.error(e)

    def flatten(self, array=None):
        try:
            if array is not None:
                return array.flatten()

            self.array = self.array.flatten()
        except Exception as e:
            self.logger.error(e)

    def open(self):
        try:
            hdu = fits.open(self.file, "readonly")
            data = hdu[0].data
            if data.ndim == 3:
                return color.rgb2gray(data).astype(np.float)
            elif data.ndim == 2:
                return data.astype(np.float)
            else:
                raise ValueError("Unknown Image type")
        except Exception as e:
            self.logger.error(e)

    def remove_background(self, array=None, min=0):
        try:
            if array is not None:
                if min is not None:
                    return self.__set_min(array - self.background(array=array), min)
                else:
                    return array - self.background(array=array)

            if min is not None:
                self.array = self.__set_min(self.array - self.background(self.array), min)
            else:
                self.array = self.array - self.background(self.array)
        except Exception as e:
            self.logger.error(e)

    def __set_min(self, arr, value):
        try:
            use_arr = arr.copy()
            use_arr[use_arr < value] = value
            return use_arr
        except Exception as e:
            self.logger.error(e)

    def background(self, array=None, isar=True):
        try:
            if array is not None:
                if isar:
                    return np.array(Background(array))
                else:
                    return Background(array)

            if isar:
                return np.array(Background(self.array))
            else:
                return Background(self.array)

        except Exception as e:
            self.logger.error(e)

    def cal_shape(self, shape, array=None):
        if isinstance(shape, tuple):
            try:
                if array is not None:
                    return tuple([int(s * p / 100) for p, s in zip(shape, array.shape)])

                return tuple([int(s * p / 100) for p, s in zip(shape, self.array.shape)])
            except Exception as e:
                self.logger.error(e)
        elif isinstance(shape, (float, int)):
            try:
                if array is not None:
                    return tuple([int(s * shape / 100) for s in array.shape])

                return tuple([int(s * shape / 100) for s in self.array.shape])
            except Exception as e:
                self.logger.error(e)
        else:
            raise ValueError("Can't parse shape")

    def resize(self, shape, array):
        try:
            if array is not None:
                return cvresize(array, shape)

            self.array = cvresize(self.array, shape)
        except Exception as e:
            self.logger.error(e)

    def copy(self):
        try:
            return self.array.copy()
        except Exception as e:
            self.logger.error(e)

    def blank(self, shape, fill=0):
        try:
            return np.ones(shape) * fill
        except Exception as e:
            self.logger.error(e)

    def histogram(self, array=None, bins=20):
        try:
            if array is not None:
                return stats.histogram(array, bins=bins)

            return stats.histogram(self.array, bins=bins)
        except Exception as e:
            self.logger.error(e)

    def nomalize(self, array):
        try:
            if array is not None:
                row_sums = array.sum(axis=1)
                return array / row_sums[:, np.newaxis]

            row_sums = self.array.sum(axis=1)
            return self.array / row_sums[:, np.newaxis]
        except Exception as e:
            self.logger.error(e)

    def hog(self, array=None, mchannel=False, ppc=(16, 16)):
        try:
            if isinstance(ppc, int):
                ppc = (ppc, ppc)

            if array is not None:
                return feature.hog(array, orientations=8,
                                   pixels_per_cell=ppc,
                                   cells_per_block=(1, 1), visualize=True,
                                   multichannel=mchannel)

            return feature.hog(self.array, orientations=8,
                               pixels_per_cell=ppc,
                               cells_per_block=(1, 1), visualize=True,
                               multichannel=mchannel)

        except Exception as e:
            self.logger.error(e)

    def show(self, array=None):
        try:
            if array is not None:
                the_array = array
            else:
                the_array = self.array

            m, s = np.mean(the_array), np.std(the_array)
            plt.imshow(the_array, interpolation='nearest',
                       cmap='gray', vmin=m - s, vmax=m + s, origin='lower')
            plt.show()

        except Exception as e:
            self.logger.error(e)
