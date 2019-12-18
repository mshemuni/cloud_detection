# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:47:51 2019

@author: mshem
"""
from math import ceil

from PIL.ImageDraw import Draw as PIDraw
from PIL.Image import new as PInew

from numpy import float64 as f64
from numpy import ogrid
from numpy import sqrt
from numpy import power
from numpy import asarray as ar
from numpy import logical_not as lnot
from numpy import pi
from numpy import deg2rad
from numpy import rad2deg
from numpy import arctan2
from numpy import arccos
from numpy import arange
from numpy import cos
from numpy import sin

from . import ast


class Mask:
    """Mask class"""
    def __init__(self, logger):
        self.logger = logger
        self.ima = ast.Image(self.logger)

    def apply(self, data, mask, bkg=None):
        """Applies a mask to a given data"""
        try:
            self.logger.log("Applying mask")
            copy_od_data = data.copy()
            if bkg is not None:
                copy_od_data[mask] = bkg[mask]
            else:
                copy_od_data[mask] = 0

            return copy_od_data
        except Exception as excpt:
            self.logger.log(excpt)

class Geometric(Mask):
    """Geometric mask generator"""
    def circular(self, shape, center=None, radius=None,
                 bigger=0, auto=min, rev=False):
        """Creates a circular mask"""
        self.logger.log("Creating circular mask")
        try:
            h, w = shape
            if center is None:
                center = [int(w/2), int(h/2)]

            if radius is None:
                radius = auto(center[0], center[1], w-center[0], h-center[1])

            Y, X = ogrid[:h, :w]
            dist_from_center = sqrt(
                power(X - center[0], 2) + power(Y-center[1], 2))

            the_mask = dist_from_center <= radius + bigger

            if rev:
                return lnot(the_mask)
            else:
                return the_mask
        except Exception as excpt:
            self.logger.log(excpt)

    def polygon(self, shape, points, rev=False):
        """Creates a poligonial mask"""
        self.logger.log("Creating ploy mask")
        try:
            img = PInew('L', (shape[1], shape[0]), 0)
            PIDraw(img).polygon(points, outline=1, fill=1)
            mask = ar(img)

            the_mask = mask == 1

            if rev:
                return lnot(the_mask)
            else:
                return the_mask
        except Exception as excpt:
            self.logger.log(excpt)

class Polar(Mask):
    """Polar mask generator"""
    def pizza(self, shape, angle_range, center=None, radius=None,
              offset=90, auto=min, rev=False):
        """Creates a pizza slice mask"""
        self.logger.log("Creating pizza mask")
        try:
            w, h = shape
            x, y = ogrid[:w, :h]

            if center is None:
                center = [int(w/2), int(h/2)]

            if radius is None:
                radius = auto(center[0], center[1], w-center[0], h-center[1])

            cx, cy = center
            tmin, tmax = deg2rad(ar(angle_range) - offset)

            if tmax < tmin:
                tmax += 2 * pi

            r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)

            theta = arctan2(x - cx, y - cy) - tmin
            theta %= (2 * pi)

            circmask = r2 <= radius * radius
            anglemask = theta <= (tmax - tmin)

            the_mask = circmask*anglemask

            if rev:
                return lnot(the_mask)
            else:
                return the_mask
        except Exception as excpt:
            self.logger.log(excpt)

    def altaz(self, shape, altitude_range, azimut_range, center=None,
              radius=None, offset=90, auto=min):
        """Creates an AltAz mask"""
        self.logger.log("Creating AltAz mask")
        try:
            w, h = shape

            if center is None:
                center = [int(w/2), int(h/2)]

            if radius is None:
                radius = auto(center[0], center[1], w-center[0], h-center[1])

            r1 = radius * self.ima.projection(deg2rad(max(altitude_range)))
            r2 = radius * self.ima.projection(deg2rad(min(altitude_range)))
            r3 = radius

            mask1 = self.pizza(shape, azimut_range, center=center,
                               radius=r1, offset=offset, auto=auto) * 1
            mask2 = self.pizza(shape, azimut_range, center=center,
                               radius=r2, offset=offset, auto=auto) * 1
            mask3 = self.pizza(shape, azimut_range, center=center,
                               radius=r3, offset=offset, auto=auto) * 1

            the_sum = mask1 + mask2 + mask3

            return the_sum == 2
        except Exception as excpt:
            self.logger.log(excpt)

class SkySlicer:
    """Generate AltAz masks"""
    def __init__(self, logger):
        self.logger = logger

    def __alt_az_angles__(self, pieces):
        """Generates AltAz for given pieces"""
        self.logger.log("Calculating altitudes for {} pieces".format(pieces))
        try:
            return rad2deg(arccos(ar(range(0, pieces + 1)) / pieces))
        except Exception as excpt:
            self.logger.log(excpt)

    def __alt_az_pieces__(self, pieces):
        """Generates AltAz widenings"""
        self.logger.log("Calculating number of areas for {} pieces".format(pieces))
        try:
            return 1 + ar(range(0, pieces)) * 2
        except Exception as excpt:
            self.logger.log(excpt)

    def equal_area(self, pieces, inner_pieces=1):
        """Splits array to equal AltAz areas"""
        self.logger.log("Slicing to equal area pieces")
        try:
            altitude_angles = self.__alt_az_angles__(pieces)
            each_band_pieces = self.__alt_az_pieces__(pieces) * inner_pieces
            for altitude in range(len(altitude_angles) - 1):
                azimuth_angles = arange(0, 361,
                                        360 / each_band_pieces[altitude])
                for azimuth in range(len(azimuth_angles) - 1):
                    yield((tuple(altitude_angles[altitude:altitude + 2]),
                           tuple(azimuth_angles[azimuth:azimuth + 2])))
        except Exception as excpt:
            self.logger.log(excpt)

class HexagonSlicer:
    """Generates hexagon masks"""
    def __init__(self, logger):
        self.logger = logger

    def __create__(self, center, radus, ang_offset=0):
        """Creates corner points of a hexagon"""
        try:
            points = []
            for ang in range(0, 360, 60):
                angle = ang_offset + ang
                x = radus * cos(deg2rad(angle)) + center[1]
                y = radus * sin(deg2rad(angle)) + center[0]
                points.append([x, y])

            return ar(points)
        except Exception as excpt:
            self.logger.log(excpt)

    def fill_center(self, shape, hex_radius=250, ang_offset=0,
                    center=None, radius=None, auto=min):
        """Fills an array with circular hexagons"""
        self.logger.log("Filling the array with hexagons")
        try:
            w, h = shape
            if center is None:
                center = [int(w/2), int(h/2)]

            if radius is None:
                radius = auto(center[0], center[1], w-center[0], h-center[1])

            each_dist = hex_radius * sqrt(3) / 2

            ring_number = ceil(radius / (2 * hex_radius))
            number_of_hexs = ar(range(0, ring_number + 1)) * 6
            number_of_hexs[0] = 1
            for it, number_of_hex in enumerate(number_of_hexs):
                angle = f64((360/number_of_hex) % 360)
                R = it * 2 * each_dist
                for the_hex in range(number_of_hex):
                    each_ang = the_hex * angle + ang_offset
                    if each_ang%60 == 0:
                        r = R
                    else:
                        r = ((sqrt(3) / 2) * R) / cos(deg2rad(30-each_ang%60))

                    x = r * cos(deg2rad(each_ang))
                    y = r * sin(deg2rad(each_ang))
                    hpoints = self.__create__([center[0] + x, center[1] + y],
                                              hex_radius,
                                              ang_offset=0)
                    yield tuple(map(tuple, hpoints))
        except Exception as excpt:
            self.logger.log(excpt)
