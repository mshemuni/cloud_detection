# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:47:51 2019

@author: mshem
"""
from PIL.ImageDraw import Draw as PIDraw
from PIL.Image import new as PInew

from numpy import ogrid
from numpy import sqrt
from numpy import power
from numpy import asarray as ar
from numpy import logical_not as lnot
from numpy import pi
from numpy import deg2rad
from numpy import arctan2
from numpy import sin

from . import ast

class Mask:
    def __init__(self, logger):
        self.logger = logger
        self.ima = ast.Image(self.logger)
        
    def apply(self, data, mask, bkg=None):
        try:
            copy_od_data = data.copy()
            if bkg is not None:
                copy_od_data[mask] = bkg[mask]
            else:
                copy_od_data[mask] = 0
                
            return(copy_od_data)
        except Exception as e:
            self.logger.log(e)

class Geometric(Mask):
    def circular(self, shape, center=None, radius=None,
                 bigger=0, auto=min, rev=False):
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
                return(lnot(the_mask))
            else:
                return(the_mask)
        except Exception as e:
            self.logger.log(e)
            
    def polygon(self, shape, points, rev=False):
        try:
            
            img = PInew('L', (shape[1], shape[0]), 0)
            PIDraw(img).polygon(points, outline=1, fill=1)
            mask = ar(img)
            
            the_mask = mask == 1
            
            if rev:
                return(lnot(the_mask))
            else:
                return(the_mask)
        except Exception as e:
            self.logger.log(e)
            
class Polar(Mask):

    def pizza(self, shape, angle_range, center=None, radius=None,
                    offset=90, auto=min, rev=False):
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
                return(lnot(the_mask))
            else:
                return(the_mask)
        except Exception as e:
            self.logger.log(e)
            
    def altaz(self, shape, altitude_range, azimut_range, center=None,
              radius=None, offset=90, auto=min):
        try:
            w, h = shape
            
            if center is None:
                center = [int(w/2), int(h/2)]
            
            if radius is None:
                radius = auto(center[0], center[1], w-center[0], h-center[1])
                
            r1 = radius * self.ima.projection(
                    deg2rad(max(altitude_range)))
            r2 = radius * self.ima.projection(
                    deg2rad(min(altitude_range)))
            r3 = radius
            
            mask1 = self.pizza(shape, azimut_range, center=center,
                               radius=r1, offset=offset, auto=auto) * 1
            mask2 = self.pizza(shape, azimut_range, center=center,
                               radius=r2, offset=offset, auto=auto) * 1
            mask3 = self.pizza(shape, azimut_range, center=center,
                               radius=r3, offset=offset, auto=auto) * 1
            
            the_sum = mask1 + mask2 + mask3
            
            return(lnot(the_sum==2))
            
        except Exception as e:
            self.logger.log(e)
        