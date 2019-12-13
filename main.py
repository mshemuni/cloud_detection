# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:52:07 2019

@author: mshem
"""

from numpy import sin
from numpy import cos
from numpy import deg2rad

from dag_cld import env
from dag_cld import ast
from dag_cld import mask


logger = env.Logger(blabla=True)

fts = ast.Fits(logger)
ima = ast.Image(logger)

gmask = mask.Geometric(logger)
pmask = mask.Polar(logger)


file = "data/2018_07_14__11_50_41.fits"

data = fts.data(file)
gimage = ima.rgb2gray(ima.array2rgb(data))

ima.show(gimage)

points = ()

r = 750

for i in range(0, 360, 60):
    x = r * cos(deg2rad(i)) + gimage.shape[1] / 2
    y = r * sin(deg2rad(i)) + gimage.shape[0] / 2
    
    points += ((x, y), )
    
print(points)

polymask = gmask.polygon(gimage.shape, points, rev=True)

new_image = gmask.apply(gimage, polymask)
c_image = ima.find_window(new_image, polymask)

ima.show(new_image)
ima.show(c_image)



#cir_mask = gmask.circular(gimage.shape, rev=True)
#pizza_mask = pmask.pizza(gimage.shape, (45, 90), rev=False)
#
#masked_data = pmask.apply(gimage, pizza_mask)
#
##ima.show(masked_data)
#
##the_mask = pmask.altaz(gimage.shape, (0, 90), (45, 90), offset=0, rev=True)
##masked_data = pmask.apply(gimage, the_mask)
##cc = ima.alt_az_crop(gimage.shape, (0, 90), (45, 90), offset=0)
##ima.show(masked_data)
#
##piza_mask = pmask.pizza(gimage.shape, (0, 90))
##pmasked_data = pmask.apply(gimage, piza_mask)
##ima.show(pmasked_data)
#
##for alt in range(0, 90, 5):
##    for az in range(0, 180, 5):
##        aiza_mask = pmask.altaz(gimage.shape, (alt, alt + 5), (az, az + 5))
##        cc = ima.alt_az_crop(gimage.shape, (alt, alt + 5), (az, az + 5), offset=0)
##        amasked_data = pmask.apply(gimage, aiza_mask)
##        ima.show(amasked_data)
#
#alt_range = (0, 45)
#az_range = (0, 45)
#
#aiza_mask = pmask.altaz(gimage.shape, alt_range, az_range)
#amasked_data = pmask.apply(gimage, aiza_mask)
#c_image = ima.find_window(amasked_data, aiza_mask)
#ima.show(amasked_data)
#ima.show(c_image)
