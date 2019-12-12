# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:52:07 2019

@author: mshem
"""

from dag_cld import env
from dag_cld import ast
from dag_cld import mask


logger = env.Logger(blabla=True)

fts = ast.Fits(logger)
ima = ast.Image(logger)

gmask = mask.Geometric(logger)
pmask = mask.Polar(logger)


file = "2018_07_14__11_50_41.fits"

data = fts.data(file)
gimage = ima.rgb2gray(ima.array2rgb(data))

#cir_mask = gmask.circular(gimage.shape, rev=True)
pol_mask = pmask.pizza(gimage.shape, (45, 90), rev=False)

masked_data = pmask.apply(gimage, pol_mask)

ima.show(masked_data)

