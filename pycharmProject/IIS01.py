#coding:utf-8

import numpy as np
from collections import defaultdict
import matplotlib.image as mpimg
import colorsys     #装换颜色模型模块

## computing the (class-conditional) probabilities from the individual pdfs (followed by a normalization)
def comput_pdfs(imfile, imfile_scrib):
    '''
    compute foreground and background pdfs
    :param imfile: input file
    :param imfile_scrib: input file with user scrib
    :return:
    '''
    rgb = mpimg.imread(imfile)[:, :, :3]    #read the img
    yuv = colorsys.rgb_to_yiq(rgb)
    R = mpimg.imread(imfile)[:, :, 1]   #read the red channel