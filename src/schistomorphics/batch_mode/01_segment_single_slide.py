#######!/usr/bin/env python
######coding: utf-8
# In[7]:
import os
from utils import util

HnE_png_dir="path/to/HnE"
label_dir="path/to/mask"

prob_thresh=0.3;nms_thresh=0.1
resultsDir="/projects/activities/deepspace/team/elaheh/results/"+results+projectName

img_out=imread(HnE_png_dir)
labels=misc.segment_cells(img_out,label_dir+'label.tif')### segmenting the HnE nuclei

