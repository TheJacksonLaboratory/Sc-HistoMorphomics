#######!/usr/bin/env python
######coding: utf-8
from tifffile import imread
from tifffile import imwrite
import os
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import os
import sys
import cellprofiler_core
import cellprofiler_core.utilities.java
import numpy as np
from csbdeep.utils import normalize
from utils import util

HnE_png_dir="path/to/HnE"
label_dir="path/to/mask"
measurement_dir="path/to/measurement"

img_out=imread(HnE_png_dir)
labels=imread(labelDir+projectName+'label2.tif').astype(np.int32)
min_label = np.abs(np.min(labels)) + 1; labels = labels + min_label
labels[np.where(labels == min_label)] = 0

Name="_nuclei_features"

if not(os.path.isfile(measurement_dir+".csv")):
    print("Cell Profiler measurements do not exist! Calculating...")
    cellprofiler_core.utilities.java.start_java()
    featu,skip_last_chunck=misc.Compute_CP_dask(img_out,labels,chunks=(7000,7000,4))
    All2,Hemotoxylin,Eosin,cyto=misc.clean_up_morphs_no_ST_data_Cyto(featu,MeasDir+projectName+Name,labelDir+projectName,chunks=7000,shape=img_out.shape[0:2])
    cellprofiler_core.utilities.java.stop_java()
