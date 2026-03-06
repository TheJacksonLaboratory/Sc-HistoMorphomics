# Sc-HistoMorphomics
sc‑HistoMorphomics, a scalable framework for profiling nuclear morphology in whole‑slide H&amp;E images and correlating these measurements with spatial transcriptomics data.


# Installation
## 1. Create and activate environment
conda create -n stardist python=3.8 -y
conda activate stardist

## 2. Upgrade pip
pip install --upgrade pip

## 3. Install TensorFlow
pip install tensorflow

## 4. Install wxPython (needed for CellProfiler GUI)
wget https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.1.0-cp38-cp38-linux_x86_64.whl
pip install wxPython-4.1.0-cp38-cp38-linux_x86_64.whl

## 5. Install CellProfiler (specific version)
pip install git+https://github.com/CellProfiler/CellProfiler.git@v4.2.5

## 6. Install additional dependencies
conda install dask -y
pip install opencv-python
conda install -c conda-forge python-javabridge -y

## 7. Install supporting libraries
pip install zarr
pip install geojson
pip install imagecodecs
