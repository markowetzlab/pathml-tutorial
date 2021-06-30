Learning PathML, a Python library for deep learning on whole-slide images
=====
[PathML](https://github.com/markowetzlab/pathml) is a Python library for performing deep learning image analysis on whole-slide images (WSIs), including deep tissue, artefact, and background filtering, tile extraction, model inference, model evaluation and more. This repository serves to teach users how to apply `PathML` on both a classification and a segmentation example problem from start to finish using best practices.

<p align="center">
  <img src="https://github.com/markowetzlab/pathml-tutorial/blob/master/figures/figure1.png" width="500" />
</p>

Installing PathML and its depedencies
----
Install PathML by cloning its repository:
```
git clone https://github.com/markowetzlab/pathml
```

PathML is best run inside an Anaconda environment. Once you have [installed Anaconda](https://docs.anaconda.com/anaconda/install), you can create `pathml-env`, a conda environment containing all of PathML's dependencies, then activate that environment. Make sure to adjust the path to your local path to the pathml repository:
```
conda env create -f /path/to/pathml/pathml-environment.yml
conda activate pathml-env
```
Note that `pathml-environment.yml` installs Python version 3.7, PyTorch version 1.4, Torchvision version 0.5, and CUDA version 10.0. Stable versions above these should also work as long as the versions are cross-compatible. Be sure that the CUDA version matches the version installed on your GPU; if not, either update your GPU's CUDA or change the `cudatoolkit` line of \codeword{pathml-environment.yml} to match your GPU's version before creating `pathml-env`.

Running the PathML tutorial
----
First clone this repository:
```
git clone https://github.com/markowetzlab/pathml-tutorial
```
The tutorial uses an example subset lymph node WSIs from the [CAMELYON16 challenge](https://camelyon16.grand-challenge.org/). Some of these WSIs contain breast cancer metastases and the goal of the tutorial is to use PathML to train deep learning models to identify metastasis-containing slides and slide regions, and then to evaluate the performance of those models.

Create a directory called `wsi_data` where there is at least 38 GB of disk space. Download the following 18 WSIs from the [CAMELYON16 dataset](https://drive.google.com/drive/folders/0BzsdkU4jWx9Ba2x1NTZhdzQ5Zjg?resourcekey=0-g2TRih6YKi5P2O1SiBB1LA) into `wsi_data`:

* `normal/normal_001.tif`
* `normal/normal_010.tif`
* `normal/normal_028.tif`
* `normal/normal_037.tif`
* `normal/normal_055.tif`
* `normal/normal_074.tif`
* `normal/normal_111.tif`
* `normal/normal_141.tif`
* `normal/normal_160.tif`
* `tumor/tumor_009.tif`
* `tumor/tumor_011.tif`
* `tumor/tumor_036.tif`
* `tumor/tumor_039.tif`
* `tumor/tumor_044.tif`
* `tumor/tumor_046.tif`
* `tumor/tumor_058.tif`
* `tumor/tumor_076.tif`
* `tumor/tumor_085.tif`

Install Jupyter notebook into `pathml-env`:
```
conda install -c conda-forge notebook
```
Now that the requisite software and data have been downloaded, you are ready to begin the tutorial, which is contained in the Jupyter notebook `pathml-tutorial.ipynb` in this repository. Start notebook and then navigate to that document in the interface:
```
jupyter notebook
```
Once up and running, `pathml-tutorial.ipynb` contains instructions for running the tutorial. For instructions on running Jupyter notebooks, see the [Jupyter documentation](https://jupyter.org/documentation).

Disclaimer
----
Note that this is prerelease software. Please use accordingly.
