# Coastal Mapping
## _Estimate the coastline for the Artic National Wildlife Refuge_

## Project Structure
```
coastal_mapping
│
└─── coastal _mapping
│   └───data
│       │   data.py
│       │   slice.py
│   └───model
│       │   frame.py
│       │   functions.py
│       │   metrics.py
│       │   unet.py
│   
└───conf
│   │   eval.yaml
│   │   ml_prepareXY.yaml
│   │   ml_train.yaml
│   │   predict_slices.yaml
│   │   slice_and_preprocess.yaml
│   │   unet_predict.yaml
│   │   unet_train.yaml
│   
│   .gitignore
│   README.md
│   requirements.txt
│   win_requirements.txt
│   eval.py
│   ml_prepareXY.py
│   ml_train.py
│   predict_slices.py
│   slice_and_preprocess.py
│   unet_predict.py
│   unet_train.py
```

## Steps to duplicate pipeline on a new machine

### Installing requirements on Ubuntu machine
<pre>
* sudo apt update                                                   Install python pip, setuptools
* sudo apt install python3-pip
* sudo python3 -m pip install -U pip
* sudo python3 -m pip install -U setuptools
* Download Anaconda from https://www.anaconda.com/products/individual
* conda create --name py36 python=3.6                               Create a new Anaconda environment for python 3.6
* conda activate py36
* sudo apt-get install gdal-bin
* git clone https://github.com/Aryal007/coastal_mapping.git         Clone Repository
* cd coastal_mapping                                                Change directory to coastal mapping
* pip3 install -r requirements.txt                                  Install all the necessary requirements
</pre>

### Install requirements on a Windows machine
<pre>
* Download Anaconda from https://www.anaconda.com/products/individual
* git clone https://github.com/Aryal007/coastal_mapping.git         Clone Repository
* Run Anaconda Powershell prompt and navigate to the directory
* conda create --name py36 python=3.6                               Create a new Anaconda environment for python 3.6
* conda activate py36
* conda install -c pytorch pytorch torchvision
* conda install -c anaconda scikit-learn
* conda config --add channels conda-forge
* conda install --file win_requirements.txt 
</pre>

```
To install nvidia drivers on compatible Azure virtual machine, 
https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
```

### Steps for training and evaluation
On a windows machine, use python instead of python3
</pre>
Required for training
* python3 slice_and_preprocess.py                                   Create slices, configuration: conf/slice.yaml
* python3 unet_train.py                                             Train model, configuration: conf/train.yaml
* python3 unet_predict.py                                           Generate masks for new image, configuration: conf/predict.yaml
* python3 ml_prepareXY.py                                           Prepare X_train, y_train, X_val, y_val for ml based algorithms, configuration: conf/ml_prepareXY.yaml
* python3 ml_train.py                                               Train ml based model, configuration: conf/ml_train.yaml
Required for testing
* python3 slice_and_preprocess.py                                   Create slices, configuration: conf/slice.yaml
* python3 predict_slices.py                                         Generate predictions for each subregion, configuration: conf/predict_slices.yaml
* python3 eval.py                                                   Generate region based evaluation csv file, configuration: conf/eval.yaml
</pre>

## Data Structure
```
noaa
│
└─── images                           Location to store *.TIF files for training
└─── labels                           Location to store corresponding train shapefiles. The filenames for the tif file and its corresponding shapefile is same
└─── ml_data                          Location to store machine learning train, validation numpy arrays, trained model. Created during ml_prepareXY
└─── processed                        Location to store train, test, val directories, normalize array. Created during slice_and_preprocess
└─── runs                             Location to store U-Net training runs. Created during unet_train.py
└─── test_images                      For Denseley labeled test set
│   └───images                        Location to store *.TIF files for testing
│   └───labels                        Location to store corresponding test shapefiles. The filenames for the tif file and its corresponding shapefile is same
│   └───preds                         Location to store prediction from trained models. Created during predict_slices
│   └───processed                     Location to store subregions for testing.
```
