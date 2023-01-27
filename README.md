# Glacier Mapping
## _Estimate clean ice and debris glaciers boundaries in the HKH_

## Project Structure
```
glacier_mapping
│
└───conf
│   │   eval.yaml
│   │   get_roc_curve.yaml
│   │   predict_slices.yaml
│   │   slice_and_preprocess.yaml
│   │   unet_predict.yaml
│   │   unet_train.yaml
|
└─── segmentation
│   └───data
│       │   data.py
│       │   slice.py
│   └───model
│       │   frame.py
│       │   functions.py
│       │   losses.py
│       │   metrics.py
│       │   unet.py  
│   
│   .gitignore
|   eval.py
|   get_roc_curve.py
│   README.md
│   requirements.txt
│   slice_and_preprocess.py
│   unet_predict.py
│   unet_train.py
│   win_requirements.txt
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
* git clone https://github.com/Aryal007/glacier_mapping.git        Clone Repository
* cd coastal_mapping                                                Change directory to coastal mapping
* pip3 install -r requirements.txt                                  Install all the necessary requirements
</pre>

### Install requirements on a Windows machine
<pre>
* Download Anaconda from https://www.anaconda.com/products/individual
* git clone https://github.com/Aryal007/glacier_mapping.git         Clone Repository
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
<pre>
Required for training
* python3 slice_and_preprocess.py                                   Create slices, configuration: conf/slice.yaml
* python3 unet_train.py                                             Train model, configuration: conf/train.yaml
* python3 unet_predict.py                                           Generate masks for new image, configuration: conf/predict.yaml
Required for testing
* python3 slice_and_preprocess.py                                   Create slices, configuration: conf/slice.yaml
* python3 eval.py                                                   Generate region based evaluation csv file, configuration: conf/eval.yaml
</pre>

## Structure for data directory
```
HKH
│
└─── Landsat                          Location to store *.tif files
└─── labels                           Location to store train and test shapefiles.
└─── processed                        Location to store train, test, val directories, normalize array and slices metadata. Created during slice_and_preprocess
    └───runs                          Location to store U-Net training runs. Created during unet_train.py
```

Some code in this repository is taken from one of our earlier project at: https://github.com/krisrs1128/glacier_mapping
