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
│   │   slice.yaml
│   │   train.yaml
│   
│   README.md
│   requirements.txt
│   slice_and_preprocess.py
│   unet_train.py
```

## Steps to run on a new machine

* sudo apt update                                                   __Install python pip, setuptools__
* sudo apt install python3-pip
* sudo python3 -m pip install -U pip
* sudo python3 -m pip install -U setuptools
* git clone https://github.com/Aryal007/coastal_mapping.git         __Clone Repository__
* cd coastal_mapping                                                __Change directory to coastal mapping__
* pip3 install -r requirements.txt                                  __Install all the necessary requirements__
* python3 slice_and_preprocess.py                                   __Create slices, configuration specified in conf/slice.yaml__
* python3 unet_train.py                                             __Train model, configuration specified in conf/train.yaml__
* python3 unet_predict.py                                           __Generate masks for new image, configuration specified in conf/predict.yaml__

```
To install nvidia drivers on compatible Azure virtual machine, 
https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
```