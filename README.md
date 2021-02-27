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
<pre>
* sudo apt update                                                   Install python pip, setuptools
* sudo apt install python3-pip
* sudo python3 -m pip install -U pip
* sudo python3 -m pip install -U setuptools
* git clone https://github.com/Aryal007/coastal_mapping.git         Clone Repository
* cd coastal_mapping                                                Change directory to coastal mapping
* pip3 install -r requirements.txt                                  Install all the necessary requirements
* python3 slice_and_preprocess.py                                   Create slices, configuration specified in conf/slice.yaml
* python3 unet_train.py                                             Train model, configuration specified in conf/train.yaml
* python3 unet_predict.py                                           Generate masks for new image, configuration specified in conf/predict.yaml
</pre>
```
To install nvidia drivers on compatible Azure virtual machine, 
https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup
```
