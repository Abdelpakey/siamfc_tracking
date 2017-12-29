# SiamFC - TensorFlow
TensorFlow port of the tracking method described in the paper [*Fully-Convolutional Siamese nets for object tracking*](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).

This repo adds multiple object tracking, tracking at different scales, optimized inference time, easy API as well as configurable options.

**Note1**: at the moment this code only allows to use a pretrained net in forward mode. 

## Requirements
See requirements.txt

## Running the tracker
Usage: `python run_tracker_demo.py -cfg [config_file] -f [folder of video images]`

Example: `python run_tracker_demo.py -cfg ./config/design_ori.json -f ./data/interoll2`

Make sure 'net_path' in the config file points to the path of your pretrained net.

## TODO 
Add training code

## Paper
This repo is a full reimplementation of Fully-Convolutional Siamese Networks for Object Tracking (see paper https://www.robots.ox.ac.uk/~luca/siamese-fc.html)

Based off the original repository (https://github.com/torrvision/siamfc-tf)
