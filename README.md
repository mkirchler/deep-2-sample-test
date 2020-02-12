# Source code for the paper "Two-sample Testing Using Deep Learning"

## Installation
Install requirements e.g. via
```
pip install -r requirements.txt
```

Install `deeptest` module via

```
pip install .
```

## Usage

Our proposed Tests are implemented in `embeddingtests.py` as `DFDATest` and `DMMDTest`.
For kernel-based tests we use wrappers around the implementation of [freqopttest](https://github.com/wittawatj/interpretable-test), found in `kerneltests.py`.
Transfer-C2STs are implemented in `c2st.py`.
Other modules (`base.py`, `data.py`, `load_tests.py` and `utils.py`) are utility functionalities.

Example usage and all experiments from the paper can be found under `experiments/experiments.py`.


## Data

To run the experiments from the paper you first need to download the data.

### Audio data
The data can be downloaded from `http://dl.lowtempmusic.com/Gramatik-TAOR.zip`. .mp3-files must be converted to 8kHz .wav files, using e.g. ffmpeg:
```
ffmpeg -i INFILE.mp3 -ar 8000 OUTFILE.wav
```
Pretrained weights are under `models/M5weights.pt`; a script for training can be found in `audio.py`.

### Aircraft Data

Data can be downloaded from `http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz` and needs to be extracted into separate directories according to classes.
You can use the script `aircraft.py` but might need to adapt to your system.
We use the pretrained resnet-152 provided in pytorch.

### Stanford Dogs Data

Data can be downloaded from `http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar`.
Pretrained weights are under `models/Supervised_dogs_weights.pt` and `models/CAEweights.pt`; a script for training can be found in `dogs.py`.

### Facial Expressions

The data can be downloaded after registering from `http://kdef.se/index.html`. Data needs to be sorted into `positive` and `negative` directories.
We use the pretrained resnet-152 provided in pytorch.

### Birds

Data can be downloaded from `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`.
We use the pretrained resnet-152 provided in pytorch.


### Your own data

To use the test on your own data, inherit from `TestData` (in `data.py`) and implement the methods `test_h0()` (returning true, if samples from H0 can be drawn) and `get_data(H0=True)` which returns two numpy-arrays X, Y drawn according to H0 or H1. Tests & data can then be used together via TestPipe(...).evalutate_test() (which returns t1er, t2er, and p-values).


