# Two-stage PHTrans

This repository is a solution for the [MICCAI FLARE2022 challenge](https://flare22.grand-challenge.org/). A detailed description of the method introduction, experiments and analysis of the results for this solution is presented in paper : [Combining Self-Training and Hybrid Architecture for Semi-supervised Abdominal Organ Segmentation](https://arxiv.org/abs/2207.11512). As shown in the figure below, this pipeline consists of two parts: (a) pseudo-label generation for unlabeled data, which is implemented using PHTrans under the nn-UNet framework (for more information, see [PHTrans](https://github.com/lseventeen/PHTrans)); (b) a two-stage segmentation framework with Lightweight PHTrans. This repository is the code implementation of this part.

<div align="center">
  <img src="img/method.png" width="80%">
</div>

 
## Prerequisites
 

 
Download our repo and install packages:
```
git clone https://github.com/lseventeen/FLARE22-TwoStagePHTrans
cd FLARE22-TwoStagePHTrans
pip install -r requirements.txt
```

 
## Datasets processing
Download [FLARE 2022](https://flare22.grand-challenge.org/Dataset/) datasets. Generate pseudo-labels for unlabeled data based on the repository [PHTrans](https://github.com/lseventeen/PHTrans). Modify the data path in the [config.py](https://github.com/lseventeen/FLARE22-TwoStagePHTrans/blob/master/config.py) file. Type this in the terminal to perform dataset processing:
 
```
python data_processing.py
```

## Training
Type this in terminal to run coarse segmentation train:
 
```
python coarse_train.py
```
Type this in terminal to run fine segmentation train:
 
```
python fine_train.py
```
## Inference
Type this in terminal to Inference:
 
```
python predict.py -dp DATA_PATH -op SAVE_RESULTS_PATH
```



