# HeaortaNet
HeaortaNet is a pre-trained model for 3D segmentation of heart and aorta from non-contrast chest CT.
 
# Model Overview
Inspired by SegResNet [1], we modified the model structure with bottleneck residual block and adding attention gate. This model is called HeaortaNet.

## Data
The model was trained to segment the heart and aorta based on non-contrast chest CT. Based on our own split, the labeled dataset from Nation Taiwan University Hospital (NTUH) was partioned into 60 training data, 20 validation data and 20 testing data. We only provide 6 testing data for trial，as shown in config/seg_cardiac_datalist.json. 

For more detail, please contact Dr. Tzung-Dau Wang (tdwang@ntu.edu.tw) and Prof. Wei-Chung Wang (wwang@math.ntu.edu.tw).

## Training configuration
The provided training configuration required 48GB GPU memory.

Model Input Shape: 160 x 160 x 160

Training Script: train.sh

## Input and output formats
### Input:
 Single channel chest CT images with fixed spacing (1x1x1 mm).
### Output:
 Three channels for heart, ascending & descending aorta.

## Scores
This Dice score on the testing data achieved by this model is 0.94 (heart) and 0.9 (ascending & descending aorta). 

# Availability
In order to access this model, please apply for general availability access at
https://developer.nvidia.com/clara

This model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container.

# Disclaimer
The content of this model is only an example.  It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 

# References
[1] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." International MICCAI Brainlesion Workshop. Springer, Cham, 2018. https://arxiv.org/pdf/1810.11654.pdf. 
