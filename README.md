# [Ghana Crop Disease Detection Challenge Solution](https://zindi.africa/competitions/ghana-crop-disease-detection-challenge)
We present our solution for the Ghana Crop Disease Detection Challenge Solution on Zindi, currently placed 7th Place

## Model Summary

The solution was an ensemble of two RT-DETR [(Realtime Detection Transformer)](https://docs.ultralytics.com/models/rtdetr/#how-does-rt-detr-support-adaptable-inference-speed-for-different-real-time-applications) models using ultralytics framework. One model is trained from a 20 fold split and the other from a 24 fold split. The models are ensembled using [Weighted boxes fusion (WBF)](https://learnopencv.com/weighted-boxes-fusion/)

## TRAINING DETAILS
dependencies
add how to use the scripts
lint the notebook to run straight on kaggle

## ONNX EXPORT
Test the stuff discusses here: https://docs.ultralytics.com/models/rtdetr/

## MODEL EXPLAINABILITY
Share model and some images

## Authors
1. [ngoym](https://github.com/ngoym)
2. [sitwalam](https://github.com/SitwalaM)
