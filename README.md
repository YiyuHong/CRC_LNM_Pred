# CRC_LNM_Pred
## Introduction
H&E whole slide images to predict metastasis of lymph node in T1 colorectal cancer using endoscopically resected specimens.

**Notice**: This repository is under construction, the full repository will be completed soon.
# Requirement
+ python >= 3.6
  + numpy >=1.17.4
  + openslide-python >= 1.1.2
  + pandas >= 1.1.3
  + scikit-image >= 0.15.0
  + scikit-learn >= 0.23.2
  + torch >= 1.5.1 (https://pytorch.org/)
  + torchvision >= 0.6.1
+ openslide >= 3.4.1 (https://openslide.org/)
# Usage
- python make_image_list_dict.py
  - prepare training and test data
- python train_patch_image.py
  - train patch-level image feature extractor
- python test_patch_image.py
  - test patch-level image LNM prediction
- python train_slide.py
  - train slide-level end-to-end LNM prediction model
- python test_slide.py
  - test slide-level LNM prediction
- python show_attention_map.py
  - show attention map of the predicted slide   

# Reference
"Utility of artificial intelligence with deep learning of hematoxylin and eosin-stained whole slide images to predict metastasis of lymph node in T1 colorectal cancer using endoscopically resected specimens" -> paper is under review







