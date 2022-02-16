# Facial Action Unit Detection with Transformers
 
This repository contains Tensorflow training code for the CVPR 2021 paper:
* Facial Action Unit Detection with Transformers
 
 
For details see [Facial Action Unit Detection with Transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Jacob_Facial_Action_Unit_Detection_With_Transformers_CVPR_2021_paper.pdf) by Geethu Miriam Jacob and
If you use this code for a paper please cite:
 
```
@inproceedings{jacob2021facial,
  title={Facial Action Unit Detection With Transformers},
  author={Jacob, Geethu Miriam and Stenger, Bjorn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7680--7689},
  year={2021}
}
```
 
# Usage
 
First, clone the repository locally:
```
git clone https://github.com/RIT/FAU_transformers.git
```
Then, install the packages in requirements file:
 
```
pip install -r requirements.txt
```
 
## Data preparation
 
 ```
python Prepare_data.py
```

## Train models
 
```
python main.py
```
 
## Evaluate models
 
# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
 
# Contributing
We welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
