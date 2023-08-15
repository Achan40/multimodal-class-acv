# multimodal-class-acv
Multimodal classification using [IRENE](https://www.nature.com/articles/s41551-023-01045-x). Since the dataset used in the original paper was not publicly available, we decided to use the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert) dataset instead.

### Environment Setup
Python version: 3.10

`Pytorch`, `Cuda 11.8`, `sklearn`, `PIL`, `matplotlib`, `skimage`, `ml-collections`, `tqdm`, `pickle`, `pandas`, `numpy`

### Training loop
Assuming all necessary packages have been installed into your python environment.
1. See the README within the `data` directory for data setup instructions
2. Modify and run the shell commands. Examples can be found in the `run.sh` file

See the `metrics_and_performance.ipynb` file for specifics on how we trained our model.

Training was conducted locally on a single machine with the following specifications. You may have to adjust certain parameters within `run_model.py` to work with your machine:

* OS: Windows 11
* CPU: Ryzen 9 7900X
* RAM: 64GB
* GPU: RTX 3080 12gb

Note: `run_model_cnn.py` is functionally the same as `run_model.py`, except this is used to train/test a custom CNN approach instead of IRENE. Notable differences include: Inherit from our `Data` class and create child `DataImageOnly` which overwrites the `__getitem__` method, removal of some arguments in the optimizer within the `train()` function, adjustments to the `item_preds` function.

### File Explanations

* `run.sh`: Commands for model training
* `run_model`: Main driver code for training and testing of IRENE
* `run_model_cnn`: Main driver code for training and testing of custom CNN approach
* `helper.py`: Various helper functions and classes
* scripts: Scripts used to generate datasets
* models: CNN and IRENE architecture
* data: Location for storing datasets
* notebooks: Various jupyter notebooks

## Code Citation
Code is adapted from [RL4M/IRENE](https://github.com/RL4M/IRENE).

```bash
@article{zhou2023irene,
  title={A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics},
  author={Zhou, Hong-Yu and Yu, Yizhou and Wang, Chengdi and Zhang, Shu and Gao, Yuanxu and Pan, Jia and Shao, Jun and Lu, Guangming and Zhang, Kang and Li, Weimin},
  journal={Nature Biomedical Engineering},
  doi={10.1038/s41551-023-01045-x}
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
