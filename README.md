# multimodal-class-acv
Multimodal classification using [IRENE](https://www.nature.com/articles/s41551-023-01045-x). Since the dataset used in the original paper was not publicly available, we decided to use the [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert) dataset instead.

### Environment Setup
`Python 3.10`, `Pytorch`, `Cuda 11.8`, `sklearn`, `PIL`, `matplotlib`, `skimage`

Manual install: `apex (from NVIDIA)`, `ml_collections`

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