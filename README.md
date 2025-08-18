# scenariofintune

## Fine-Tuned Object Detection Models Across Multiple Scenarios

This repository provides fine-tuned models across diverse domains and challenging scenarios:

| Dataset | Domain | Focus | Link |
|---------|--------|-------|------|
| **BDD100K** | Autonomous Driving | Urban traffic, corner cases | [→ BDD20K_FT](./BDD20K_FT.md) |
| **WAID** | Wildlife Monitoring | Aerial small object detection | [→ WAID_FT](./WAID_FT.md) |
| **SeaDronesSee** | Search & Rescue | Maritime small object detection | [→ SeaDronesSee](./SeaDronesSee.md) |



## Checkpoints & Configs

| Model                | Config                                    | Weight                                                                                  | Log                                   |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------- |
| DDQ DeTR-4scale R-50 | `configs/bdd10k_ft_ddq_detr4scale_r50.py` |  |  |
| …                    | …                                         | …                                                                                       | …                                     |

*(All links are direct-download, MD5-verified.)*

---

## Citation
For datasets used in the fintuning and scenario generation, these are the corresponding resources:
```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A diverse driving dataset for heterogeneous multitask learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2636--2645},
  year={2020}
}
```
```bibtex
@article{mou2023waid,
  title={WAID: A Large-Scale Dataset for Wildlife Detection with Drones},
  author={Mou, Chao and Liu, Tengfei and Zhu, Chengcheng and Cui, Xiaohui},
  journal={Applied Sciences},
  volume={13},
  number={18},
  pages={10397},
  year={2023},
  publisher={MDPI},
  doi={10.3390/app131810397}
}
```
```bibtex
@inproceedings{varga2022seadronessee,
title={Seadronessee: A maritime benchmark for detecting humans in open water},
author={Varga, Leon Amadeus and Kiefer, Benjamin and Messmer, Martin and Zell, Andreas},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
pages={2260--2270},
year={2022} }
```
