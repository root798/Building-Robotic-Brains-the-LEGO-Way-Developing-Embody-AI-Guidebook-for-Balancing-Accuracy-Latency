# LEGO-AI · Multi-Domain Detection Training Guide

**Building Robotic Brains the LEGO Way** — practical, auditable recipes for
fine-tuning the perception models used in accuracy–latency research.

Maintained by [root798](https://github.com/root798).
This repository publishes training configurations, numerical safeguards, tests,
and dated experiment status. It is not a checkpoint download service or a
certified latency/energy benchmark.

## Start here

- [Training status](docs/training-status.md): the 20-job completion campaign,
  with the RepPoints repair tracked separately rather than double-counted.
- [Reproduction guide](docs/reproduction.md): environment, external assets,
  configuration layout, and a guarded single-run training command.
- [Published configurations](training/configs/): four domains, portable paths,
  and source/export hashes. Configurations preserve the audited recipes,
  including explicitly documented limitations.
- [Numerical stability](docs/numerical-stability.md): reproduce the FP16 focal
  loss failure, apply the FP32 repair, and reject nonfinite training losses.
- [Contributing](CONTRIBUTING.md): evidence requirements and safe publication.

## Four domains

| Domain | Perception setting | Guide |
| --- | --- | --- |
| BDD100K-FT | Driving scenes | [Historical model table](BDD20K_FT.md) |
| Mobility / CMA | Mobility perception | [Dataset and current recipes](Mobility.md) |
| SeaDronesSee | Maritime aerial detection | [Historical model table](SeaDronesSee.md) |
| WAID | Wildlife aerial detection | [Historical model table](WAID_FT.md) |

Models are fine-tuned for each domain's own categories. Four-domain coverage
does not imply one checkpoint transfers unchanged across all four datasets.
The 32 × 4 grid is a coverage target, not a claim that every cell is verified.

## What is available

| Artifact | Availability |
| --- | --- |
| 20 campaign configurations + 1 candidate replacement | Source in `training/configs/` |
| FP32 focal loss and per-batch finite-loss guard | Source and regression tests |
| Machine-readable training status | Dated snapshot in `training/status.json` |
| Model weights, images and COCO annotations | Not distributed in this update |
| Clean latency/energy measurements | Not part of this training release |

The existing dataset pages and performance CSVs are historical records. They
are **not** the evaluation results of the current completion campaign. No
unverified download links, invented metrics, or "all training finished" badge
are supplied. A completed process or `TRAIN_DONE` marker alone is insufficient
evidence of a healthy run; see the status definitions and acceptance checks.

## Check the publication without a GPU

From the repository root, using Python 3.8 or newer:

```bash
python -m unittest discover -s tests -p 'test_training_configs.py'
python -m unittest discover -s tests -p 'test_publication.py'
python -m unittest discover -s tests -p 'test_train_cli.py'
```

These standard-library checks validate the published files, not model quality.
The [reproduction guide](docs/reproduction.md) separates CPU hook tests from the
optional CUDA numerical regression and full training. Do not run new GPU tests
on a shared device while another experiment is active.

## Provenance and scope

Keep checkpoint identity, configuration, data split, evaluation protocol and
training health together when reporting accuracy. A numerical repair and a
preprocessing change are different recipe revisions; do not silently mix them.
Training throughput is not inference latency, and a free GPU alone does not
make a shared machine suitable for timing measurements.

Upstream frameworks and datasets retain their own terms and attribution; see
[third-party notices](THIRD_PARTY_NOTICES.md). No repository-wide license or
dataset redistribution permission is implied by this update.

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
