# Datasets

Counts describe the project’s COCO-format splits before training-time filtering.
Model identifiers containing `10k` or `20k` are names, not split-size guarantees.

| Domain | Train images | Train boxes | Validation images | Validation boxes |
| --- | ---: | ---: | ---: | ---: |
| BDD100K-FT | 12,000 | 208,517 | 4,000 | 69,759 |
| Mobility / CMA | 8,456 | 21,850 | 784 | 1,580 |
| SeaDronesSee | 8,930 | 57,760 | 1,547 | 9,630 |
| WAID | 10,056 | 163,243 | 2,873 | 46,696 |

Class names and order are recorded in the [catalog](../model_zoo/catalog.json)
and each domain page. BDD uses the project’s 11-class conversion rather than
the standard detection benchmark category set. Mobility / CMA uses person,
wheelchair, rollator, crutch, and cane; a public dataset download is not provided
by this repository.

## References

The following dataset citations are retained from the original guidebook.
Obtain datasets from their providers and follow the applicable terms.

```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A diverse driving dataset for heterogeneous multitask learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2636--2645},
  year={2020}
}

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

@inproceedings{varga2022seadronessee,
  title={Seadronessee: A maritime benchmark for detecting humans in open water},
  author={Varga, Leon Amadeus and Kiefer, Benjamin and Messmer, Martin and Zell, Andreas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2260--2270},
  year={2022}
}
```
