# scenariofintune

# BDD20K-FT – Fine-Tuned Detection Models for the BDD100K Dataset
A lightweight model zoo and training recipe collection that push the accuracy of classic and modern object-detectors on the **BDD100K** driving-scene benchmark by fine-tuning them on a curated, domain-specific split.


---

## 1. Dataset

| Split        | Images | Annotation source | Notes                                |
|--------------|-------:|-------------------|--------------------------------------|
| **Train**    | 14 k   | BDD100K DET       | Coner Case train set                    |
| **Val**      | 2 k   | BDD100K DET       | Used for all numbers in the table    |
| **Test**     | 2 k   | BDD100K DET     | Used for all numbers in the table         |

All classes follow the original 11-category BDD label set.  
Sample frames are dominated by **urban U.S. traffic**, but we keep the long-tail corner-cases (rain, night, snow) to ensure robustness.

---

## 2. Model Zoo (↑ = best)

| Model                              | bbox_mAP | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
|:-----------------------------------|:--------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|
| bdd20k_ft_ddq_detr4scale_r50       | 0.351 | 0.605 | 0.332 | 0.160 | 0.398 | 0.577 |
| bdd20k_ft_dino_swin_l              | 0.342 | 0.606 | 0.326 | 0.158 | 0.388 | 0.549 |
| bdd20k_ft_frcnn_r50_fpn            | 0.320 | 0.576 | 0.307 | 0.144 | 0.383 | 0.512 |
| bdd20k_ft_atss_r50                 | 0.313 | 0.550 | 0.299 | 0.140 | 0.362 | 0.540 |
| bdd20k_ft_tood_r50        | 0.312 | 0.556 | 0.296 | 0.135 | 0.367 | 0.547 |
| bdd20k_ft_diffusiondet_r50         | 0.304 | 0.561 | 0.275 | 0.146 | 0.351 | 0.496 |
| bdd20k_ft_varifocalnet_r50 | 0.305 | 0.548 | 0.289 | 0.128 | 0.360 | 0.538 |
| bdd20k_ft_nas_fcos_r50      | 0.301 | 0.544 | 0.285 | 0.130 | 0.358 | 0.538 |
| bdd20k_ft_retinanet_r50            | 0.296 | 0.525 | 0.284 | 0.118 | 0.353 | 0.533 |
| bdd20k_ft_conditional_detr_r50 | 0.297 | 0.538 | 0.279 | 0.120 | 0.350 | 0.529 |
| bdd20k_ft_fcos_r50                 | 0.294 | 0.529 | 0.274 | 0.123 | 0.336 | 0.509 |
| bdd20k_ft_grid_rcnn_r50     | 0.291 | 0.535 | 0.275 | 0.118 | 0.345 | 0.525 |
| bdd20k_ft_deformable_detr_r50      | 0.290 | 0.528 | 0.268 | 0.109 | 0.335 | 0.483 |
| bdd20k_ft_cascade_rcnn_r50         | 0.289 | 0.515 | 0.271 | 0.120 | 0.345 | 0.480 |
| bdd20k_ft_tridentnet_r50    | 0.297 | 0.540 | 0.280 | 0.122 | 0.352 | 0.530 |
| bdd20k_ft_paa_r50           | 0.299 | 0.540 | 0.282 | 0.125 | 0.353 | 0.532 |
| bdd20k_ft_detr_r50         | 0.289 | 0.530 | 0.272 | 0.115 | 0.340 | 0.520 |
| bdd20k_ft_yolof_r50                | 0.277 | 0.480 | 0.278 | 0.065 | 0.351 | 0.527 |
| bdd20k_ft_yolox_s          | 0.278 | 0.490 | 0.266 | 0.070 | 0.330 | 0.525 |
| bdd20k_ft_rtmdet_tiny              | 0.265 | 0.466 | 0.249 | 0.083 | 0.318 | 0.524 |
| bdd20k_ft_dynamic_rcnn_r50         | 0.264 | 0.466 | 0.251 | 0.098 | 0.320 | 0.432 |
| bdd20k_ft_retinanet_pvtt           | 0.253 | 0.465 | 0.232 | 0.086 | 0.309 | 0.439 |
| bdd20k_ft_ssd300_vgg16      | 0.233 | 0.420 | 0.215 | 0.040 | 0.280 | 0.410 |
| bdd20k_ft_retinanet_effb3          | 0.246 | 0.441 | 0.229 | 0.045 | 0.304 | 0.458 |
| bdd20k_ft_freeanchor_r50           | 0.234 | 0.432 | 0.212 | 0.083 | 0.273 | 0.393 |
| bdd20k_ft_mask_rcnn_swin_t         | 0.231 | 0.462 | 0.191 | 0.103 | 0.282 | 0.357 |
| bdd20k_ft_efficientdet_d3          | 0.213 | 0.388 | 0.195 | 0.042 | 0.276 | 0.405 |
| bdd20k_ft_reppoints_r50            | 0.208 | 0.397 | 0.185 | 0.074 | 0.250 | 0.353 |
| bdd20k_ft_sparse_rcnn_r50    | 0.155 | 0.300 | 0.135 | 0.050 | 0.170 | 0.240 |
| bdd20k_ft_yolov3_320         | 0.148 | 0.175 | 0.105 | 0.025 | 0.155 | 0.165 |
| bdd20k_ft_centernet_r18_dcn  | 0.095 | 0.140 | 0.070 | 0.010 | 0.090 | 0.110 |



*Numbers are on **BDD20K val**. Each checkpoint is COCO-pre-trained, then fine-tuned for with augmentation detailed in each config.*
| model                             | dens\_bin\:dense | dens\_bin\:medium | dens\_bin\:sparse | infra\_bin\:few | infra\_bin\:none | infra\_bin\:rich | scene\:city street | scene\:highway | scene\:residential | timeofday\:dawn/dusk | timeofday\:daytime | timeofday\:night | vru\_bin\:few | vru\_bin\:many | vru\_bin\:none | weather\:clear | weather\:overcast | weather\:partly cloudy | weather\:rainy | weather\:snowy | average   |
| --------------------------------- | ---------------- | ----------------- | ----------------- | --------------- | ---------------- | ---------------- | ------------------ | -------------- | ------------------ | -------------------- | ------------------ | ---------------- | ------------- | -------------- | -------------- | -------------- | ----------------- | ---------------------- | -------------- | -------------- | --------- |
| bdd10k\_ft\_ddq\_detr4scale\_r50  | 0.364            | 0.362             | 0.412             | 0.361           | 0.497            | 0.344            | 0.314              | 0.350          | 0.356              | 0.357                | 0.480              | 0.299            | 0.346         | 0.365          | 0.301          | 0.344          | 0.400             | 0.353                  | 0.344          | 0.308          | **0.363** |
| bdd10k\_ft\_dino\_swin\_l         | 0.340            | 0.333             | 0.432             | 0.352           | 0.495            | 0.349            | 0.328              | 0.297          | 0.369              | 0.366                | 0.357              | 0.316            | 0.332         | 0.360          | 0.279          | 0.341          | 0.389             | 0.341                  | 0.345          | 0.312          | **0.352** |
| bdd20k\_ft\_tood\_r50             | 0.314            | 0.308             | 0.377             | 0.311           | 0.433            | 0.323            | 0.281              | 0.301          | 0.321              | 0.318                | 0.353              | 0.290            | 0.309         | 0.323          | 0.257          | 0.289          | 0.353             | 0.322                  | 0.334          | 0.285          | **0.320** |
| bdd10k\_ft\_atss\_r50             | 0.312            | 0.288             | 0.365             | 0.310           | 0.452            | 0.317            | 0.284              | 0.278          | 0.305              | 0.329                | 0.351              | 0.296            | 0.298         | 0.318          | 0.253          | 0.322          | 0.353             | 0.321                  | 0.326          | 0.294          | **0.319** |
| bdd10k\_ft\_varifocalnet\_r50     | 0.306            | 0.274             | 0.380             | 0.291           | 0.450            | 0.301            | 0.281              | 0.325          | 0.301              | 0.318                | 0.360              | 0.269            | 0.289         | 0.319          | 0.246          | 0.293          | 0.346             | 0.319                  | 0.307          | 0.267          | **0.312** |
| bdd10k\_ft\_diffusiondet\_r50     | 0.315            | 0.284             | 0.313             | 0.311           | 0.383            | 0.318            | 0.277              | 0.356          | 0.305              | 0.302                | 0.356              | 0.269            | 0.293         | 0.309          | 0.254          | 0.291          | 0.368             | 0.308                  | 0.323          | 0.256          | **0.310** |
| bdd10k\_ft\_frcnn\_r50\_fpn       | 0.308            | 0.275             | 0.332             | 0.288           | 0.414            | 0.313            | 0.265              | 0.333          | 0.288              | 0.297                | 0.334              | 0.284            | 0.284         | 0.302          | 0.242          | 0.314          | 0.341             | 0.299                  | 0.324          | 0.263          | **0.305** |
| bdd20k\_ft\_tridentnet\_r50       | 0.285            | 0.286             | 0.308             | 0.290           | 0.364            | 0.296            | 0.268              | 0.313          | 0.288              | 0.296                | 0.324              | 0.254            | 0.284         | 0.293          | 0.236          | 0.288          | 0.316             | 0.299                  | 0.319          | 0.250          | **0.293** |
| bdd10k\_ft\_retinanet\_r50        | 0.300            | 0.258             | 0.333             | 0.289           | 0.379            | 0.291            | 0.267              | 0.267          | 0.282              | 0.299                | 0.317              | 0.279            | 0.262         | 0.295          | 0.225          | 0.282          | 0.325             | 0.291                  | 0.316          | 0.274          | **0.292** |
| bdd10k\_ft\_cascade\_rcnn\_r50    | 0.302            | 0.276             | 0.264             | 0.273           | 0.382            | 0.309            | 0.248              | 0.293          | 0.271              | 0.292                | 0.304              | 0.251            | 0.266         | 0.304          | 0.233          | 0.283          | 0.346             | 0.283                  | 0.317          | 0.250          | **0.287** |
| bdd10k\_ft\_fcos\_r50             | 0.274            | 0.269             | 0.295             | 0.289           | 0.330            | 0.297            | 0.271              | 0.265          | 0.277              | 0.292                | 0.331              | 0.257            | 0.277         | 0.283          | 0.245          | 0.294          | 0.321             | 0.280                  | 0.296          | 0.252          | **0.285** |
| bdd10k\_ft\_rtmdet\_tiny          | 0.251            | 0.263             | 0.301             | 0.281           | 0.306            | 0.274            | 0.234              | 0.240          | 0.281              | 0.268                | 0.293              | 0.248            | 0.258         | 0.280          | 0.228          | 0.268          | 0.308             | 0.272                  | 0.286          | 0.225          | **0.268** |
| bdd10k\_ft\_dynamic\_rcnn\_r50    | 0.272            | 0.245             | 0.265             | 0.264           | 0.331            | 0.271            | 0.245              | 0.233          | 0.244              | 0.271                | 0.299              | 0.233            | 0.254         | 0.276          | 0.228          | 0.261          | 0.328             | 0.261                  | 0.285          | 0.234          | **0.265** |
| bdd10k\_ft\_deformable\_detr\_r50 | 0.249            | 0.251             | 0.293             | 0.268           | 0.453            | 0.246            | 0.241              | 0.205          | 0.251              | 0.248                | 0.286              | 0.208            | 0.249         | 0.260          | 0.199          | 0.252          | 0.289             | 0.267                  | 0.270          | 0.241          | **0.261** |
| bdd10k\_ft\_retinanet\_effb3      | 0.231            | 0.233             | 0.341             | 0.290           | 0.411            | 0.246            | 0.220              | 0.231          | 0.232              | 0.250                | 0.260              | 0.229            | 0.230         | 0.252          | 0.203          | 0.236          | 0.266             | 0.263                  | 0.280          | 0.217          | **0.256** |
| bdd10k\_ft\_retinanet\_pvtt       | 0.236            | 0.249             | 0.217             | 0.251           | 0.322            | 0.269            | 0.231              | 0.224          | 0.249              | 0.263                | 0.272              | 0.241            | 0.237         | 0.268          | 0.210          | 0.246          | 0.284             | 0.268                  | 0.279          | 0.243          | **0.253** |
| bdd10k\_ft\_freeanchor\_r50       | 0.241            | 0.232             | 0.269             | 0.250           | 0.269            | 0.244            | 0.216              | 0.236          | 0.227              | 0.259                | 0.259              | 0.212            | 0.227         | 0.245          | 0.206          | 0.231          | 0.275             | 0.255                  | 0.258          | 0.227          | **0.242** |
| bdd10k\_ft\_mask\_rcnn\_swin\_t   | 0.244            | 0.225             | 0.258             | 0.240           | 0.238            | 0.279            | 0.219              | 0.246          | 0.229              | 0.244                | 0.254              | 0.218            | 0.221         | 0.247          | 0.195          | 0.231          | 0.288             | 0.238                  | 0.237          | 0.201          | **0.238** |
| bdd10k\_ft\_yolof\_r50            | 0.183            | 0.195             | 0.285             | 0.242           | 0.318            | 0.187            | 0.179              | 0.193          | 0.195              | 0.198                | 0.196              | 0.189            | 0.195         | 0.199          | 0.179          | 0.197          | 0.243             | 0.222                  | 0.230          | 0.173          | **0.210** |
| bdd10k\_ft\_sparse\_rcnn\_r50     | 0.099            | 0.090             | 0.210             | 0.119           | 0.273            | 0.092            | 0.085              | 0.097          | 0.112              | 0.116                | 0.145              | 0.085            | 0.096         | 0.094          | 0.064          | 0.095          | 0.120             | 0.123                  | 0.135          | 0.106          | **0.118** |
| bdd10k_ft_yolov3_320     | 0.097 | 0.103 | 0.087 | 0.120 | 0.118 | 0.107 | 0.100 | 0.084 | 0.081 | 0.094 | 0.099 | 0.089 | 0.102 | 0.104 | 0.092 | 0.115 | 0.102 | 0.094 | 0.112 | 0.103 | **0.100** |
| bdd10k_ft_centernet_r18_dcn| 0.087 | 0.093 | 0.078 | 0.108 | 0.106 | 0.096 | 0.090 | 0.076 | 0.073 | 0.085 | 0.089 | 0.080 | 0.092 | 0.094 | 0.083 | 0.104 | 0.092 | 0.085 | 0.101 | 0.093 | **0.090** |



# WAID dataset
Below are fintuned model performance on WAID dataset:

## WAID Models Summary (↑ = best)

| Model | bbox_mAP | bbox_mAP_50 | bbox_mAP_75 | bbox_mAP_s | bbox_mAP_m | bbox_mAP_l |
|:------|:--------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|
| waid_ft_ddq_detr4scale_r50 | **0.616** | **0.972** | **0.697** | **0.474** | 0.630 | **0.714** |
| waid_ft_tood_r50 | 0.610 | 0.964 | 0.685 | 0.457 | **0.631** | 0.680 |
| waid_ft_varifocalnet_r50 | 0.602 | 0.959 | 0.669 | 0.443 | 0.627 | 0.676 |
| waid_ft_atss_r50 | 0.601 | 0.963 | 0.671 | 0.450 | 0.626 | 0.678 |
| waid_ft_diffusiondet_r50_8xb2-50e_coco | 0.601 | 0.962 | 0.675 | 0.460 | 0.624 | 0.694 |
| waid_ft_paa_r50 | 0.598 | 0.958 | 0.667 | 0.434 | 0.624 | 0.679 |
| waid_ft_grid_rcnn_r50 | 0.596 | 0.950 | 0.676 | 0.452 | 0.616 | 0.698 |
| waid_ft_dino_swin_l | 0.593 | 0.964 | 0.659 | 0.461 | 0.609 | 0.679 |
| waid_cascade_rcnn_r50 | 0.590 | 0.949 | 0.664 | 0.441 | 0.609 | 0.683 |
| waid_ft_retinanet_r50 | 0.586 | 0.946 | 0.650 | 0.423 | 0.620 | 0.685 |
| waid_ft_nas_fcos_r50 | 0.586 | 0.957 | 0.646 | 0.441 | 0.610 | 0.682 |
| waid_ft_frcnn_r50_fpn | 0.584 | 0.956 | 0.648 | 0.437 | 0.606 | 0.658 |
| waid_ft_rtmdet_tiny | 0.583 | 0.938 | 0.645 | 0.403 | 0.611 | 0.680 |
| waid_ft_deformable_detr_r50 | 0.581 | 0.966 | 0.628 | 0.439 | 0.601 | 0.670 |
| waid_ft_tridentnet_r50 | 0.577 | 0.948 | 0.632 | 0.412 | 0.611 | 0.669 |
| waid_ft_faster_rcnn | 0.574 | 0.951 | 0.628 | 0.423 | 0.600 | 0.635 |
| waid_ft_fcos_r50 | 0.574 | 0.952 | 0.627 | 0.426 | 0.600 | 0.685 |
| waid_ft_retinanet_effb3 | 0.565 | 0.928 | 0.622 | 0.368 | 0.605 | 0.672 |
| waid_ft_retinanet_pvtt | 0.556 | 0.940 | 0.596 | 0.399 | 0.592 | 0.644 |
| waid_ft_mask_rcnn_swin_t | 0.547 | 0.940 | 0.588 | 0.407 | 0.576 | 0.606 |
| waid_ft_yolof_r50 | 0.537 | 0.905 | 0.577 | 0.334 | 0.611 | 0.679 |
| waid_ft_conditional_detr_r50 | 0.494 | 0.894 | 0.490 | 0.304 | 0.560 | 0.647 |
| waid_ft_detr_r50 | 0.455 | 0.850 | 0.440 | 0.239 | 0.531 | 0.620 |
| waid_ft_dynamic_rcnn_r50 | 0.402 | 0.694 | 0.425 | 0.275 | 0.447 | 0.355 |
| waid_sparse_rcnn_r50 | 0.257 | 0.497 | 0.245 | 0.125 | 0.310 | 0.276 |
| waid_ft_yolov3_d53_320 | 0.244 | 0.603 | 0.142 | 0.107 | 0.297 | 0.401 |
| waid_ft_efficientdet_d3 | 0.173 | 0.303 | 0.184 | 0.094 | 0.218 | 0.196 |
| waid_ft_centernet_r18_dcn | 0.111 | 0.284 | 0.066 | 0.044 | 0.147 | 0.154 |
| waid_ft_reppoints_r50 | 0.014 | 0.041 | 0.006 | 0.006 | 0.026 | 0.000 |


**Note**: Bold values indicate the best performance in each metric category among WAID models.



## 3. Quick Start

```bash
# Conda env (CUDA 11.x, PyTorch 1.13 +)
conda create -n bdd10k-ft python=3.10 -y
conda activate bdd10k-ft
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch -y

# MMDetection + COCO API
pip install mmcv-full mmdet
pip install 'git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools'

# Clone & link dataset
git clone https://github.com/<your-org>/bdd10k-ft.git
ln -s /path/to/bdd100k <your-repo>/data/bdd100k
````

### Inference (single GPU)

```bash
python tools/test.py \
    configs/bdd10k_ft_ddq_detr4scale_r50.py \
    checkpoints/bdd10k_ft_ddq_detr4scale_r50.pth \
    --show-dir vis/
```

### Re-training / Fine-tuning

```bash
./tools/dist_train.sh \
    configs/bdd10k_ft_frcnn_r50_fpn.py 8
```

### Validation-set evaluation

```bash
python -m bdd100k.eval.run -t det \
    -g data/bdd100k/labels/det_20/det_val.json \
    -r work_dirs/<exp>/bbox.json
```

---

## 4. Visualization

The repo includes a minimal Scalabel wrapper:

```python
from scalabel.vis.label import LabelViewer
viewer = LabelViewer()
viewer.draw(image, frame)      # draw GT or prediction frame
viewer.save("demo_vis.jpg")
```

Rendered examples live under `assets/vis/`.

---

## 5. Checkpoints & Configs

| Model                | Config                                    | Weight                                                                                  | Log                                   |
| -------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------- |
| DDQ DeTR-4scale R-50 | `configs/bdd10k_ft_ddq_detr4scale_r50.py` |  |  |
| …                    | …                                         | …                                                                                       | …                                     |

*(All links are direct-download, MD5-verified.)*

---

## 6. Citation

```bibtex

```


