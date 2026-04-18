# [AAAI 2026] BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2601.15123)
[![Paper](https://img.shields.io/badge/Paper-AAAI-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/37757)


This repository contains offiсial dataset and code implementation for the paper:  
[BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation](https://arxiv.org/abs/2601.15123)

<img width="790" height="564" alt="image" src="https://github.com/user-attachments/assets/0f8414f7-603c-4b4b-911c-30c726f23f36" />


# Setting Environment

## Install Dependencies:

```pip3 install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
conda install -y scikit-image
conda install -y -c anaconda cmake
pip install -e .
```

## Prepare Datasets & Models Checkpoints

#### This project builds upon on [RITM](https://github.com/saic-vul/ritm_interactive_segmentation) and [TETRIS](https://arxiv.org/abs/2402.06132),and so it uses the same dataset structure and evaluation scripts. Thus, you should configure the paths to the datasets in [config.yml](./config.yml). We measured out BREPS attack on datasets below.

---

| Dataset     | Description                            |              Download Link              |
| ----------- | -------------------------------------- | :-------------------------------------: |
| GrabCut     | 50 images with one object each (test)  |     [GrabCut.zip (11 MB)][GrabCut]      |
| Berkeley    | 96 images with 100 instances (test)    |     [Berkeley.zip (7 MB)][Berkeley]     |
| DAVIS       | 345 images with one object each (test) |       [DAVIS.zip (43 MB)][DAVIS]        |
| COCO_MVal   | 800 images with 800 instances (test)   |   [COCO_MVal.zip (127 MB)][COCO_MVal]   |
| TETRIS      | 2000 images with 2531 instances (test) |      [TETRIS.zip (6.3 GB)][TETRIS]      |
| PASCAL-VOC  | 1000 images with 1000 instances (sampled, test) |      [PASCAL.zip (112 MB)][PASCAL]      |
| ADE20K      | 1000 images with 1000 instances (sampled, test) |      [ADE20K.zip (125 MB)][ADE20K]      |
| ACDC        | 100 images with 100 instances (test)   |        [ACDC.zip (13 MB)][ACDC]         |
| BUID        | 503 images with 503 instances (benign and malignant, test)   |        [BUID.zip (24 MB)][BUID]         |
| MedScribble | 56 images with 56 instances (test)     | [MedScribble.zip (2.5 MB)][MedScribble] |

[GrabCut]: https://drive.google.com/file/d/1tU17eJaevYd5PQAwzEK6oG1N9eBJORC4/view?usp=sharing
[Berkeley]: https://drive.google.com/file/d/1yStolKW8AnS5rYm3VhgnID_XReNCXbEJ/view?usp=sharing
[DAVIS]: https://drive.google.com/file/d/1DhwMOqbwH4tcIrVfqtSczdofjdHHnMCJ/view?usp=sharing
[COCO_MVal]: https://drive.google.com/file/d/1oz1VT8YqykQFfx_Vtp_wtYFGz5mYBOnp/view?usp=sharing
[TETRIS]: https://drive.google.com/file/d/1iJgohY1XBSnY-kRUoaRZJlu0HZbyHcyK/view?usp=sharing
[PASCAL]: https://drive.google.com/file/d/1MAbjGogMbYq8mg_H-JqUmsHwfxfkUniY/view?usp=sharing
[ADE20K]: https://drive.google.com/file/d/1sc__FtGU3mogxeevvOhWl9ajuRaEdIyB/view?usp=sharing
[ACDC]: https://drive.google.com/file/d/158tEwGGEWHR6F6Gk7BPM910fKwwJIhT5/view?usp=sharing
[BUID]: https://drive.google.com/file/d/1W83P700gzEvdD1KLJeQw5j0NyE-WwllJ/view?usp=sharing
[MedScribble]: https://drive.google.com/file/d/1fpuP74HdMsob6BiPdvVH2lrb7rzQfY-H/view?usp=sharing

## Real-Users Study:

We collected 25000 annotations, 50 user bboxes for 500 images from 10 datasets (All attack datasets and also [ADE20K](https://ade20k.csail.mit.edu/) and [PascalVOC](https://zenodo.org/records/8312614)).
You can download full user study data from this [link](https://drive.google.com/file/d/1cl2AheaxxvAt1pHeSvCtySqVYpE6BOQB/view?usp=sharing).

To download checkpoints, please refer to the repositories of the relevant papers or download all checkpoints used in this work at once — [MODELS_CHECKPONTS.zip (18 GB)](https://mega.nz/file/stIkCQLT#P53Pfw9YKzDVMBgELmjCdhMZlBpic3tzXols2snavrI)

# Run Optimization

## Short Example

`python3 scripts/evaluate_boxes_model_sam.py NoBRS --checkpoint ../MODEL_CHECKPOINTS/SAM/sam_vit_b_01ec64.pth --deterministic  --save-ious --datasets=GrabCut  --n_opt_steps=50 --lr_mult=9 --iou-analysis --gpus="0" --thresh=0.5 --optim_min --modality=bbox --lambda_mult 0.1`

All flags the same as in original models except following additional flags:

```
--n_opt_steps — number of optimization steps for the bounding box
--optim_min — minimization optimization (maximization by default)
--lr_mult — learning rate multiplyer for optimization (can be set to 0 with n_opt_steps=1 for baseline clicking strategy)
--n_workers — number of parallel workers for evaluation (the maximum number you can fit depends on your GPU)
--deterministic — force determistic algroritms in PyTorch (however, some models may use non-deterministic ops)
--lambda_mult — regularization strength
```

## Full Benchmarking

Some models ([SAM](https://github.com/facebookresearch/segment-anything), [SAM-HQ](https://github.com/SysCV/sam-hq), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [SAM2.1](https://github.com/facebookresearch/sam2), [SAM-HQ 2](https://github.com/SysCV/sam-hq/blob/main/sam-hq2/README.md),[RobustSAM](https://github.com/robustsam/RobustSAM/blob/main/README.md), [MedSAM](https://github.com/bowang-lab/MedSAM)) should be installed using separate package and don't support backpropagation from the box due to torch.no_grad calls. Thus, you can manually remove such calls or [download already patched versions](https://drive.google.com/file/d/1DsBMiTJENqwlQSny-83oWvMBy4QJS59S/view?usp=sharing) and install it using:

```
cd segment-anything-custom-build; pip install -e .
cd sam-hq-custom-build; pip install -e .
cd MobileSAM-custom-build; pip install -e .
...
```

etc.

To benchmark all models after setting up an environment and downloading all checkpoints to `MODEL_CHECKPOINTS` folder and just run: `bash runbboxparallel.sh`, selecting the amount of models appropriate for your server. All hyperparameters are set following author implementations.

# Metrics Calculation

After benchmarking, each model creates entries in the folder `EXPS_PATH` from config.yml. Merge it with baseline experiments folder if you need IoU-Base@BBox scores.Use provided `Evaluate Models.ipynb` Jupyter Notebook to calculate all metrics — IoU-Min@BBox, IoU-Max@BBox, IoU-Base@BBox. One can download obtained results from benchmarking all models — [experiments.zip (570 MB)](https://drive.google.com/file/d/14efw-_o83HJOXt449-SbuURf7ZJJVioy/view?usp=sharing). Sample output:

```
--------------------------------
GrabCut
--------------------------------
mobile_sam
IOU  | Min 96.21 | Base 94.77 | Max 97.19 | Delta 0.99
--------------------------------
robustsam_checkpoint_b
IOU  | Min 45.73 | Base 84.62 | Max 90.07 | Delta 44.33
--------------------------------
```

To compute and visualise heatmaps, please checkout [HEATMAPS.md](heatmaps/HEATMAPS.md)

# Citation

If you find this work useful for your research, please cite the original paper:

```
@article{moskalenko2026breps,
 title={BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation},
 author={Moskalenko, Andrey and Kuznetsov, Danil and Dudko, Irina and Iasakova, Anastasiia and Boldyrev, Nikita and Shepelev, Denis and Spiridonov, Andrei and Kuznetsov, Andrey and Shakhuro, Vlad},
 journal={Proceedings of the AAAI Conference on Artificial Intelligence},
 volume={40},
 number={10},
 year={2026},
 month={Mar.},
 pages={8107-8115},
 DOI={10.1609/aaai.v40i10.37757}
}
```
