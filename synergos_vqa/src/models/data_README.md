# Dataset Preparation

This project requires the following four core datasets to fully reproduce the experiments described in the paper: **COCO**, **OK-VQA**, **A-OKVQA**, and **ScienceQA**. This document provides detailed instructions for downloading and organizing these datasets.

## Final Directory Structure

Please follow the steps below. After completion, your `data/` directory should have the following structure:
```
data/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2014.json
│   │   └── instances_val2014.json
│   ├── train2014/
│   │   └── *.jpg
│   └── val2014/
│       └── *.jpg
│
├── okvqa/
│   ├── OpenEnded_mscoco_train2014_questions.json
│   ├── mscoco_train2014_annotations.json
│   ├── OpenEnded_mscoco_val2014_questions.json
│   └── mscoco_val2014_annotations.json
│
├── aokvqa/
│   ├── aokvqa_v1p0_train.json
│   └── aokvqa_v1p0_val.json
│
└── scienceqa/
├── problems.json
├── pid_splits.json
└── images/
├── train/
└── val/
```

---

## 1. COCO (Common Objects in Context)

The COCO dataset provides the base images for both OK-VQA and A-OKVQA. It is also required for building the offline prototype library.

* **Purpose**: Provides images, used to build the prototype library
* **Version**: 2014

#### Download Links:

* [2014 Train images](http://images.cocodataset.org/zips/train2014.zip) (~13 GB)
* [2014 Val images](http://images.cocodataset.org/zips/val2014.zip) (~6 GB)
* [2014 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) (~241 MB)

#### Setup Instructions:

1.  Create a subdirectory named `coco` inside the `data/` directory.
2.  Unzip all three downloaded `zip` files into the `data/coco/` directory.
3.  After unzipping, ensure your directory structure matches the `coco/` section shown in the "Final Directory Structure" diagram above.

---

## 2. OK-VQA (Outside Knowledge VQA)

OK-VQA is one of the core benchmarks for evaluating knowledge-based visual question answering in this project.

* **Purpose**: Model training and evaluation
* **Note**: OK-VQA uses images from the COCO 2014 dataset. **You do not need to download the images again.**

#### Download Links:

You can download the following files from the [OK-VQA official website](https://okvqa.allenai.org/download.html):

* [Train: Questions](https://okvqa.s3.amazonaws.com/data/OpenEnded_mscoco_train2014_questions.json)
* [Train: Annotations](https://okvqa.s3.amazonaws.com/data/mscoco_train2014_annotations.json)
* [Val: Questions](https://okvqa.s3.amazonaws.com/data/OpenEnded_mscoco_val2014_questions.json)
* [Val: Annotations](https://okvqa.s3.amazonaws.com/data/mscoco_val2014_annotations.json)

#### Setup Instructions:

1.  Create a subdirectory named `okvqa` inside the `data/` directory.
2.  Place all four downloaded `.json` files into the `data/okvqa/` directory.

---

## 3. A-OKVQA (Argumentation-based OKVQA)

A-OKVQA is a more challenging benchmark for knowledge-based VQA.

* **Purpose**: Model evaluation
* **Note**: A-OKVQA also uses images from the COCO dataset. **You do not need to download the images again.**

#### Download Links:

You can download the `v1p0` data from the [A-OKVQA official website](https://aokvqa.allenai.org/download.html):

* [A-OKVQA v1p0 All Splits](https://aokvqa.s3.amazonaws.com/data/aokvqa_v1p0.zip) (~15 MB)

#### Setup Instructions:

1.  Create a subdirectory named `aokvqa` inside the `data/` directory.
2.  Unzip the downloaded `aokvqa_v1p0.zip` file.
3.  Place the resulting `aokvqa_v1p0_train.json` and `aokvqa_v1p0_val.json` files into the `data/aokvqa/` directory.

---

## 4. ScienceQA

ScienceQA is a multimodal science question answering benchmark. As described in our paper, we use the subset that includes image contexts.

* **Purpose**: Model evaluation
* **Note**: ScienceQA has its own set of images that must be downloaded separately.

#### Download Links:

You can find the download links at the [ScienceQA GitHub repository](https://github.com/lupantech/ScienceQA).

* **Data Files**: The GitHub page provides links for the `problems.json` and `pid_splits.json` files.
* **Image Files**: The link to download the `images` folder can also be found on the GitHub page.

#### Setup Instructions:

1.  Create a subdirectory named `scienceqa` inside the `data/` directory.
2.  Place the downloaded `problems.json` and `pid_splits.json` files into `data/scienceqa/`.
3.  Unzip the image files and place the `train` and `val` image subdirectories under
