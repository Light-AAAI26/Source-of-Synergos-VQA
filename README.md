# Source-of-Synergos-VQA
This repository contains the reletive code for our AAAI-26 submission.
# 🌲 Synergos-VQA 🌳

This is the official PyTorch implementation of the paper **"See the Forest and the Trees: A Synergistic Reasoning Framework for Knowledge-Based Visual Question Answering"**. **Synergos-VQA** addresses the uni-dimensional evidence bottleneck in Knowledge-Based Visual Question Answering (KBVQA). We propose a synergistic reasoning framework that achieves a deeper and more robust visual understanding by concurrently generating and fusing three complementary evidence streams at inference time: **Holistic Evidence**, **Structural Evidence**, and **Causal Evidence**.
<img width="5063" height="2527" alt="Fig2" src="https://github.com/user-attachments/assets/8f96d0d6-fcaa-45f3-ae29-4bc035d7e37a" />



## 📋 Table of Contents

- [✨ Features](#-features)
- [⚠️ Hardware Requirements](#️-hardware-requirements)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 End-to-End Reproduction Workflow](#-end-to-end-reproduction-workflow)
  - [Step 1: Download Datasets](#step-1-download-datasets)
  - [Step 2: Build the Prototype Library (Offline)](#step-2-build-the-prototype-library-offline)
  - [Step 3: Train the Full Model](#step-3-train-the-full-model)
  - [Step 4: Evaluate the Model](#step-4-evaluate-the-model)
- [📓 Demo Notebook](#-demo-notebook)
- [🤝 Contributing](#-contributing)


## ✨ Features

- 💡 Full implementation of the **Synergos-VQA** framework, featuring its three core evidence-generation tracks.
- 🤖 Plug-and-play modules for **Holistic Scene Analysis**, **Structural Backbone Reasoning**, and a **Causal Robustness Probe**.
- 🔄 An efficient multi-source evidence fusion module based on **Fusion-in-Decoder (FiD) / T5**.
- 🏆 Complete scripts to reproduce the state-of-the-art (SOTA) results on benchmarks like **OK-VQA**, **A-OKVQA**, and **ScienceQA**.
- 🔌 Demonstrates the framework's capability as a model-agnostic toolkit to enhance the reasoning of various MLLMs.

## ⚠️ Hardware Requirements

> **Important:** Training and evaluating the full Synergos-VQA pipeline is computationally intensive due to the online evidence generation by large multimodal models.
> - **Recommended GPU:** NVIDIA A100 (80GB) or a comparable GPU with at least 48GB of VRAM.
> - **Recommended RAM:** At least 128GB of system RAM.
>
> *48GB is enough to train and evaluate the full Synergos-VQA, if your GPU has a VRAM lower than 40GB, please use the 3B MLLMs as the external engine*

## 📂 Project Structure
```bash
📂 Synergos-VQA/
├── configs/              # .yaml configuration files for experiments
├── data/                 # Placeholder for datasets
├── notebooks/            # Jupyter Notebooks for analysis and demos (demo.ipynb)
├── pipelines/            # Offline pipeline scripts (e.g., build_prototype_library.py)
├── scripts/              # Bash scripts for running full experiments
├── src/                  # Core source code
│   ├── evidence_generator/ # Implementation of the three evidence generators
│   ├── fusion_module/      # Implementation of the FiD/T5 fusion module
│   ├── data_loader.py    # Data loaders
│   └── utils.py          # Utility functions
├── checkpoints/          # Directory for saving/loading model weights and prototype library
├── train.py              # Main script for end-to-end training
├── evaluate.py           # Main script for end-to-end evaluation
├── requirements.yaml      # Project dependencies
└── README.md             # This file
```
## ⚙️ Installation

1.  **Clone the repository:**

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *Please ensure you have a compatible version of PyTorch and CUDA installed for your system.*
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 End-to-End Reproduction Workflow

Follow these steps to reproduce the results from scratch.

### Step 1: Download Datasets

Please download the required datasets and place them in the `./data` directory. You will need:
- **OK-VQA**
- **A-OKVQA**
- **ScienceQA**
- **COCO (for Prototype Library)**

Your `./data` directory should look something like this:
data/
├── okvqa/
│   ├── train2014/
│   ├── val2014/
│   └── ...
└── coco/
├── train2017/
├── val2017/
└── ...


### Step 2: Build the Prototype Library (Offline)

This step uses the COCO dataset to build the prototype library required for the Structural Backbone Reasoning module.

```bash
python3 pipelines/build_prototype_library.py \
    --corpus_path ./data/coco \
    --output_path ./checkpoints/prototype_library.pkl
```
This will generate the prototype_library.pkl file inside the checkpoints directory.

Step 3: Train the Full Model
This command launches the end-to-end training process. For each batch, it will perform online evidence generation before feeding the data to the decision module for training.

```Bash

# Example for training on OK-VQA
python3 train.py --config configs/synergos_okvqa_train.yaml
Training logs and model checkpoints will be saved to the directory specified in your config file.
```
Step 4: Evaluate the Model
4a. Evaluate Your Own Trained Model
Once your training is complete, use the following command to evaluate your model's performance on the test set.

```Bash

python3 evaluate.py \
    --config configs/synergos_okvqa_eval.yaml \
    --checkpoint /path/to/your/trained_model.pth
```

Download the used pre-trained model:

Link: [在此处放置您的 Google Drive, Hugging Face, 或其他可访问的下载链接]

Run evaluation:

```Bash

python3 evaluate.py \
    --config configs/synergos_okvqa_eval.yaml \
    --checkpoint ./checkpoints/synergos_vqa_sota.pth
This should reproduce the SOTA accuracy reported in our paper's main results table.
```
## 📓 Demo Notebook
For an intuitive demonstration of the Synergos-VQA reasoning process, please see notebooks/demo.ipynb. This notebook loads our pre-trained models and provides a step-by-step visualization of how the Holistic, Structural, and Causal evidence streams are generated and fused to produce the final answer for a given image-question pair.


## 🤝 Contributing
We welcome contributions to enhance this project. If you're interested in contributing, please follow these steps:

🍴 Fork the repository

🌿 Create a new branch for your feature or bug fix (git checkout -b feature/AmazingFeature)

💬 Commit your changes with clear and concise messages (git commit -m 'Add some AmazingFeature')

📤 Push your branch to your forked repository (git push origin feature/AmazingFeature)

📥 Open a Pull Request detailing your changes and the motivation behind them.

Please ensure that your code adheres to the existing style and includes appropriate tests.
