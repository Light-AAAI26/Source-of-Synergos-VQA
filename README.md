# 🌲 Synergos-VQA 🌳

This is the official PyTorch implementation of the paper **"See the Forest and the Trees: A Synergistic Reasoning Framework for Knowledge-Based Visual Question Answering"**. This repository contains the complete end-to-end code to reproduce our results, including the online generation of all evidence streams described in our work, this project will be updated more user-friendly persistently.
<img alt="Method" src="https://github.com/user-attachments/assets/ae0c8a30-1461-4048-97bc-f9d4873b97c2" />


## 📋 Table of Contents

- [✨ Features](#-features)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 End-to-End Reproduction Workflow](#-end-to-end-reproduction-workflow)
  - [Step 1: Download Datasets](#step-1-download-datasets)
  - [Step 2: Build the Prototype Library (Offline)](#step-2-build-the-prototype-library-offline)
  - [Step 3: Train the Full Model](#step-3-train-the-full-model)
  - [Step 4: Evaluate the Model](#step-4-evaluate-the-model)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#️-contributing)

## ✨ Features

- 💡 Full end-to-end implementation of the **Synergos-VQA** framework.
- 🤖 Includes modules for online **Holistic, Structural, and Causal** evidence generation.
- 🔄 Implements a **Synergistic Decision Module** using a Fusion-in-Decoder (FiD) / T5 architecture to fuse all generated evidence.
- 🏆 Provides all necessary scripts to train our model from scratch and reproduce the SOTA results on **OK-VQA**.

## ⚙️ Installation & Setup (Detailed Guide)
This guide provides step-by-step instructions to create a suitable environment for running this project. We offer two recommended methods for environment management: venv (standard Python) and conda (popular in the scientific community).

Step 1: Clone the Repository
First, clone this repository to your local machine:

Step 2: Create and Activate the Environment
Choose one of the following options.

Option A: Using venv (Recommended for Simplicity)

This method uses Python's built-in environment manager.
```Bash
# Create a virtual environment using Python 3.10
python3.10 -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```
You will see (venv) at the beginning of your terminal prompt, indicating the environment is active.
Option B: Using conda
If you have Anaconda or Miniconda installed, you can create a conda environment.
```Bash
# Create a new conda environment named "synergos_vqa" with Python 3.10
conda create -n synergos_vqa python=3.10 -y

# Activate the environment
conda activate synergos_vqa
```
You will see (synergos_vqa) at the beginning of your terminal prompt.

Step 3: Install PyTorch with CUDA Support
This is a critical step. The version of PyTorch must match your system's CUDA driver. Our project was tested with CUDA 12.1 and PyTorch 2.1.0.
Important: We strongly recommend visiting the official PyTorch website to get the precise installation command for your specific system configuration.
For our verified environment (CUDA 12.1), the command is:
```Bash
# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Step 4: Install Other Dependencies
Once PyTorch is installed correctly, install all other required packages from the requirements.txt file. Make sure the requirements.txt file is in your root directory.
```Bash
pip install -r requirements.txt
```

Step 5: Verify the Installation (Optional but Recommended)
To ensure everything is set up correctly, you can run the following Python commands:
```Bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```
You should see an output similar to this, confirming that PyTorch can detect your GPU:

PyTorch version: 2.1.0
CUDA available: True
Device name: NVIDIA A100-SXM4-80GB
Environment Summary

## 🚀 End-to-End Reproduction Workflow
Step 1: Download Datasets
Please download the required raw datasets according to our data_README.md (e.g., OK-VQA images and annotations, COCO images) and place them in a directory, for example, ./data/. Update the paths in the configuration files in the configs/ directory accordingly.

Step 2: Build the Prototype Library (Offline)
This step uses the COCO dataset to build the prototype library required for the Structural Backbone Reasoning module.

```
# This script (e.g., library.py or a new pipelines/build_prototype_library.py) 
# should be run to generate the prototype_library.pkl
python src/library.py --coco_path ./data/coco --output_path ./checkpoint/prototype_library.pkl
```
Step 3: Train the Full Model
This command launches the end-to-end training process, which performs online evidence generation for each batch before training the decision module.
```
# Ensure your train.sh script is configured with the correct paths and parameters
bash train.sh
```
Step 4: Evaluate the Model
To evaluate your trained model or our provided checkpoint, run the evaluation script. This will also perform online evidence generation for each test sample.
```
#Ensure your test.sh script is configured with the correct checkpoint path
bash test.sh
```
## 📂 Project Structure
```
synergos_vqa/
├── checkpoints/              # Directory for model weights and generated libraries
├── configs/                  # Directory for configuration files
├── data/                     # Directory for datasets (created by user)
├── src/                      # Source code
│   ├── models/               # Model definitions (FiDT5, etc.)
│   ├── data.py               # Data loading and processing logic
│   ├── evidence_generator.py # Online evidence generation module
│   ├── library.py            # Logic for prototype library (generation/usage)
│   └── util.py               # Utility functions
├── .gitignore                # Specifies files to ignore for Git
├── LICENSE                   # Your open-source license (e.g., MIT)
├── README.md                 # This file
├── requirements.txt          # Pip dependencies
├── test.py                   # Main script for evaluation
├── test.sh                   # Bash script for launching evaluation
├── train.py                  # Main script for training
└── train.sh                  # Bash script for launching training
```
## 🤝 Contributing
We welcome contributions to enhance this project. If you're interested in contributing, please follow these steps:

🍴 Fork the repository

🌿 Create a new branch for your feature or bug fix (git checkout -b feature/AmazingFeature)

💬 Commit your changes with clear and concise messages (git commit -m 'Add some AmazingFeature')

📤 Push your branch to your forked repository (git push origin feature/AmazingFeature)

📥 Open a Pull Request detailing your changes and the motivation behind them.

Please ensure that your code adheres to the existing style and includes appropriate tests.
