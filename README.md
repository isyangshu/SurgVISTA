<!-- # SurgVISTA
## Large-scale Self-supervised Video Foundation Model
for Intelligent Surgery -->
![header](https://capsule-render.vercel.app/api?type=waving&height=160&color=gradient&text=SurgVISTA:&section=header&fontAlign=15&fontSize=42&textBg=false&descAlignY=45&fontAlignY=20&descSize=23&desc=Large-scale%20Self-supervised%20Video%20Foundation%20Model%20for%20Intelligent%20Surgery&descAlign=52)
<!-- [![Arxiv Page](https://img.shields.io/badge/Arxiv-2407.15362-red?style=flat-square)](https://arxiv.org/abs/2407.15362) -->
![GitHub last commit](https://img.shields.io/github/last-commit/Innse/mSTAR?style=flat-square)
[![Hugging face](https://img.shields.io/badge/%F0%9F%A4%97%20%20-SurgVISTA-yellow)](https://huggingface.co/syangcw/SurgVISTA)
--- 

## 🔧 Status

- [x] Paper submitted
- [ ] Paper accepted
- [ ] Full code released
---

## 🔬 Abstract
<img src="assets/IntelligentSurgery.png" width="300px" align="right"/>

Computer-Assisted Intervention (CAI) has the potential to revolutionize modern surgery, with surgical scene understanding serving as a critical component in supporting decision-making, improving procedural efficacy, and ensuring intraoperative safety.
While existing AI-driven approaches alleviate annotation burdens via self-supervised spatial representation learning, their lack of explicit temporal modeling during pre-training fundamentally restricts the capture of dynamic surgical contexts, resulting in incomplete spatiotemporal understanding. In this work, we introduce the first video-level surgical pre-training framework
that enables joint spatiotemporal representation learning from large-scale surgical video data. To achieve this, we constructed a large-scale surgical video dataset comprising 3,650 videos and approximately 3.55 million frames, spanning more than 20 surgical procedures and over 10 anatomical structures. Building upon this dataset, we propose **SurgVISTA** (**Surg**ical
**Vi**deo-level **S**patial-**T**emporal **A**rchitecture), a reconstruction-based pre-training method that captures intricate spatial structures
and temporal dynamics through joint spatiotemporal modeling. Additionally, SurgVISTA incorporates image-level knowledge distillation guided by a surgery-specific expert to enhance the learning of fine-grained anatomical and semantic features. To
validate its effectiveness, we established a comprehensive benchmark comprising 13 video-level datasets spanning six surgical procedures across four tasks. Extensive experiments demonstrate that SurgVISTA consistently outperforms both natural- and surgical-domain pre-trained models, demonstrating strong potential to advance intelligent surgical systems in clinically meaningful scenarios.

## ⚙️ Installation
Instructions for setting up the environment...
### OS Requirements
This repo has been tested on the following system and GPU:
- Ubuntu 22.04.3 LTS
- NVIDIA H800 PCIe 80GB


First clone the repo and cd into the directory:

```bash
git clone https://github.com/isyangshu/SurgVISTA
cd SurgVISTA
```

To get started, create a conda environment containing the required dependencies:

```bash
conda env create -f SurgVISTA.yml
```
Activate the environment:
```bash
conda activate SurgVISTA

## 📂 Data
Download and preprocess the dataset...

## 🧠 Pre-training
Run the pre-training pipeline on your own data...

## 🎯 Finetuning
Fine-tune the pre-trained model on downstream tasks...