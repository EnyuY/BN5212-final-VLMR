# LLM-Based Retrieval-Augmented Cyclic Report-Image Generation for CXR


## Overview


Radiology report generation model for MIMIC-CXR dataset using large language models (NUS-BN5212 project).


## Directory Structure

```
VLMR_dev/
├── scripts/              # Training and testing scripts
│   ├── train_stage1.sh   # Stage 1 training (frozen LLM)
│   ├── train_stage2.sh   # Stage 2 training (LoRA fine-tuning)
│   └── test.sh           # Testing script
├── vanda/                # PBS job submission scripts for HPC
│   ├── vanda_setup.sh    # Environment setup
│   ├── aaa_01_TRAIN_S1.txt  # Stage 1 training job
│   ├── aaa_02_TRAIN_S2.txt  # Stage 2 training job
│   └── aaa_04_Test.txt      # Testing job
├── train.py              # Main training entry point
├── config/config.py      # Configuration parameters
├── models/               # Model implementations
│   ├── LLM_RG4.py        # Main model (7B version)
│   ├── LLM_RG4_1B_2048.py  # 1B version (2048 tokens)
│   ├── LLM_RG4_1B_4096.py  # 1B version (4096 tokens)
│   ├── LLM_RAG.py        # RAG-augmented model
│   ├── chexbert.py       # CheXbert evaluation model
│   └── metrics.py        # Evaluation metrics
├── dataset/              # Data loading
│   ├── data_module.py    # PyTorch Lightning DataModule
│   ├── data_helper_sn.py # Single-view, no prior report
│   ├── data_helper_sw.py # Single-view, with prior report
│   ├── data_helper_mn.py # Multi-view, no prior report
│   ├── data_helper_mw.py # Multi-view, with prior report
│   └── annotation/       # Annotation files
├── evalcap/              # NLG evaluation metrics
│   ├── bleu/             # BLEU metric
│   ├── rouge/            # ROUGE metric
│   ├── cider/            # CIDEr metric
│   └── meteor/           # METEOR metric
├── DiscBERT/             # Report quality discriminator
├── lightning_tools/      # PyTorch Lightning utilities
└── hf/                   # Pre-trained models directory
```

## Requirements

- Python 3.9
- PyTorch 2.1.0 + CUDA 11.8
- Dependencies: see `requirements.txt`

Installation:
```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Image Data

Download MIMIC-CXR-JPG 2.0.0 from PhysioNet:
```
https://physionet.org/content/mimic-cxr-jpg/2.0.0/
```

### 2. Text Annotations

Download MIMIC-RG4 annotation files:
```
https://drive.google.com/file/d/1X8V1H6oxxGfutGsLFofXDzvOnoq7BEyf/view
```

Extract to `dataset/annotation/`

### 3. Pre-trained Models

Download from Hugging Face:
- Vicuna-7b-v1.5 or Tiny-Vicuna-1B
- rad-dino
- BiomedVLP-CXR-BERT-specialized
- bert-base-uncased

CheXbert weights:
```
https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9
```

Place all models in `hf/` directory

### 4. Model Code Modifications

**Required:** Modify two files:

1. Replace `BiomedVLP-CXR-BERT-specialized/modeling_cxrbert.py`:
   Use the version in `hf/BiomedVLP-CXR-BERT-specialized/modeling_cxrbert.py`

2. Modify `transformers/models/llama/modeling_llama.py` line 1192:
   ```python
   loss_fct = nn.CrossEntropyLoss(reduction='none')  # Change from 'mean'
   ```

### 5. Evaluation Tools

Download evalcap:
```
https://drive.google.com/file/d/1B1_WUotp4IYFiQiIGVPb2ppyGsh4TtIH/view
```

Extract to `./evalcap`

## Usage

### Local Execution

1. **Train Stage 1** (train vision-text mapping layer)
   ```bash
   bash scripts/train_stage1.sh
   ```

2. **Train Stage 2** (LoRA fine-tune LLM)
   ```bash
   bash scripts/train_stage2.sh
   ```

3. **Test**
   ```bash
   bash scripts/test.sh
   ```
   Modify `test_mode` variable to select test scenario (sn/sw/mn/mw)

### HPC Cluster Execution (PBS)

1. **Configure paths**: Edit `vanda/vanda_setup.sh`
   ```bash
   WORKDIR="/scratch/$USER/my_project/VLMR_dev"
   IMAGE="/path/to/singularity/image.sif"
   ENV_DIR="/path/to/python/venv"
   ```

2. **Submit jobs**:
   ```bash
   qsub vanda/aaa_01_TRAIN_S1.txt  # Stage 1 training
   qsub vanda/aaa_02_TRAIN_S2.txt  # Stage 2 training
   qsub vanda/aaa_04_Test.txt      # Testing
   ```

## Key Parameters

### Training Parameters (modify in scripts/*.sh)

- `stage_class`: Training stage (1 or 2)
- `llm_use_lora`: Use LoRA in Stage 2
- `visual_token_number`: Number of visual tokens (128 or 64)
- `batch_size`: Training batch size
- `learning_rate`: Learning rate (default 3e-4)
- `max_epochs`: Number of training epochs

### Testing Parameters

- `test_mode`: Test scenario
  - `sn`: Single-view, no prior report
  - `sw`: Single-view, with prior report
  - `mn`: Multi-view, no prior report
  - `mw`: Multi-view, with prior report

## Model Architecture

- Visual encoder: rad-dino (frozen)
- Text encoder: BiomedVLP-CXR-BERT (frozen)
- Alignment and prompt projection (APPA): learnable queries, multi-head cross-attention, linear projections to d_LLM, layer normalization
- Lateral/text branches and IIT fusion: lateral image branch, text branch, and Image-Image-Text fusion head
- Language model: Vicuna-7B-v1.5 (7B) or Tiny-Vicuna-1B (1B)
  - Stage 1: freeze LLM; train alignment and fusion modules
  - Stage 2: LoRA on LLM (rank=32, alpha=64); train lateral/text branches and IIT fusion
- Retrieval-augmented generation (RAG): encode external text; cross-attend with learnable queries; project to d_LLM; insert via a special prompt token; auxiliary loss on RAG token prediction
- SAIA: two-pass generation; the first-pass draft is used as RAG input in the second pass
- Loss: token-level cross-entropy averaged over tokens plus sentence-weighted term (λ=0.75) and RAG auxiliary term (γ=0.1)

### Reported variants
- LLM_RG4 (7B)
- LLM_RG4_1B_2048 (1B, 2048 embedding)
- LLM_RG4_1B_4096 (1B, 4096 bottleneck)
- LLM_RAG (RAG augmentation; used with SAIA in two-pass setting)

## Output

Training outputs saved to `save/<dataset>/<version>/`:
- `pths/`: checkpoint files
- `log.txt`: training log

Testing outputs saved to `save/<dataset>/<version>/<test_mode>/`


## Acknowledgments

This implementation is based on and adapted from the following works:

### LLM-RG4
```bibtex
@misc{wang2024llmrg4flexiblefactualradiology,
      title={LLM-RG4: Flexible and Factual Radiology Report Generation across Diverse Input Contexts}, 
      author={Zhuhao Wang and Yihua Sun and Zihan Li and Xuan Yang and Fang Chen and Hongen Liao},
      year={2024},
      eprint={2412.12001},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.12001}, 
}
```

### R2GenGPT
```bibtex
@inproceedings{wang2023r2gengpt,
      title={R2GenGPT: Radiology Report Generation with Frozen LLMs},
      author={Wang, Zhanyu and Tang, Mingkang and Wang, Lei and others},
      booktitle={Medical Imaging with Deep Learning},
      year={2023}
}
```

### CheXbert
```bibtex
@inproceedings{smit2020chexbert,
      title={CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT},
      author={Smit, Akshay and Jain, Saahil and Rajpurkar, Pranav and others},
      booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
      year={2020}
}
```

We thank the authors for making their code publicly available.

## Notes

1. Modify all paths according to your environment
2. Ensure dataset path is correct: `base_dir` points to MIMIC-CXR image root directory
3. Stage 2 training requires loading Stage 1 checkpoint (`visual_delta_file` parameter)
4. Testing requires specifying trained checkpoint (`delta_file` parameter)
5. Random seed fixed at 42 (set in `train.py`)

