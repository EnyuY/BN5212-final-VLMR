#!/bin/bash
dataset="mimic_cxr"
base_dir="/scratch/e1505296/my_project/LLM-RAG/dataset/mimic-cxr-jpg/2.0.0/files/"
sn_annotation="./dataset/annotation/final_single_view_no_long_add1score_sentence_level.json"
sw_annotation="./dataset/annotation/final_single_view_with_long_add1score_sentence_level.json"
mn_annotation="./dataset/annotation/final_multi_view_no_long_add1score_sentence_level.json"
mw_annotation="./dataset/annotation/final_multi_view_with_long_add1score_sentence_level.json"
#vicuna_model="./hf/vicuna-7b-v1.5"
vicuna_model="./hf/Tiny-Vicuna-1B"
rad_dino_path="./hf/rad-dino"
cxr_bert_path="./hf/BiomedVLP-CXR-BERT-specialized"
chexbert_path="./hf/chexbert.pth"
bert_path="./hf/bert-base-uncased"
version="train_stage1_4096"
savepath="./save/$dataset/$version"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --dataset ${dataset} \
    --data_ratio 1\
    --sn_annotation ${sn_annotation} \
    --sw_annotation ${sw_annotation} \
    --mn_annotation ${mn_annotation} \
    --mw_annotation ${mw_annotation} \
    --base_dir ${base_dir} \
    --vicuna_model ${vicuna_model} \
    --rad_dino_path ${rad_dino_path} \
    --cxr_bert_path ${cxr_bert_path} \
    --chexbert_path ${chexbert_path} \
    --bert_path ${bert_path} \
    --batch_size 40 \
    --val_batch_size 4 \
    --freeze_vm True \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 50 \
    --max_new_tokens 150 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    --max_epochs 2 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    --stage_class 1 \
    --llm_use_lora False \
    --llm_r 32 \
    --llm_alpha 64 \
    --lora_dropout 0.1 \
    --accumulate_grad_batches 2 \
    --loss_mode 'sentence' \
    --sentence_ratio 0.75 \
    --learning_rate 3e-4 \
    --visual_token_number 128 \
    --test_mode 'train_1' \
    --test_batch_size 8 \
    2>&1 | tee -a ${savepath}/log.txt
