import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import AutoImageProcessor
from models.chexbert import CheXbert
import numpy as np
import pandas as pd
from models.metrics import compute_mlc
from transformers import get_cosine_schedule_with_warmup
import ipdb

class LLM_RAG(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.visual_encoder = AutoModel.from_pretrained(args.rad_dino_path)
        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.rad_dino_path} -- Done')
        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.cxr_bert_path, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(args.cxr_bert_path, trust_remote_code=True)
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        print(f'Loading Frozen text encoder:{args.cxr_bert_path} -- Done')

        if args.stage_class == 1:
            self.APPA_visual_query_tokens = nn.Parameter(
                    torch.zeros(1, args.visual_token_number, 768)
                )
            self.APPA_visual_query_tokens.data.normal_(mean=0.0, std=0.02)
            self.APPA_crossattention_block = nn.MultiheadAttention(768, 12, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.APPA_layernorm_768_1 = nn.LayerNorm(768)
            self.APPA_llama_proj_1 = nn.Linear(768, 768)
            self.APPA_layernorm_768_2 = nn.LayerNorm(768)
            self.APPA_llama_proj_2 = nn.Linear(768, 2048)
            self.APPA_llama_proj_3 = nn.Linear(2048, 4096)
            self.APPA_layernorm_4096_1 = nn.LayerNorm(4096)

        elif args.stage_class == 2:
            # APPA
            self.APPA_visual_query_tokens = nn.Parameter(
                    torch.zeros(1, args.visual_token_number, 768)
                )
            self.APPA_visual_query_tokens.data.normal_(mean=0.0, std=0.02)
            self.APPA_crossattention_block = nn.MultiheadAttention(768, 12, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.APPA_layernorm_768_1 = nn.LayerNorm(768)
            self.APPA_llama_proj_1 = nn.Linear(768, 768)
            self.APPA_layernorm_768_2 = nn.LayerNorm(768)
            self.APPA_llama_proj_2 = nn.Linear(768, 2048)
            self.APPA_llama_proj_3 = nn.Linear(2048, 4096)
            self.APPA_layernorm_4096_1 = nn.LayerNorm(4096)

            self.lateral_crossattention_block = nn.MultiheadAttention(768, 12, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.lateral_layernorm_768_1 = nn.LayerNorm(768)
            self.lateral_llama_proj_1 = nn.Linear(768, 768)
            self.lateral_layernorm_768_2 = nn.LayerNorm(768)
            self.lateral_llama_proj_2 = nn.Linear(768, 2048)
            self.lateral_llama_proj_3 = nn.Linear(2048, 4096)
            self.lateral_layernorm_4096_1 = nn.LayerNorm(4096)

            self.text_crossattention_block = nn.MultiheadAttention(768, 12, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.text_layernorm_768_1 = nn.LayerNorm(768)
            self.text_llama_proj_1 = nn.Linear(768, 768)
            self.text_layernorm_768_2 = nn.LayerNorm(768)
            self.text_llama_proj_2 = nn.Linear(768, 2048)
            self.text_llama_proj_3 = nn.Linear(2048, 4096)
            self.text_layernorm_4096_1 = nn.LayerNorm(4096)

            self.iit_proj = nn.Linear(4096 * 3,4096)
            self.iit_layernorm = nn.LayerNorm(4096)

        # RAG
        self.rag_crossattention_block = nn.MultiheadAttention(768, 12, dropout=0.1, add_bias_kv=True, add_zero_attn=True)
        self.rag_layernorm_768_1 = nn.LayerNorm(768)
        self.rag_llama_proj_1 = nn.Linear(768, 2048)
        self.rag_llama_proj_2 = nn.Linear(2048, 4096)
        self.rag_layernorm_4096_1 = nn.LayerNorm(4096)

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.vicuna_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.vicuna_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.vicuna_model,
                torch_dtype=torch.bfloat16,
            )

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                # target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                lora_dropout=args.lora_dropout,
                bias='none',
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.end_sym = args.end_sym
        self.PA_fi_prompt = 'Write radiological finding and impression for this posterior anterior chest X-ray visual feature'
        self.PA_f_prompt = 'Write radiological finding section of report for posterior anterior chest X-ray visual feature'
        self.PA_i_prompt = 'Write radiological impression section of report for posterior anterior chest X-ray visual feature'
        self.AP_fi_prompt = 'Write radiological finding and impression for this anterior posterior chest X-ray visual feature'
        self.AP_f_prompt = 'Write radiological finding section of report for anterior posterior chest X-ray visual feature'
        self.AP_i_prompt = 'Write radiological impression section of report for anterior posterior chest X-ray visual feature'

        self.fi_prompt = 'Write radiological finding and impression for this chest X-ray examination'
        self.f_prompt = 'Write radiological finding section of report for chest X-ray examination'
        self.i_prompt = 'Write radiological impression section of report for chest X-ray examination'

        self.val_step_outputs = []
        self.val_sn_step_outputs = []
        self.val_sw_step_outputs = []
        self.val_mn_step_outputs = []
        self.val_mw_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        visual_state_dict = self.visual_encoder.state_dict()
        if args.visual_delta_file is not None:
            state_dict = \
            torch.load(args.visual_delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.visual_delta_file}')
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_APPA_img(self, images):
        outputs = self.visual_encoder(images).last_hidden_state
        outputs = outputs[:,1:,:]
        vision_query_tokens = self.APPA_visual_query_tokens.expand(images.shape[0], -1, -1)
        inputs_llama = self.APPA_crossattention_block(vision_query_tokens.transpose(0, 1),
                                                         outputs.transpose(0, 1),
                                                         outputs.transpose(0, 1))[0].transpose(0, 1)
        inputs_llama = self.APPA_layernorm_768_1(inputs_llama)
        inputs_llama1 = self.APPA_llama_proj_1(inputs_llama)
        inputs_llama1 = self.APPA_layernorm_768_2(inputs_llama1) + inputs_llama

        inputs_llama2 = self.APPA_llama_proj_2(inputs_llama1)
        inputs_llama2 = F.gelu(inputs_llama2)
        inputs_llama2 = self.APPA_llama_proj_3(inputs_llama2)
        inputs_llama2 = self.APPA_layernorm_4096_1(inputs_llama2)
        atts_llama = torch.ones(inputs_llama2.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama2, atts_llama, inputs_llama1

    def encode_lateral_image(self, images, query_feature):
        outputs = self.visual_encoder(images).last_hidden_state
        outputs = outputs[:,1:,:]
        inputs_llama = self.lateral_crossattention_block(query_feature.transpose(0, 1),
                                                         outputs.transpose(0, 1),
                                                         outputs.transpose(0, 1))[0].transpose(0, 1)
        inputs_llama = self.lateral_layernorm_768_1(inputs_llama)
        inputs_llama1 = self.lateral_llama_proj_1(inputs_llama)
        inputs_llama1 = self.lateral_layernorm_768_2(inputs_llama1) + inputs_llama
        inputs_llama1 = self.lateral_llama_proj_2(inputs_llama1)
        inputs_llama1 = F.gelu(inputs_llama1)
        inputs_llama1 = self.lateral_llama_proj_3(inputs_llama1)
        inputs_llama1 = self.lateral_layernorm_4096_1(inputs_llama1)
        atts_llama = torch.ones(inputs_llama1.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama1, atts_llama

    def encode_text(self, text,query_feature):
        tokens = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.args.max_length,
        )
        input_ids = tokens["input_ids"].to(query_feature.device)
        token_type_ids = tokens["token_type_ids"].to(query_feature.device)
        attention_mask = tokens["attention_mask"].to(query_feature.device)
        output = self.text_encoder(input_ids, attention_mask, token_type_ids,
                        return_dict=True, mode="text")
        outputs = output.last_hidden_state
        inputs_llama = self.text_crossattention_block(query_feature.transpose(0, 1),
                                                         outputs.transpose(0, 1),
                                                         outputs.transpose(0, 1))[0].transpose(0, 1)
        inputs_llama = self.text_layernorm_768_1(inputs_llama)
        inputs_llama1 = self.text_llama_proj_1(inputs_llama)
        inputs_llama1 = self.text_layernorm_768_2(inputs_llama1) + inputs_llama
        inputs_llama1 = self.text_llama_proj_2(inputs_llama1)
        inputs_llama1 = F.gelu(inputs_llama1)
        inputs_llama1 = self.text_llama_proj_3(inputs_llama1)
        inputs_llama1 = self.text_layernorm_4096_1(inputs_llama1)
        atts_llama = torch.ones(inputs_llama1.size()[:-1], dtype=torch.long).to(query_feature.device)
        return inputs_llama1, atts_llama

    def encode_rag_content(self, rag_text, query_feature):
        """Enhanced RAG content encoding with dimension safety checks"""
        batch_size = query_feature.shape[0]
        
        # Check if RAG content is empty
        if not rag_text or all(not t for t in rag_text):
            # Create a consistent placeholder with fixed size
            empty_embeds = torch.zeros(batch_size, 4, 4096, dtype=query_feature.dtype, device=query_feature.device)
            empty_atts = torch.ones(batch_size, 4, dtype=torch.long, device=query_feature.device)
            return empty_embeds, empty_atts

        # Process RAG text through BERT
        tokens = self.bert_tokenizer(
            rag_text,
            return_tensors="pt",
            padding="max_length",  # Use consistent max_length
            truncation=True,
            max_length=128,  # Fixed size to ensure consistency
        ).to(query_feature.device)
        
        # Process through text encoder
        output = self.text_encoder(tokens.input_ids, tokens.attention_mask, 
                                  return_dict=True, mode="text")
        bert_outputs = output.last_hidden_state
        
        # Apply cross-attention between query and RAG content
        rag_embeds = self.rag_crossattention_block(query_feature.transpose(0, 1),
                                              bert_outputs.transpose(0, 1),
                                              bert_outputs.transpose(0, 1))[0].transpose(0, 1)
        
        # Process through projection layers
        rag_embeds = self.rag_layernorm_768_1(rag_embeds)
        rag_embeds = self.rag_llama_proj_1(rag_embeds)
        rag_embeds = F.gelu(rag_embeds)
        rag_embeds = self.rag_llama_proj_2(rag_embeds)
        rag_embeds = self.rag_layernorm_4096_1(rag_embeds)
        
        # Ensure attention mask has the same sequence length as the embeddings
        if tokens.attention_mask.shape[1] != rag_embeds.shape[1]:
            # Create a new matching attention mask
            atts_rag = torch.ones(batch_size, rag_embeds.shape[1], 
                                  dtype=torch.long, device=query_feature.device)
        else:
            atts_rag = tokens.attention_mask
            
        return rag_embeds, atts_rag

    def prompt_sn(self, img_embeds, atts_img, samples, rag_embeds=None, atts_rag=None): # update for RAG
        batch_size = img_embeds.shape[0]
        prompt = []
        for b in range(batch_size):
            h_i = samples['h_i'][b]
            if samples['APPA_flag'][b] == 'AP':
                if samples['finding_flag'][b] and samples['impression_flag'][b]:
                    prompt_text = self.AP_fi_prompt
                else:
                    if samples['finding_flag'][b]:
                        prompt_text = self.AP_f_prompt
                    if samples['impression_flag'][b]:
                        prompt_text = self.AP_i_prompt
            elif samples['APPA_flag'][b] == 'PA':
                if samples['finding_flag'][b] and samples['impression_flag'][b]:
                    prompt_text = self.PA_fi_prompt
                else:
                    if samples['finding_flag'][b]:
                        prompt_text = self.PA_f_prompt
                    if samples['impression_flag'][b]:
                        prompt_text = self.PA_i_prompt
            else:
                prompt_text = self.f_prompt

            # Add specific RAG prefix/instruction
            prompt.append(f'Human: {h_i} <Img><ImageHere></Img> Using the following relevant information: <RAG><RAGHere></RAG> {prompt_text}. \nAssistant:')

        # Split the prompt into three parts: before ImageHere, between ImageHere and RAGHere, and after RAGHere
        p_before = []
        p_middle = []
        p_after = []
        for p in prompt:
            parts = p.split('<ImageHere>')
            before_img = parts[0]
            
            if '<RAGHere>' in parts[1]:
                middle_rag_parts = parts[1].split('<RAGHere>')
                after_img_before_rag = middle_rag_parts[0]
                after_rag = middle_rag_parts[1]
            else:
                after_img_before_rag = ""
                after_rag = parts[1]
                
            p_before.append(before_img)
            p_middle.append(after_img_before_rag)
            p_after.append(after_rag)
        
        # Tokenize and embed parts
        pad_size = self.llama_tokenizer.padding_side
        
        # Process p_before (text before image)
        self.llama_tokenizer.padding_side = 'left'
        self.llama_tokenizer.truncation_side = 'right'  # truncate history
        p_before_tokens = self.llama_tokenizer(
            p_before,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=50,
            add_special_tokens=False
        ).to(img_embeds.device)
        
        # Process p_middle with fixed length
        self.llama_tokenizer.padding_side = 'right'
        p_middle_tokens = self.llama_tokenizer(
            p_middle,
            return_tensors="pt", 
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            max_length=20  # Set a fixed length
        ).to(img_embeds.device)
        
        # Process p_after with fixed length
        p_after_tokens = self.llama_tokenizer(
            p_after, 
            return_tensors="pt", 
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            max_length=30  # Set a fixed length
        ).to(img_embeds.device)
        
        # Reset tokenizer padding side to original
        self.llama_tokenizer.padding_side = pad_size
        
        # Convert tokens to embeddings
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
        p_middle_embeds = self.embed_tokens(p_middle_tokens.input_ids)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)

        # Handle special padding for p_before similar to original code
        post_pad_length = torch.logical_not(p_before_tokens.attention_mask).sum(-1)
        new_p_before_embeds = torch.zeros(batch_size, p_before_embeds.size(1) + 1, 4096).type(img_embeds.dtype).to(img_embeds.device)
        new_p_before_mask = torch.zeros(batch_size, p_before_embeds.size(1) + 1).type(atts_img.dtype).to(img_embeds.device)
        
        for b in range(batch_size):
            post_pad_len = post_pad_length[b]
            new_p_before_embeds[b,:post_pad_len,:] = p_before_embeds[b,:post_pad_len,:]
            bos = torch.ones([1, 1], dtype=p_before_tokens.input_ids.dtype, device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            new_p_before_embeds[b,post_pad_len,:] = bos_embeds
            new_p_before_mask[b,post_pad_len:] = 1
            new_p_before_embeds[b,(post_pad_len+1):,:] = p_before_embeds[b,post_pad_len:,:]

        # If RAG embeddings are not provided, create empty placeholder
        if rag_embeds is None:
            rag_embeds = torch.zeros(batch_size, 1, 4096).type(img_embeds.dtype).to(img_embeds.device)
            atts_rag = torch.ones(batch_size, 1).type(atts_img.dtype).to(img_embeds.device)

        # Concatenate all parts: before, image, middle, RAG, after
        wrapped_img_embeds = torch.cat([
            new_p_before_embeds,        # Text before image
            img_embeds,                 # Image embeddings
            p_middle_embeds,            # Text between image and RAG
            rag_embeds,                 # RAG embeddings
            p_after_embeds              # Text after RAG
        ], dim=1)
        
        wrapped_atts_img = torch.cat([
            new_p_before_mask,           # Attention mask for text before image
            atts_img,                    # Attention mask for image
            p_middle_tokens.attention_mask,  # Attention mask for text between image and RAG
            atts_rag,                    # Attention mask for RAG
            p_after_tokens.attention_mask    # Attention mask for text after RAG
        ], dim=1)
        
        # Before returning, verify dimensions match
        wrapped_seq_len = wrapped_img_embeds.shape[1]
        wrapped_mask_len = wrapped_atts_img.shape[1]
        
        if wrapped_seq_len != wrapped_mask_len:
            print(f"WARNING: Dimension mismatch in prompt_sn - embeddings: {wrapped_seq_len}, mask: {wrapped_mask_len}")
            
            # Fix the mismatch by adjusting the longer one
            if wrapped_seq_len > wrapped_mask_len:
                wrapped_img_embeds = wrapped_img_embeds[:, :wrapped_mask_len, :]
            else:
                # Extend attention mask
                padding = torch.ones(batch_size, wrapped_seq_len - wrapped_mask_len, 
                                    device=wrapped_atts_img.device, dtype=wrapped_atts_img.dtype)
                wrapped_atts_img = torch.cat([wrapped_atts_img, padding], dim=1)
        
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples, capture_attention=False):
        image = samples["image"]
        dataset_id = samples['dataset_id'][0]
        print(dataset_id)

        def all_elements_equal(lst):
            return all(x == lst[0] for x in lst)

        lst = samples['dataset_id']
        if all_elements_equal(lst) == False:
            print(all_elements_equal(lst))
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=True
        ).to(image.device)
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        # Get RAG content if available
        rag_text = samples.get('rag_content', [''] * image.shape[0])

        # Initialize RAG variables
        rag_embeds = None
        atts_rag = None

        img_embeds = ''
        atts_img = ''
        if dataset_id == 'sn':
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            if self.args.stage_class == 1:
                batch_size = img_embeds1.shape[0]
                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.embed_tokens(bos)
                atts_bos = atts_img1[:, :1]
                img_embeds = torch.cat([bos_embeds,img_embeds1], dim=1)
                atts_img = torch.cat([atts_bos,atts_img1], dim=1)
                
                # Also encode RAG content for stage_class 1
                rag_embeds, atts_rag = self.encode_rag_content(rag_text, query_feature)
                
                # For stage 1, we need to explicitly apply RAG content
                if rag_embeds is not None:
                    # Create a weighted fusion of image and RAG embeddings
                    fusion_weight = 0.7  # Weight more toward image content, adjust as needed

                    # Match dimensions if needed (assuming rag_embeds and img_embeds have different sequence lengths)
                    rag_mean = torch.mean(rag_embeds, dim=1, keepdim=True)  # Create a compact representation
                    rag_attention = torch.sigmoid(torch.bmm(img_embeds, rag_mean.transpose(1, 2)))

                    # Apply attention-weighted RAG influence to image embeddings
                    img_embeds = img_embeds * (1 + 0.3 * rag_attention)  # Subtle influence via residual connection

            if self.args.stage_class != 1:
                zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 8192, device=img_embeds1.device)
                new_embeds = torch.cat([img_embeds1, zero_padding], dim=2)
                new_embeds = self.iit_proj(new_embeds)
                new_embeds = self.iit_layernorm(new_embeds)

                # Encode RAG content
                rag_embeds, atts_rag = self.encode_rag_content(rag_text, query_feature)

                # Pass RAG content to prompt_sn
                img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples, rag_embeds, atts_rag)

        if dataset_id == 'mn':
            APPA_image = samples["image"]
            img_embeds1, atts_img1,query_feature = self.encode_APPA_img(APPA_image)
            lateral_image = samples['lateral_image']
            lateral_embed, atts_lateral = self.encode_lateral_image(lateral_image, query_feature)
            new_embeds = ''
            zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096, device=img_embeds1.device)
            new_embeds = torch.cat([img_embeds1, lateral_embed,zero_padding], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'sw':
            image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            last_report = samples['last_report']
            last_report_embed, atts_text = self.encode_text(last_report, query_feature)
            zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096, device=img_embeds1.device)
            new_embeds = torch.cat([img_embeds1,zero_padding,last_report_embed], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)      #.expand(query_feature.shape[0], -1, -1)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'mw':
            APPA_image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image)
            lateral_image = samples['lateral_image']
            lateral_embed, atts_lateral = self.encode_lateral_image(lateral_image, query_feature)
            last_report = samples['last_report']
            last_report_embed, atts_text = self.encode_text(last_report, query_feature)
            new_embeds = torch.cat([img_embeds1, lateral_embed,last_report_embed], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        targets = torch.zeros_like(to_regress_tokens.attention_mask).long().fill_(-100)

        # only apply loss to answer tokens
        targets_idx = to_regress_tokens.attention_mask.bool()
        targets[:, -targets_idx.shape[1]:][targets_idx] = to_regress_tokens.input_ids[targets_idx]
        targets[:, -targets_idx.shape[1]] = -100

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)

        device = inputs_embeds.device
        length = attention_mask.size(1)
        zero_tensor = torch.zeros(attention_mask.size(0), length - 100, device=device)
        scores_ori = samples['scores']
        scores = torch.cat([zero_tensor, scores_ori], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        shift_scores = scores[..., 1:].contiguous()
        shift_scores = shift_scores.view(-1)

        average_loss = loss.sum() / (torch.count_nonzero(loss) + 0.001)
        key_loss = (loss * shift_scores)
        key_count = torch.count_nonzero(key_loss) + 0.001
        key_loss = key_loss.sum()
        key_loss = key_loss/key_count
        total_loss = average_loss + self.args.sentence_ratio * key_loss
        if self.args.loss_mode == 'None':
            total_loss = average_loss

        # Add auxiliary loss to encourage RAG utilization
        if rag_embeds is not None and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Get the output embeddings from the final layer
            output_embeds = outputs.hidden_states[-1]
            
            # Only calculate attention if RAG embeddings have non-zero values
            if not torch.all(torch.isclose(rag_embeds, torch.zeros_like(rag_embeds))):
                if capture_attention:
                    # Use the capture method when requested (validation/testing)
                    attention_dir = os.path.join(self.hparams.savedmodel_path, 'attention_maps')
                    os.makedirs(attention_dir, exist_ok=True)
                    
                    rag_attention = self.capture_rag_attention(
                        output_embeds, 
                        rag_embeds,
                        save_path=attention_dir,
                        sample_ids=samples.get('id', None)
                    )
                else:
                    # During training, calculate attention for the loss
                    # Normalize embeddings for more stable attention scores
                    norm_output = F.normalize(output_embeds, p=2, dim=-1)
                    norm_rag = F.normalize(rag_embeds, p=2, dim=-1)
                    
                    # Calculate cosine similarity-based attention
                    rag_attention = torch.bmm(norm_output, norm_rag.transpose(1, 2)).mean()
                    
                    # Add to loss with adaptive weight based on attention magnitude
                    rag_loss = -0.1 * torch.clamp(rag_attention, min=0.01, max=1.0)
                    total_loss = total_loss + rag_loss
                    
                    # Log attention statistics
                    self.log('rag_attention', rag_attention.item(), prog_bar=True)

        return {"loss": total_loss}

    def capture_rag_attention(self, output_embeds, rag_embeds, save_path=None, sample_ids=None):
        """Capture RAG attention maps without affecting training computation graph.
        Only call this method during validation or inference, not during training.
        """
        with torch.no_grad():
            try:
                # Calculate full attention map
                attention_maps = torch.bmm(output_embeds, rag_embeds.transpose(1, 2))
                
                # Calculate statistics for logging
                attention_mean = attention_maps.mean().item()
                attention_max = attention_maps.max().item()
                attention_sparsity = (attention_maps < 0.01).float().mean().item()
                
                # Log statistics
                self.log('rag_attention_mean', attention_mean, sync_dist=True)
                self.log('rag_attention_max', attention_max, sync_dist=True)
                self.log('rag_attention_sparsity', attention_sparsity, sync_dist=True)
                
                # Save attention maps if path is provided
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    
                    batch_size = attention_maps.shape[0]
                    for i in range(batch_size):
                        # Get sample ID or use index if not provided
                        sample_id = sample_ids[i] if sample_ids is not None else f"sample_{i}"
                        
                        # Save attention map as numpy file for later analysis
                        attn_map = attention_maps[i].cpu().numpy()
                        np.save(os.path.join(save_path, f"attn_map_{sample_id}.npy"), attn_map)
                        
                        # Optionally create visualization
                        if i < 3:  # Only visualize first few samples to avoid overhead
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(10, 8))
                            plt.imshow(attn_map, cmap='viridis')
                            plt.colorbar()
                            plt.title(f"RAG Attention Map - Sample {sample_id}")
                            plt.xlabel("RAG Token Position")
                            plt.ylabel("Output Token Position")
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_path, f"attn_map_{sample_id}.png"))
                            plt.close()
                    
                    # After saving all maps
                    torch.cuda.empty_cache()  # Free up GPU memory after visualization
                
                # Return mean attention for regular loss computation
                return attention_maps.mean()
            except Exception as e:
                print(f"Error capturing attention maps: {e}")
                # Return zero to not affect loss
                return torch.tensor(0.0, device=output_embeds.device)

    @staticmethod
    def analyze_attention_maps(attention_dir, top_k=5):
        """
        Analyze saved attention maps to identify patterns
        
        Args:
            attention_dir: Directory containing saved attention maps
            top_k: Number of highest attention positions to report
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from glob import glob
        
        # Find all saved numpy attention maps
        attn_files = glob(os.path.join(attention_dir, "attn_map_*.npy"))
        
        if not attn_files:
            print(f"No attention maps found in {attention_dir}")
            return
        
        # Aggregate statistics across maps
        all_means = []
        all_maxes = []
        all_sparsity = []
        
        for attn_file in attn_files:
            attn_map = np.load(attn_file)
            
            # Calculate statistics
            mean_attn = np.mean(attn_map)
            max_attn = np.max(attn_map)
            sparsity = np.mean(attn_map < 0.01)  # % of values below threshold
            
            all_means.append(mean_attn)
            all_maxes.append(max_attn)
            all_sparsity.append(sparsity)
            
            # Find top-k attention positions
            flat_indices = np.argsort(attn_map.flatten())[-top_k:]
            for idx in flat_indices:
                out_pos, rag_pos = np.unravel_index(idx, attn_map.shape)
                print(f"File: {os.path.basename(attn_file)}, "
                      f"High attention: {attn_map[out_pos, rag_pos]:.4f} "
                      f"at output token {out_pos}, RAG token {rag_pos}")
        
        # Plot overall statistics
        print(f"\nAnalysis of {len(attn_files)} attention maps:")
        print(f"Mean attention: {np.mean(all_means):.4f} (std: {np.std(all_means):.4f})")
        print(f"Max attention: {np.mean(all_maxes):.4f} (std: {np.std(all_maxes):.4f})")
        print(f"Average sparsity: {np.mean(all_sparsity):.4f} (std: {np.std(all_sparsity):.4f})")
        
        # Create summary visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(all_means, bins=20)
        plt.title("Distribution of Mean Attention")
        plt.xlabel("Mean Attention Value")
        
        plt.subplot(1, 2, 2)
        plt.hist(all_maxes, bins=20)
        plt.title("Distribution of Max Attention")
        plt.xlabel("Max Attention Value")
        
        plt.tight_layout()
        plt.savefig(os.path.join(attention_dir, "attention_statistics.png"))
        plt.close()

    @staticmethod
    def test_rag_effectiveness(attention_dir, threshold=0.05):
        """Test if RAG content is effectively being used by the model"""
        import os
        import numpy as np
        from glob import glob
        
        attn_files = glob(os.path.join(attention_dir, "attn_map_*.npy"))
        
        if not attn_files:
            return False
        
        # Calculate average maximum attention per token
        max_attentions = []
        for file in attn_files:
            attn = np.load(file)
            # For each output token, what's the max attention to any RAG token
            token_max_attns = np.max(attn, axis=1)
            max_attentions.append(np.mean(token_max_attns))
        
        avg_max_attn = np.mean(max_attentions)
        return avg_max_attn > threshold, avg_max_attn

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res,chexbert_f1):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'pths'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'pths',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}_chexbert{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'],
                                                                        eval_res['CIDEr'],chexbert_f1),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(samples["image"].device)  # Make sure it's on the right device
        
        image = samples["image"]
        dataset_id = samples['dataset_id'][0]
        
        # Get RAG content if available
        rag_text = samples.get('rag_content', [''] * image.shape[0])
        
        img_embeds = ''
        atts_img = ''
        print(dataset_id)
        if dataset_id == 'sn':
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            if self.args.stage_class == 1:
                batch_size = img_embeds1.shape[0]
                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=img_embeds1.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.embed_tokens(bos)
                atts_bos = atts_img1[:, :1]
                img_embeds = torch.cat([bos_embeds, img_embeds1], dim=1)
                atts_img = torch.cat([atts_bos, atts_img1], dim=1)
                
                # Also encode RAG content for stage_class 1
                rag_embeds, atts_rag = self.encode_rag_content(rag_text, query_feature)
                
                # Apply RAG influence for stage 1
                if rag_embeds is not None:
                    # Apply weighted RAG influence
                    rag_mean = torch.mean(rag_embeds, dim=1, keepdim=True)
                    rag_attention = torch.sigmoid(torch.bmm(img_embeds, rag_mean.transpose(1, 2)))
                    img_embeds = img_embeds * (1 + 0.3 * rag_attention)
                    
            if self.args.stage_class != 1:
                zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 8192,
                                         device=img_embeds1.device)
                new_embeds = torch.cat([img_embeds1, zero_padding], dim=2)
                new_embeds = self.iit_proj(new_embeds)
                new_embeds = self.iit_layernorm(new_embeds)
                
                # Encode RAG content
                rag_embeds, atts_rag = self.encode_rag_content(rag_text, query_feature)
                
                # Pass RAG content to prompt_sn
                img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples, rag_embeds, atts_rag)
        
        # Process other dataset types (mn, sw, mw)...
        if dataset_id == 'mn':
            APPA_image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image)
            lateral_image = samples['lateral_image']
            lateral_embed, atts_lateral = self.encode_lateral_image(lateral_image, query_feature)
            zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096,
                                       device=img_embeds1.device)
            new_embeds = torch.cat([img_embeds1, lateral_embed, zero_padding], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'sw':
            image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            last_report = samples['last_report']
            last_report_embed, atts_text = self.encode_text(last_report, query_feature)
            zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096,
                                       device=img_embeds1.device)
            new_embeds = torch.cat([img_embeds1, zero_padding, last_report_embed], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'mw':
            APPA_image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image)
            lateral_image = samples['lateral_image']
            lateral_embed, atts_lateral = self.encode_lateral_image(lateral_image, query_feature)
            last_report = samples['last_report']
            last_report_embed, atts_text = self.encode_text(last_report, query_feature)
            new_embeds = torch.cat([img_embeds1, lateral_embed, last_report_embed], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, -1:]

        inputs_embeds = torch.cat([img_embeds, bos_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_bos], dim=1)
        
        # Debug info - print shapes before generate
        print(f"[Debug] inputs_embeds shape: {inputs_embeds.shape}, attention_mask shape: {attention_mask.shape}")
        
        # Ensure all batch items have same sequence length
        max_len = attention_mask.shape[1]
        if inputs_embeds.shape[1] != max_len:
            print(f"[Warning] Fixing dimension mismatch: {inputs_embeds.shape[1]} vs {max_len}")
            if inputs_embeds.shape[1] > max_len:
                inputs_embeds = inputs_embeds[:, :max_len, :]
            else:
                # Pad embeddings to match attention mask
                padding = torch.zeros(batch_size, max_len - inputs_embeds.shape[1], 
                                     inputs_embeds.shape[2], device=inputs_embeds.device)
                inputs_embeds = torch.cat([inputs_embeds, padding], dim=1)
        
        try:
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=self.hparams.temperature,
            )
            hypo = [self.decode(i) for i in outputs]
            ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        except RuntimeError as e:
            print(f"Error during generation: {e}")
            # Fallback to provide a minimal output to prevent training failure
            hypo = ["Error during generation"] * len(samples["input_text"])  
            ref = samples["input_text"]
            
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        if dataset_id == 'sn':
            self.val_sn_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        if dataset_id == 'sw':
            self.val_sw_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        if dataset_id == 'mn':
            self.val_mn_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        if dataset_id == 'mw':
            self.val_mw_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        

        
        # Always save attention maps for the first 5 batches
        if batch_idx < 5:  # Always save first 5 batches
            attention_dir = os.path.join(self.hparams.savedmodel_path, 'debug_attention_maps')
            os.makedirs(attention_dir, exist_ok=True)
            
            print(f"Saving attention map for batch {batch_idx} to {attention_dir}")
            
            # Run forward with explicit attention capture
            with torch.no_grad():
                _ = self(samples, capture_attention=True)
            
            print(f"Attention maps saved to {attention_dir}")
        
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def calculate_metric(self,val_step_outputs):
        ref, hypo, ids = [], [], []
        for i in val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # chexbert
        device = torch.device('cuda:0')
        chexbert = CheXbert(
            ckpt_dir='./hf',
            bert_path=self.hparams.bert_path,
            checkpoint_path=self.hparams.chexbert_path,
            device=device
        ).to(device)

        # calculate chexbert
        ref_pred_list = []
        hypo_pred_list = []
        for index in range(len(ref)):
            refs = [ref[index]]
            hypos = [hypo[index]]
            ref_pred = chexbert(refs).squeeze().cpu().numpy()
            hypo_pred = chexbert(hypos).squeeze().cpu().numpy()
            ref_pred_list.append(ref_pred)
            hypo_pred_list.append(hypo_pred)

        ref_pred_list = np.array(ref_pred_list)
        df = pd.DataFrame(ref_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        gts_path = os.path.join(self.args.savedmodel_path, 'ref_labeled_reports.csv')
        df.to_csv(gts_path, index=False)

        hypo_pred_list = np.array(hypo_pred_list)
        df = pd.DataFrame(hypo_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        res_path = os.path.join(self.args.savedmodel_path, 'pred_labeled_reports.csv')
        df.to_csv(res_path, index=False)

        res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
        res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

        label_set = res_data.columns[1:].tolist()
        res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
        res_data[res_data == -1] = 1
        gts_data[gts_data == -1] = 1
        chexbert_f1_MACRO = 0
        if len(ref) > 100:
            metrics = compute_mlc(gts_data, res_data, label_set)
            # print(metrics)
            chexbert_f1_MACRO = metrics['F1_MACRO']

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        return metrics,eval_res

    def on_validation_epoch_end(self):
        if self.args.stage_class == 2 and len(self.val_sn_step_outputs) > 5:
            sn_ce, sn_eval = self.calculate_metric(self.val_sn_step_outputs)
            sw_ce, sw_eval = self.calculate_metric(self.val_sw_step_outputs)
            mn_ce, mn_eval = self.calculate_metric(self.val_mn_step_outputs)
            mw_ce, mw_eval = self.calculate_metric(self.val_mw_step_outputs)

            ce_list = [sn_ce, sw_ce, mn_ce, mw_ce]
            ce = {}
            for key in sn_ce.keys():
                total = sum(d[key] for d in ce_list)
                ce[key] = total / len(ce_list)

            nlg_list = [sn_eval, sw_eval, mn_eval, mw_eval]
            nlg = {}
            for key in sn_eval.keys():
                total = sum(d[key] for d in nlg_list)
                nlg[key] = total / len(nlg_list)

            print(ce)
            print(nlg)

            if self.trainer.local_rank == 0:
                # if val_score > self.val_score:
                self.save_checkpoint(nlg, ce['F1_MICRO'])
        if self.args.stage_class == 1 and len(self.val_step_outputs) > 5:
            sn_ce, sn_eval = self.calculate_metric(self.val_step_outputs)
            print(sn_ce)
            print(sn_eval)
            if self.trainer.local_rank == 0:
                # if val_score > self.val_score:
                self.save_checkpoint(sn_eval, sn_ce['F1_MICRO'])
        self.val_step_outputs.clear()
        self.val_sn_step_outputs.clear()
        self.val_sw_step_outputs.clear()
        self.val_mn_step_outputs.clear()
        self.val_mw_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        dataset_id = samples['dataset_id'][0]
        print(dataset_id)

        def all_elements_equal(lst):
            return all(x == lst[0] for x in lst)

        lst = samples['dataset_id']
        if all_elements_equal(lst) == False:
            print(all_elements_equal(lst))
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=500,  #self.hparams.max_length
            add_special_tokens=False
        )
        
        # Get RAG content if available # update rag
        image = samples["image"]
        rag_text = samples.get('rag_content', [''] * image.shape[0])
        
        dataset_id = samples['dataset_id'][0]
        img_embeds = ''
        atts_img = ''
        if dataset_id == 'sn':
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            if self.args.stage_class == 1:
                batch_size = img_embeds1.shape[0]
                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=img_embeds1.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.embed_tokens(bos)
                atts_bos = atts_img1[:, :1]
                img_embeds = torch.cat([bos_embeds, img_embeds1], dim=1)
                atts_img = torch.cat([atts_bos, atts_img1], dim=1)
            if self.args.stage_class != 1:
                zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 8192,
                                           device=img_embeds1.device)
                new_embeds = torch.cat([img_embeds1, zero_padding], dim=2)
                new_embeds = self.iit_proj(new_embeds)
                new_embeds = self.iit_layernorm(new_embeds)
                
                # Encode RAG content # update rag
                rag_embeds, atts_rag = self.encode_rag_content(rag_text, query_feature)
                
                # Pass RAG content to prompt_sn # update rag
                img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples, rag_embeds, atts_rag)

        if dataset_id == 'mn':
            APPA_image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image)
            lateral_image = samples['lateral_image']
            lateral_embed, atts_lateral = self.encode_lateral_image(lateral_image, query_feature)
            zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096,
                                       device=img_embeds1.device)
            new_embeds = torch.cat([img_embeds1, lateral_embed, zero_padding], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'sw':
            image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            last_report = samples['last_report']
            last_report_embed, atts_text = self.encode_text(last_report, query_feature)
            zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096,
                                       device=img_embeds1.device)
            new_embeds = torch.cat([img_embeds1, zero_padding, last_report_embed], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'mw':
            APPA_image = samples["image"]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image)
            lateral_image = samples['lateral_image']
            lateral_embed, atts_lateral = self.encode_lateral_image(lateral_image, query_feature)
            last_report = samples['last_report']
            last_report_embed, atts_text = self.encode_text(last_report, query_feature)
            new_embeds = torch.cat([img_embeds1, lateral_embed, last_report_embed], dim=2)
            new_embeds = self.iit_proj(new_embeds)
            new_embeds = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, -1:]

        inputs_embeds = torch.cat([img_embeds, bos_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_bos], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask = attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        device = torch.device('cuda:0')
        chexbert = CheXbert(
            ckpt_dir='./hf',
            bert_path=self.hparams.bert_path,
            checkpoint_path=self.hparams.chexbert_path,
            device=device
        ).to(device)

        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # calculate chexbert
        ref_pred_list = []
        hypo_pred_list = []
        for index in range(len(ref)):
            refs = [ref[index]]
            hypos = [hypo[index]]
            ref_pred = chexbert(refs).squeeze().cpu().numpy()
            hypo_pred = chexbert(hypos).squeeze().cpu().numpy()
            ref_pred_list.append(ref_pred)
            hypo_pred_list.append(hypo_pred)

        ref_pred_list = np.array(ref_pred_list)
        df = pd.DataFrame(ref_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        gts_path = os.path.join(self.args.savedmodel_path, 'ref_labeled_reports.csv')
        df.to_csv(gts_path, index=False)

        hypo_pred_list = np.array(hypo_pred_list)
        df = pd.DataFrame(hypo_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        res_path = os.path.join(self.args.savedmodel_path, 'pred_labeled_reports.csv')
        df.to_csv(res_path, index=False)

        res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
        res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

        label_set = res_data.columns[1:].tolist()
        res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
        res_data[res_data == -1] = 1
        gts_data[gts_data == -1] = 1
        metrics = compute_mlc(gts_data, res_data, label_set)
        chexbert_f1_MICRO = metrics['F1_MICRO']
        print('chexbert_f1_MICRO: ')
        print(chexbert_f1_MICRO)
        print('metric')
        print(metrics)

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Total number of training steps
        len_data = 0
        if self.hparams.stage_class == 1:
            len_data = 172608  # len of sn
        else:
            len_data = 690432   # len of all
        total_steps = self.hparams.max_epochs * len_data / self.hparams.batch_size / self.hparams.accumulate_grad_batches  # all:345216 mw:23849 mn:45668
        # Scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps / self.hparams.max_epochs),  # 10% of total steps for warmup
            num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
