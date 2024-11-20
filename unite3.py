from tqdm import tqdm
import numpy as np
import os
import seaborn as sns
import numpy as np
import re
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

import torch
import argparse

from utils.ans_process import *
from utils.collate_fun import *
from utils.extract_response import *

from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object
import matplotlib.pyplot as plt

def softmax(x):
  x = x - np.max(x)
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  softmax_x = exp_x / sum_exp_x

  return softmax_x


def count_words_split(text):
  words = text.split()
  return len(words)

def get_top_k_tokens(outputs, tokenizer, k=10):
    logits = outputs.logits[0]
    probs = logits

    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()

    top_k_probs = []
    for idx, prob in zip(top_k_indices,probs):
        prob_item = []
        for i in idx:
            prob_item.append(prob[i])
        top_k_probs.append(prob_item)

    top_k_tokens = []
    for indices in top_k_indices:
        token_item = []
        for idx in indices:
            token_item.append(tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True))
        top_k_tokens.append(token_item)

    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append(
            {token.replace('▁','Ġ').replace('<0x0A>','/n').replace('Ċ','/n'): [prob, int(id)] for token, prob, id in zip(token, prob, id)})

    return v1

def get_union_vocab(v1, v2, v3):
    unique_tokens = []
    for v1_tokens, v2_tokens, v3_tokens in zip(v1, v2, v3):
        combined_tokens = set(v1_tokens.keys()) | set(v2_tokens.keys()) | set(v3_tokens.keys())
        unique_tokens.append(list(combined_tokens))

    return unique_tokens

def update_vocab(v1, vu, tokenizer, logits, model_name):
    for vu_token, v1_token, logit_ele in zip(vu,v1,logits):
        v1_token_ids = []
        for item in v1_token.values():
            v1_token_ids.append(item[1])
        for token in vu_token:
            if token not in v1_token.keys():
              if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                  token = token.replace('Ġ','▁')
              if token != '':
                  subtoken_id = tokenizer.convert_tokens_to_ids(token)
                  if subtoken_id != 0 and subtoken_id != None: #Mistral and Llama2 oov id 0
                      logit = logit_ele[subtoken_id]
                  else:
                      subtokens = tokenizer.tokenize(token)
                      for token_id in tokenizer.convert_tokens_to_ids(subtokens):
                          if 'llama2' in model_name:
                              if token_id != 29871:
                                  subtoken_id = token_id
                                  break
                          if 'mistral' in model_name:
                              if token_id != 29473:
                                  subtoken_id = token_id
                                  break
                          if 'deepseek' in model_name:
                              if token_id != 207:
                                  subtoken_id = token_id
                                  break
                          if 'openchat' in model_name:
                              if token_id != 28705:
                                  subtoken_id = token_id
                                  break
                          else:
                              subtoken_id = token_id
                              break
                      logit = logit_ele[subtoken_id]
              else:
                  if 'llama3' in model_name or 'qwen2' in model_name:
                      logit = logit_ele[220]
                      subtoken_id = 220
                  if 'llama2' in model_name:
                      logit = logit_ele[29871]
                      subtoken_id = 29871
                  if 'mistral' in model_name:
                      logit = logit_ele[29473]
                      subtoken_id = 29473
                  if 'deepseek' in model_name:
                      logit = logit_ele[207]
                      subtoken_id = 207
                  if 'openchat' in model_name:
                      logit = logit_ele[28705]
                      subtoken_id = 28705
              # 将{token: logit}添加到v1中
              if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                v1_token[token.replace('▁','Ġ')] = [logit,subtoken_id]
              else:
                if subtoken_id not in v1_token_ids:
                    v1_token[token] = [logit, subtoken_id]
                    v1_token_ids.append(subtoken_id)
                else:
                    v1_token[token] = [0, subtoken_id]

    v1_new = vocab_softmax(v1)
    return v1_new


def vocab_softmax(v1):
    v1_new = []
    for element in v1:
        ele = {}
        ele_values = list(element.values())
        ele_values0, ele_values1 = [], []
        for item in ele_values:
            ele_values0.append(item[0])
            ele_values1.append(item[1])
        ele_values0 = torch.softmax(torch.tensor(ele_values0), dim=0)
        for token, prob, ids in zip(element.keys(),ele_values0,ele_values1):
          ele[token] = [prob, ids]
        v1_new.append(ele)

    return v1_new

def drop_token(v1,v2,t):
    v1_new, v2_new = [], []
    # 删除在ref model中很大，但是在base model中很小的tokens
    for v1_element, v2_element in zip(v1,v2):
        v1_, v2_ = {}, {}
        for key in v1_element.keys():
            if v1_element[key][0] > t:
                v1_[key] = v1_element[key]
                v2_[key] = v2_element[key]
        v1_new.append(v1_)
        v2_new.append(v2_)
    return v1_new,v2_new


def average_and_sample(v1, v2, v3, lamda, tokenizer):
    next_token, v_avg, next_token_id1, next_token_id2, next_token_id3 = [], [], [], [], []

    for element_v1, element_v2, element_v3 in zip(v1, v2, v3):
        assert len(element_v1) == len(element_v2) == len(element_v3)

        v_new = {}

        for token1 in element_v1:
            v_new[token1] = [
                1/3 * element_v1[token1][0] +
                1/3 * element_v2[token1][0] + 1/3 * element_v3[token1][0],
                element_v1[token1][1]
            ]

        v_avg.append(v_new)
        probs = [item[0] for item in v_new.values()]


        sample_index = probs.index(max(probs))

        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
                next_token_id3.append(element_v3[item1][1])
            i += 1

    return next_token, v_avg, next_token_id1, next_token_id2, next_token_id3

def pad_list(list_name,pad_id):
    list_len = [len(item) for item in list_name]
    max_len = max(list_len)
    for item in list_name:
        if len(item) < max_len:
            pad = [pad_id] * (max_len - len(item))
            pad.extend(item)
            item[:] = pad

    return list_name

def ensemble_decoding(test):
    fw = open(args.output_file, "a", encoding="utf-8")

    accelerator.wait_for_everyone()
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []


    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    # iter_item = ds_loader

    max_length = args.max_new_tokens
    for questions, answers in iter_item:
        output_ans = []

        inputs1 = tokenizer1(questions, padding=True, return_tensors="pt").to(device1)
        inputs2 = tokenizer2(questions, padding=True, return_tensors="pt").to(device2)
        inputs3 = tokenizer3(questions, padding=True, return_tensors="pt").to(device3)

        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)
        input_ids3 = inputs3['input_ids'].to(device3)

        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)
        attention_mask3 = inputs3['attention_mask'].to(device3)

        input_length = [len(qs) for qs in input_ids1]

        for i in range(max_length):
            start_time = time.time()
            if i == 0: #first step
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           generation_config=generation_config1,
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           generation_config=generation_config2,
                                           )
                outputs3 = model3.generate(input_ids=input_ids3,
                                           attention_mask=attention_mask3,
                                           generation_config=generation_config3,
                                           )

                past_key_values2 = outputs2.past_key_values
                past_key_values3 = outputs3.past_key_values

            else:
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           past_key_values=past_key_values1,
                                           generation_config=generation_config1,
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           past_key_values=past_key_values2,
                                           generation_config=generation_config2,
                                           )
                outputs3 = model3.generate(input_ids=input_ids3,
                                           attention_mask=attention_mask3,
                                           past_key_values=past_key_values3,
                                           generation_config=generation_config3,
                                           )


            past_key_values1 = outputs1.past_key_values

            v1 = get_top_k_tokens(outputs1,tokenizer1,10)
            v2 = get_top_k_tokens(outputs2,tokenizer2,10)
            v3 = get_top_k_tokens(outputs3,tokenizer3, 10)

            v1_sfmx = vocab_softmax(v1)
            v2_sfmx = vocab_softmax(v2)
            v3_sfmx = vocab_softmax(v3)

            vu = get_union_vocab(v1, v2, v3)

            v1_update = update_vocab(v1, vu, tokenizer1, outputs1.logits[0],'llama3.1')
            v2_update = update_vocab(v2, vu, tokenizer2, outputs2.logits[0],'llama3')
            v3_update = update_vocab(v3, vu, tokenizer3, outputs3.logits[0], 'qwen2')

            v1_new, v2_new, v3_new = v1_update, v2_update, v3_update


            next_token, v_avg, next_token_id1, next_token_id2, next_token_id3 = average_and_sample(v1_new,v2_new,v3_new,0.5, tokenizer1)

            end_time = time.time()
            latency = start_time - end_time

            i1,  m1= [], []
            for pred_token_id1, input1_ids, mask1 in zip(next_token_id1,input_ids1,attention_mask1):
                input1_ids = input1_ids.tolist()
                mask1 = mask1.tolist()


                input1_ids.append(pred_token_id1)
                mask1.append(1)

                i1.append(input1_ids)
                m1.append(mask1)


            input_ids1 = torch.tensor(i1).to(device1)
            attention_mask1 = torch.tensor(m1).to(device1)

            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1), padding=True, return_tensors="pt").to(device2)
            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)

            iter_input3 = tokenizer3(tokenizer1.batch_decode(input_ids1), padding=True, return_tensors="pt").to(device3)
            input_ids3 = iter_input3['input_ids'].to(device3)
            attention_mask3 = iter_input3['attention_mask'].to(device3)


        for qs_len, ans in zip(input_length, input_ids1):
            output = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            output = ' '.join(output.split())
            output_ans.append(output)

        ans_num = []
        for gold_ans in answers:
            if 'gsm' in test:
                ans_num.append(float(re.search(r"#### (-?\d+)", gold_ans).group(1)))
            else:
                ans_num.append(gold_ans)
        label_list.extend(ans_num)
        ori_ans_list.extend(answers)

        pred_num = []
        ans_list = []
        for gold_ans in output_ans:
            if 'Question' in gold_ans:
                gold_ans = gold_ans.split('Question:')[0].strip()
            if 'Explanation' in gold_ans:
                gold_ans = gold_ans.split('Explanation')[0].strip()
            ans_list.append(gold_ans)
            if 'gsm' in test.lower():
                pred_num.append(gsm_extract_math_answer(gold_ans))
            else:
                pred_num.append(gold_ans)
            print('==========output========\n', ans_num[-1], "=======", pred_num[-1])
        pred_list.extend(pred_num)
        solution_list.extend(ans_list)
        question_list.extend(questions)



    accelerator.print("======= waiting for everyone ==========")
    accelerator.wait_for_everyone()
    accelerator.print("======= start gather ==========")
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)

    # if accelerator.is_main_process:
    # duplicate_set = set()
    for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label, gather_solution,
                                                  gather_ori_solution):
        fw.write(json.dumps(
            {"question": qs, "original_sln": ori_ans, "pred_solution": solution, "pred": pred, "label": label},
            ensure_ascii=False) + "\n")


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str,
                           default="Your data path")
    arg_parse.add_argument("--prompts", type=str,
                           default="Your prompt path")
    arg_parse.add_argument("--model_path1", type=str, default="Your model path")
    arg_parse.add_argument("--model_path2", type=str, default="Your model path")
    arg_parse.add_argument("--model_path3", type=str, default="Your model path")
    arg_parse.add_argument("--output_file", type=str,
                           default="Your output path")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)

    arg_parse.add_argument("--max_new_tokens", type=int, default=1)

    args = arg_parse.parse_args()


    accelerator = Accelerator()
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model_path1, model_path2, model_path3 = args.model_path1, args.model_path2, args.model_path3

    model1 = AutoModelForCausalLM.from_pretrained(
        model_path1,
        device_map=device1,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()


    model2 = AutoModelForCausalLM.from_pretrained(model_path2, output_attentions=True, device_map=device2,
                                       attn_implementation="flash_attention_2",
                                       torch_dtype=torch.float16).eval()

    model3 = AutoModelForCausalLM.from_pretrained(model_path3, output_attentions=True, device_map=device3,
                                       attn_implementation="flash_attention_2",
                                       torch_dtype=torch.float16
                                                  ).eval()


    tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
    tokenizer3 = AutoTokenizer.from_pretrained(model_path3)

    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer3.pad_token = tokenizer3.eos_token

    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"
    tokenizer3.padding_side = "left"

    generation_config1 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        # output_attentions =True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    generation_config2 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer2.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    generation_config3 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer3.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    # load_data
    test_dataset = load_dataset("json", data_files=args.test_set)['train']
    if 'gsm' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=gsm_collate_fn,
                               num_workers=2)
    if 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=qa_collate_fn,
                               num_workers=2)
    if 'arc' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=arc_collate_fn,
                               num_workers=2)
    if 'piqa' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=piqa_collate_fn,
                               num_workers=2)

    ds_loader = accelerator.prepare_data_loader(ds_loader)

    seed_list = [1987]
    for seed in seed_list:
        print('Start ensembling *********************:')
        ensemble_decoding(args.test_set.lower())
        if 'gsm' in args.test_set.lower():
            gsm_parse_pred_ans(args.output_file)
        if 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
            qa_parse_pred_ans(args.output_file)
        if 'arc' in args.test_set.lower() or 'piqa' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        print('End ensembling =======================:')
