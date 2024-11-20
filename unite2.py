from tqdm import tqdm
import numpy as np
import re
import seaborn as sns
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
    """
    计算输入数组的softmax值。

    参数:
        x (numpy.ndarray): 输入的数值数组，通常表示一个向量或矩阵。

    返回:
        numpy.ndarray: 输入数组的softmax值，形状与输入相同。
    """
    # 为了数值稳定性，将输入数组减去其最大值
    x = x - np.max(x)

    # 计算指数函数值
    exp_x = np.exp(x)

    # 计算指数函数值的总和
    sum_exp_x = np.sum(exp_x)

    # 计算softmax值，即每个指数值除以总和
    softmax_x = exp_x / sum_exp_x

    return softmax_x


def qa_collate_fn(batch):
    """
    该函数用于处理一批问答数据，通常用于TriviaQA或Natural Questions (NQ) 数据集。

    参数:
        batch (list): 一个包含多个样本的列表，每个样本是一个字典，字典中应包含 "question" 和 "answer" 两个键。
                      - "question": 表示问题的字符串。
                      - "answer": 表示答案的字符串或数据结构。

    返回值:
        tuple: 包含两个列表的元组：
               - questions: 处理后的问题列表，每个问题都附加了固定的提示模板。
               - answers: 原始答案列表，未经过任何修改。
    """

    # 初始化问题和答案的存储列表
    questions, answers = [], []

    # 遍历批次中的每个样本，提取问题并生成带提示的问题文本，同时保存对应的答案
    for b in batch:
        ques = b["question"]
        prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers



def gsm_collate_fn(batch):
    """
    该函数用于处理GSM8K数据集的批次数据，生成适合模型输入的问题和答案对。

    参数:
        batch (list): 一个批次的数据，其中每个元素是一个字典，包含以下键：
                      - "question": 表示问题的字符串。
                      - "answer": 表示答案的字符串。

    返回值:
        tuple: 包含两个列表的元组：
               - questions: 经过格式化处理后的问题列表，每个问题都添加了固定的提示前缀。
               - answers: 原始答案字符串组成的列表。
    """
    questions, answers = [], []

    # 遍历批次中的每个样本，提取问题和答案，并对问题进行格式化处理
    for b in batch:
        ques = b["question"]
        prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nLet\'s think step by step\n'
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers



def count_words_split(text):
    """
    统计输入文本中单词的数量，通过空格分割字符串实现。

    参数:
        text (str): 输入的字符串文本。

    返回:
        int: 文本中单词的数量。单词以空格分隔，连续的空格视为分隔符。
    """
    # 将输入文本按空格分割为单词列表
    words = text.split()

    # 返回单词列表的长度，即单词数量
    return len(words)


def get_top_k_tokens(outputs, tokenizer, k=10):
    """
    获取模型输出中概率最高的前k个token及其相关信息。

    参数:
        outputs: 模型的输出对象，包含logits属性，表示每个token的概率分布。
        tokenizer: 用于将token ID转换为实际token字符串的分词器对象。
        k: 需要提取的最高概率token的数量，默认为10。

    返回值:
        v1: 一个列表，其中每个元素是一个字典。字典的键是处理后的token字符串，
            值是一个列表，包含该token的概率和对应的ID。
    """

    # 提取logits并将其作为概率分布（未归一化）
    logits = outputs.logits[0]
    probs = logits

    # 获取概率最高的前k个token的索引
    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()

    # 提取前k个token对应的实际概率值
    top_k_probs = []
    for idx, prob in zip(top_k_indices, probs):
        prob_item = []
        for i in idx:
            prob_item.append(prob[i])
        top_k_probs.append(prob_item)

    # 将前k个token的索引转换为实际的token字符串
    top_k_tokens = []
    for indices in top_k_indices:
        token_item = []
        for idx in indices:
            token_item.append(tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True))
        top_k_tokens.append(token_item)

    # 构建最终的结果列表，每个元素是一个字典，包含token、概率和ID的信息
    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append(
            {token.replace('▁','Ġ').replace('<0x0A>','/n').replace('Ċ','/n'): [prob, int(id)] for token, prob, id in zip(token, prob, id)})

    return v1


def get_union_vocab(v1, v2):
    """
    获取两个字典列表中每对字典的词汇并集。

    参数:
        v1 (list): 包含字典的列表，每个字典的键为词汇。
        v2 (list): 包含字典的列表，每个字典的键为词汇，与v1一一对应。

    返回:
        list: 每对字典的词汇并集组成的列表，每个元素是一个包含词汇的列表。
    """
    # 提取两个字典列表中每对字典的词汇并集
    unique_tokens = []
    for v1_tokens, v2_tokens in zip(v1, v2):
        unique_tokens.append(list(set(v1_tokens.keys()) | set(v2_tokens.keys())))

    return unique_tokens


"""
更新词汇表函数，用于根据输入的词汇和模型特性动态调整词汇表。

参数:
    v1 (dict): 初始词汇表，存储词汇及其对应的logit值和token ID。
    vu (list): 需要更新的词汇列表，每个元素是一个词汇集合。
    tokenizer: 分词器对象，用于将词汇转换为token ID或进行分词操作。
    logits (list): 模型输出的logits列表，每个元素对应一个词汇集合的logits。
    model_name (str): 模型名称，用于区分不同模型的特殊处理逻辑。

返回值:
    dict: 更新后的词汇表，经过softmax处理后的结果。
"""

def update_vocab(v1, vu, tokenizer, logits, model_name):
    # 遍历需要更新的词汇、初始词汇和logits，逐个处理每个词汇集合
    for vu_token, v1_token, logit_ele in zip(vu, v1, logits):
        v1_token_ids = []
        # 提取初始词汇表中所有token ID，用于后续去重检查
        for item in v1_token.values():
            v1_token_ids.append(item[1])

        # 遍历需要更新的词汇集合，处理每个词汇
        for token in vu_token:
            if token not in v1_token.keys():
                # 根据模型名称处理特殊token（如替换字符）
                if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                    token = token.replace('Ġ', '▁')

                if token != '':
                    # 将词汇转换为token ID，并获取对应的logit值
                    subtoken_id = tokenizer.convert_tokens_to_ids(token)
                    if subtoken_id != 0 and subtoken_id is not None:
                        logit = logit_ele[subtoken_id]
                    else:
                        # 如果词汇无法直接转换为token ID，则进行分词处理
                        subtokens = tokenizer.tokenize(token)
                        for token_id in tokenizer.convert_tokens_to_ids(subtokens):
                            # 根据模型名称跳过特定的无效token ID
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
                    # 处理空词汇的情况，根据模型名称设置默认logit和token ID
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
                    if 'glm' in model_name:
                        logit = logit_ele[128]
                        subtoken_id = 128

                # 根据模型名称处理特殊token，并更新词汇表
                if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                    v1_token[token.replace('▁', 'Ġ')] = [logit, subtoken_id]
                else:
                    if subtoken_id not in v1_token_ids:
                        v1_token[token] = [logit, subtoken_id]
                        v1_token_ids.append(subtoken_id)
                    else:
                        v1_token[token] = [0, subtoken_id]

    # 对更新后的词汇表进行softmax处理并返回
    v1_new = vocab_softmax(v1)
    return v1_new



def vocab_softmax(v1):
    """
    对输入的词汇概率分布进行 softmax 归一化处理。

    参数:
        v1 (list): 一个列表，其中每个元素是一个字典。字典的键是词汇（token），
                   值是一个包含两个元素的列表，第一个元素是一个数值（用于 softmax 计算），
                   第二个元素是与该词汇相关的标识符（ids）。

    返回值:
        list: 一个新的列表，结构与输入相同，但每个词汇的概率值已经过 softmax 归一化处理。
              每个字典的值仍然是一个包含两个元素的列表，第一个元素是归一化后的概率值，
              第二个元素保持不变。
    """
    v1_new = []
    for element in v1:
        # 初始化一个新的字典，用于存储当前元素的处理结果
        ele = {}
        ele_values = list(element.values())
        ele_values0, ele_values1 = [], []

        # 分离每个词汇的概率值和标识符
        for item in ele_values:
            ele_values0.append(item[0])  # 提取概率值
            ele_values1.append(item[1])  # 提取标识符

        # 对概率值进行 softmax 归一化处理
        ele_values0 = torch.softmax(torch.tensor(ele_values0), dim=0)




def drop_token(v1, v2, t):
    """
    根据给定的阈值 t，过滤两个列表中的字典元素。

    参数:
        v1 (list): 一个包含字典的列表，每个字典的键对应一个列表，列表的第一个元素用于比较。
        v2 (list): 一个与 v1 结构相同的列表，与 v1 的元素一一对应。
        t (float): 阈值，用于过滤 v1 中字典的键值对。

    返回:
        tuple: 包含两个列表的元组：
            - 第一个列表是过滤后的 v1。
            - 第二个列表是过滤后的 v2，与 v1 的过滤结果一一对应。
    """
    # 初始化两个空列表，用于存储过滤后的结果
    v1_new, v2_new = [], []

    # 遍历 v1 和 v2 中对应的元素
    for v1_element, v2_element in zip(v1, v2):
        v1_, v2_ = {}, {}

        # 遍历当前字典的所有键，根据阈值 t 进行过滤
        for key in v1_element.keys():
            if v1_element[key][0] > t:
                # 如果满足条件，将对应的键值对保留到新的字典中
                v1_[key] = v1_element[key]
                v2_[key] = v2_element[key]

        # 将过滤后的字典添加到结果列表中
        v1_new.append(v1_)
        v2_new.append(v2_)

    return v1_new, v2_new



def average_and_sample(v1, v2, lamda, tokenizer):
    """
    计算两个输入序列的加权平均值，并根据概率分布采样下一个token。

    参数:
        v1 (list of dict): 第一个输入序列，每个元素是一个字典，键为token，值为包含概率和token ID的列表。
        v2 (list of dict): 第二个输入序列，结构与v1相同。
        lamda (float): 加权平均的权重参数，用于计算v1和v2的加权平均值。
        tokenizer: 用于将token ID转换为实际token的对象，需实现convert_ids_to_tokens方法。

    返回:
        next_token (list): 采样得到的下一个token列表。
        v_avg (list of dict): 加权平均后的概率分布列表。
        next_token_id1 (list): 从v1中采样得到的token ID列表。
        next_token_id2 (list): 从v2中采样得到的token ID列表。
    """
    # 初始化返回值列表
    next_token, v_avg, next_token_id1, next_token_id2 = [], [], [], []

    # 遍历v1和v2中的每一对元素
    for element_v1, element_v2 in zip(v1, v2):
        # 确保v1和v2中对应元素的长度一致
        assert len(element_v1) == len(element_v2)

        # 计算加权平均值并存储在v_new中
        v_new = {}
        for token1 in element_v1:
            v_new[token1] = [
                lamda * element_v1[token1][0] + (1 - lamda) * element_v2[token1][0],
                element_v1[token1][1]
            ]
        v_avg.append(v_new)

        # 提取加权平均后的概率值
        probs = []
        for item in v_new.values():
            probs.append(item[0])

        # 找到概率最大的token索引
        sample_index = probs.index(max(probs))

        # 根据概率最大的索引采样token及其ID
        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
            i += 1

    return next_token, v_avg, next_token_id1, next_token_id2


def pad_list(list_name, pad_id):
    """
    对列表中的每个子列表进行填充，使其长度与最长的子列表一致。

    参数:
        list_name (list): 一个包含多个子列表的列表，每个子列表可以有不同的长度。
        pad_id (int): 用于填充的值，填充时会在子列表的左侧添加该值。

    返回:
        list: 修改后的列表，其中每个子列表的长度都等于原列表中最长子列表的长度。
    """
    # 计算每个子列表的长度，并找到最大长度
    list_len = [len(item) for item in list_name]
    max_len = max(list_len)

    # 遍历每个子列表，对长度不足的子列表进行填充
    for item in list_name:
        if len(item) < max_len:
            # 创建填充部分并将其与原子列表合并
            pad = [pad_id] * (max_len - len(item))
            pad.extend(item)
            item[:] = pad

    return list_name


def ensemble_decoding(test):
    """
    该函数实现了一个基于双模型集成解码的推理过程，用于生成问题的答案并评估结果。

    参数:
        test (str): 测试集的标识符，用于区分不同的测试任务（如"gsm"任务）。

    返回值:
        无返回值。函数的主要输出是将推理结果写入指定的文件中。
    """

    # 打开输出文件，准备写入推理结果
    fw = open(args.output_file, "w", encoding="utf-8")

    # 等待所有进程同步
    accelerator.wait_for_everyone()

    # 初始化存储推理结果的列表
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []

    # 根据是否为主进程决定是否使用进度条
    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    # 获取最大生成长度
    max_length = args.max_new_tokens

    # 遍历数据集中的每个批次
    for questions, answers in iter_item:
        output_ans = []

        # 对问题进行编码，分别输入到两个模型中
        inputs1 = tokenizer1(questions, padding=True, return_tensors="pt").to(device1)
        inputs2 = tokenizer2(questions, padding=True, return_tensors="pt").to(device2)
        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)

        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)

        # 计算输入问题的长度
        input_length = [len(qs) for qs in input_ids1]

        distribution1, distribution2 = [], []

        # 开始逐步生成答案
        for i in range(max_length):
            if i == 0:  # 第一步生成
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           generation_config=generation_config1,
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           generation_config=generation_config2,
                                           )
            else:  # 后续步骤生成
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           past_key_values=past_key_values1,
                                           generation_config=generation_config1,
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           generation_config=generation_config2,
                                           )

            # 更新第一个模型的past_key_values
            past_key_values1 = outputs1.past_key_values

            # 提取模型输出的概率分布信息
            logits1 = torch.max(torch.softmax(torch.topk(outputs1.logits[0][0], 10).values, dim=0)).item()
            logits2 = torch.max(torch.softmax(torch.topk(outputs2.logits[0][0], 10).values, dim=0)).item()

            distribution1.append(logits1)
            distribution2.append(logits2)

            # 获取两个模型的top-k词汇表并计算联合词汇表
            v1 = get_top_k_tokens(outputs1, tokenizer1, 10)
            v2 = get_top_k_tokens(outputs2, tokenizer2, 10)

            v1_sfmx = vocab_softmax(v1)
            v2_sfmx = vocab_softmax(v2)

            vu = get_union_vocab(v1, v2)

            # 更新词汇表以适应联合词汇表
            v1_update = update_vocab(v1, vu, tokenizer1, outputs1.logits[0], 'qwen2')
            v2_update = update_vocab(v2, vu, tokenizer2, outputs2.logits[0], 'llama3')

            v1_new, v2_new = v1_update, v2_update

            # 对更新后的词汇表进行平均采样，生成下一个token
            next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(v1_new, v2_new, 0.5, tokenizer1)

            # 更新输入和注意力掩码以包含新生成的token
            i1, i2, m1, m2 = [], [], [], []
            for pred_token_id1, pred_token_id2, input1_ids, input2_ids, mask1, mask2 in zip(next_token_id1, next_token_id2, input_ids1, input_ids2, attention_mask1, attention_mask2):
                input1_ids = input1_ids.tolist()
                mask1 = mask1.tolist()
                input1_ids.append(pred_token_id1)
                mask1.append(1)
                i1.append(input1_ids)
                m1.append(mask1)

            input_ids1 = torch.tensor(i1).to(device1)
            attention_mask1 = torch.tensor(m1).to(device1)

            # 将更新后的输入重新编码为第二个模型的输入格式
            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1), padding=True, return_tensors="pt").to(device2)

            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)

        # 解码生成的答案并存储
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
            print(gold_ans)
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

    # 等待所有进程完成
    accelerator.print("======= waiting for everyone ==========")
    accelerator.wait_for_everyone()
    accelerator.print("======= start gather ==========")

    # 收集所有进程的结果
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)

    # 将收集到的结果写入文件
    for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label, gather_solution, gather_ori_solution):
        fw.write(json.dumps(
            {"question": qs, "original_sln": ori_ans, "pred_solution": solution, "pred": pred, "label": label},
            ensure_ascii=False) + "\n")

    """
    主函数入口，用于执行模型集成推理流程。

    参数通过命令行传入，支持以下参数：
    --test_set: 测试数据集路径，默认值为 "Your data path"。
    --prompts: 提示模板文件路径，默认值为 "Your prompt path"。
    --model_path1: 第一个模型的路径，默认值为 "Your model path"。
    --model_path2: 第二个模型的路径，默认值为 "Your model path"。
    --output_file: 输出文件路径，默认值为 "Your output file path"。
    --per_device_batch_size: 每个设备的批量大小，默认值为 1。
    --max_new_tokens: 最大生成 token 数，默认值为 10（不同数据集可能需要调整）。

    流程概述：
    1. 初始化加速器和设备。
    2. 加载提示模板、模型、分词器及生成配置。
    3. 根据测试数据集类型加载数据并创建数据加载器。
    4. 执行模型集成推理，并根据数据集类型解析预测结果。
    """
if __name__ == "__main__":

    # 初始化参数解析器，定义命令行参数
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str, default="/home/lonelydoll/project/unite/datasets/GSM/test.cleand.jsonl")
    arg_parse.add_argument("--prompts", type=str, default="/home/lonelydoll/project/unite/datasets/GSM/gsm_prompt.txt")
    arg_parse.add_argument("--model_path1", type=str, default="/home/lonelydoll/models/shakechen/Llama-2-7b-chat-hf")
    arg_parse.add_argument("--model_path2", type=str, default="/home/lonelydoll/models/deepseek-ai/deepseek-llm-7b-chat")
    arg_parse.add_argument("--output_file", type=str, default="/home/lonelydoll/project/unite/output_file_llama7b_deepseek7b.jsonl")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)
    arg_parse.add_argument("--max_new_tokens", type=int, default=10)  # 不同数据集的最大 token 数可能不同

    args = arg_parse.parse_args()

    # 初始化加速器，用于分布式计算
    accelerator = Accelerator()

    # 加载设备信息和提示模板
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    prompt_complex = open(args.prompts, "r", encoding="utf-8").read()

    # 加载模型、分词器及生成配置
    model_path1, model_path2 = args.model_path1, args.model_path2

    model1 = AutoModelForCausalLM.from_pretrained(
        model_path1, output_attentions=True, device_map=device1,
        attn_implementation="flash_attention_2", torch_dtype=torch.float16
    ).eval()

    model2 = AutoModelForCausalLM.from_pretrained(
        model_path2, output_attentions=True, device_map=device2,
        attn_implementation="flash_attention_2", torch_dtype=torch.float16
    ).eval()

    tokenizer1, tokenizer2 = AutoTokenizer.from_pretrained(model_path1), AutoTokenizer.from_pretrained(model_path2)

    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token

    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"

    generation_config1 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
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

    # 加载测试数据集并创建数据加载器
    test_dataset = load_dataset("json", data_files=args.test_set)['train']
    if 'gsm' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=gsm_collate_fn, num_workers=2)
    if 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=qa_collate_fn, num_workers=2)
    if 'arc' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=arc_collate_fn, num_workers=2)
    if 'piqa' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=piqa_collate_fn, num_workers=2)

    ds_loader = accelerator.prepare_data_loader(ds_loader)

    # 设置随机种子并执行模型集成推理
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

