import json
import re

# GSM
def gsm_parse_pred_ans(filename):
    """
    解析并评估预测答案的准确性。

    该函数读取包含预测结果的文件，对比预测答案与实际标签，计算预测的准确率。
    它只考虑文件中第一次出现的解决方案，忽略重复的解决方案。

    参数:
    filename (str): 包含预测结果的文件名。

    返回:
    无。但会打印出问题数量、正确预测的数量和准确率。
    """
    # 初始化总数和正确数计数器
    total, correct = 0, 0
    # 存储黄金答案以避免重复计数
    gold_ans = []
    # 打开文件读取数据
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            # 解析每一行的JSON对象
            jo = json.loads(line.strip())
            # 检查当前解决方案是否已记录
            if jo["original_sln"] not in gold_ans:
                # 如果预测答案与标签匹配，增加正确计数
                correct += jo["pred"] == jo["label"]
                # 增加总数计数器
                total += 1
                # 将新解决方案添加到黄金答案列表
                gold_ans.append(jo["original_sln"])
            else:
                # 如果解决方案已存在，跳过当前循环迭代
                continue
    # 打印问题总数、正确数量和准确率
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))


# ARC/PIQA/MMLU
def arc_parse_pred_ans(filename):
    """
    计算模型预测答案的准确率。

    该函数读取包含模型预测结果的文件，统计预测正确的题目数量和总题数，
    并计算准确率。每个问题只统计一次，即使文件中包含多个预测结果。

    参数:
    filename (str): 包含模型预测结果的文件名。

    返回:
    float: 预测答案的准确率。
    """
    # 初始化总题数和正确题数为0
    total, correct = 0, 0
    # 初始化黄金答案列表和问题列表为空
    gold_ans = []
    qs = []

    # 打开文件读取预测结果
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            # 解析每一行的JSON对象
            jo = json.loads(line.strip())
            # 如果当前问题不在已记录的问题列表中
            if jo["question"] not in qs:
                # 比较预测答案和标签答案，相同则正确题数加一
                correct += jo["pred"].strip() == jo["label"].strip()
                # 总题数加一
                total += 1
                # 将当前问题添加到问题列表中
                qs.append(jo["question"])
            else:
                # 如果问题已经在列表中，则跳过当前循环
                continue

    # 打印总题数、正确题数和准确率
    if total > 0:
        print(f'num_q {total} correct {correct} ratio {correct / total:.4f}')
    else:
        print('Error: No questions available for calculation')
    # 返回准确率
    return float(correct / total)


#TriviaQA NQ
def qa_parse_pred_ans(filename):
    """
    解析给定文件中的问题回答预测结果，并计算准确率。

    该函数读取文件中的每一行，每一行都是一个包含预测答案和标准答案的JSON对象。
    它通过比较预测答案和标准答案来计算预测的准确率，并打印出问题总数、正确数和准确率。

    参数:
    filename: str - 输入文件名，文件应包含预测答案和标准答案。
    """
    # 初始化问题总数和正确回答数
    total, correct = 0, 0

    # 打开文件读取数据
    with open(filename, "r", encoding="utf-8") as fr:
        # 遍历文件中的每一行
        for line in fr:
            # 解析JSON对象
            jo = json.loads(line.strip())
            # 遍历标准答案
            for gold in jo["label"]:
                # 如果预测答案在标准答案中，则计数正确回答
                if jo['pred'][:-1].strip() in gold:
                    correct += 1
                    break
            # 更新问题总数
            total += 1

    # 打印问题总数、正确回答数和准确率
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))


