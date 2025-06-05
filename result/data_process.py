import json
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

def calculate_metrics(output_file, test_set_type):
    # 解析输出文件
    with open(output_file, 'r') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)

    # 根据任务类型计算指标
    if 'gsm' in test_set_type.lower():
        # 数值型任务指标
        df['pred'] = pd.to_numeric(df['pred'], errors='coerce')
        df['label'] = pd.to_numeric(df['label'], errors='coerce')

        metrics = {
            "准确率": accuracy_score(df['label'], df['pred'].round()),
            "平均绝对误差": mean_absolute_error(df['label'], df['pred']),
            "完全匹配率": (df['pred'] == df['label']).mean()
        }
    else:
        # 文本型任务指标
        metrics = {
            "完全匹配率": (df['pred'] == df['original_sln']).mean(),
            "部分匹配率": df.apply(lambda x: x['original_sln'] in x['pred'], axis=1).mean()
        }
    return metrics

# 添加main函数处理命令行参数
if __name__ == "__main__":
    # 直接设置参数替代命令行解析
    output_file = "/home/wujie/project/unite/output_file_llama7b_deepseek7b.jsonl"
    test_set_type = "gsm"

    # 调用封装的计算函数
    metrics = calculate_metrics(output_file, test_set_type)
    print("计算完成的指标:", metrics)
