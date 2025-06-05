import re
def gsm_extract_math_answer(pred_str):
    """
    从预测字符串中提取数学答案。

    参数:
        pred_str (str): 包含预测结果的字符串，可能包含数学答案或特定标记（如 'boxed' 或 'the answer is'）。

    返回值:
        float: 提取的数学答案。如果无法提取有效数字，则返回 float("nan")。
    """
    # try:
    #     # 根据特定标记提取答案部分
    #     if 'boxed' in pred_str:
    #         ans = pred_str.split('boxed')[-1]
    #     elif 'the answer is ' in pred_str:
    #         ans = pred_str.split('the answer is ')[-1].strip()
    #     elif 'The answer is ' in pred_str:
    #         ans = pred_str.split('The answer is ')[-1].strip()
    #     else:
    #         ans = pred_str
    #
    #     # 使用正则表达式匹配数字（包括整数、小数和负数）
    #     pattern = r'-?\d*[\.,]?\d+'
    #     pred = re.findall(pattern, ans)
    #
    #     # 如果匹配到数字，提取最后一个数字并转换为浮点数
    #     if len(pred) >= 1:
    #         pred = float(pred[-1].replace(',', ''))
    #     else:
    #         pred = float("nan")
    try:
        # 优先匹配两种标准格式
        patterns = [
            r'the answer is (\d+)',  # 新增：直接匹配答案声明
            r'\\boxed{(-?\d+[\.,]?\d*)}',  # LaTeX格式
            r'####\s*(-?\d+[\.,]?\d*)'  # GSM标准格式
        ]

        # 逆序搜索以获取最后出现的答案
        for pattern in reversed(patterns):
            matches = re.findall(pattern, pred_str)
            if matches:
                # 提取最后一个匹配项（应对多答案情况）
                last_match = matches[-1].replace(',', '')
                return float(last_match)

        # 兼容旧版非标准写法
        fallback_pattern = r'-?\d+[\.,]?\d*'
        matches = re.findall(fallback_pattern, pred_str)
        if matches:
            return float(matches[-1].replace(',', ''))

        return float("nan")

    except Exception:
        # 捕获异常并打印错误信息，返回 NaN 表示解析失败
        print(f"Cannot parse the resulting num in predicted solution {pred_str}.\n")
        pred = float("nan")

    return pred

