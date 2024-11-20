def piqa_collate_fn(batch):
    """
    该函数用于处理一批数据，生成适合模型输入的格式。

    参数:
        batch (list): 一个包含多个样本的列表，每个样本是一个字典，字典中包含以下键：
                      - "question": 问题字符串；
                      - "A": 选项A的描述字符串；
                      - "B": 选项B的描述字符串；
                      - "answer": 正确答案（通常为"A"或"B"）。

    返回值:
        tuple: 包含两个列表的元组：
               - questions: 处理后的提示问题列表，每个问题包含选项和回答提示；
               - answers: 对应的正确答案列表。
    """
    questions, answers = [], []

    # 遍历批次中的每个样本，构造提示问题并提取答案
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        # 构造提示问题，要求用户通过回复A或B来回答问题
        prompt_q = f'Answer the question by replying A or B.\nQuestion: {ques}\nA: {A}\nB: {B}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers



def arc_collate_fn(batch):
    """
    该函数用于处理一批数据，将其转换为问题和答案的格式，适用于ARC（AI2 Reasoning Challenge）数据集。

    参数:
        batch (list): 一个包含多个样本的列表，每个样本是一个字典，字典中包含以下键：
                      - "question": 问题文本 (str)
                      - "A", "B", "C", "D": 四个选项的文本 (str)
                      - "answer": 正确答案的选项 (str)

    返回值:
        tuple: 包含两个元素的元组：
               - questions (list): 格式化后的问题列表，每个问题包含选项提示。
               - answers (list): 对应的正确答案列表。
    """
    questions, answers = [], []

    # 遍历批次中的每个样本，构造问题和答案
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        C = b["C"]
        D = b["D"]

        # 构造带有选项提示的问题文本
        prompt_q = f'Answer the question by replying A, B, C or D.\nQuestion: {ques}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers

