import torch
import random
import json
import re
import os
import math
from model_utils import LM

class Node:
    def __init__(self, question, partial_answer, correct_answer):
        self.question = question
        self.partial_answer = partial_answer
        self.correct_answer = correct_answer
        self.mc_score = None
        self.visits = 0
        self.rollouts = []
        self.visited_rollouts = []

    def add_rollout(self, result):
        self.rollouts.append(result)
        self.visited_rollouts.append(False)

    def increment_visits(self):
        self.visits += 1

# xxw 
def check_correctness(expected_answer, generated_response):
    """
    用于检查生成的响应generated_response 中，是否包含预期答案expected_answer。
    具体来说，它从生成的响应中提取最后一句话，然后检查预期答案是否存在于这句话中。
    """
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_response.strip()
    )
    last_sentence = sentences[-1] if sentences else ''
    return expected_answer.strip() in last_sentence.strip()

def perform_rollouts(node, model: LM, num_rollouts=None):
    correctness_flags = []
    results = model.generate(node.question, node.partial_answer, num_rollouts)
    for result in results:
        node.add_rollout(result)
        is_correct = check_correctness(node.correct_answer, result)
        correctness_flags.append(int(is_correct))
    return node.rollouts, correctness_flags

# 用于计算一个节点的蒙特卡洛评分（MC评分）
# 该评分表示节点的展开序列在多次模拟中产生正确答案的比例
def calculate_mc_score(node):
    # check_correctness用于判断某个模拟结果 r 是否正确。
    # 使用列表生成式和sum函数来计算correct_count，即节点所有模拟结果（node.rollouts）中与正确答案匹配的次数
    correct_count = sum(
        check_correctness(node.correct_answer, r) for r in node.rollouts
    )
    # 返回这些模拟结果中与正确答案匹配的比例，即正确的次数与总模拟次数的比值。
    return correct_count / len(node.rollouts) if node.rollouts else 0

# xxw core
# 用于在给定的节点列表中选择最优的节点
# 该选择基于某种评估函数的计算结果，特别是对于未访问过的选项（或展开序列）进行评估
def select_best_node(nodes):
    best_node = None
    best_rollout_idx = -1
    highest_qu_value = -1
    for node in nodes:
        # 获取节点的蒙特卡洛评分mc_score
        mc_score = (
            node.mc_score if node.mc_score is not None else calculate_mc_score(node)
        )
        # 跳过评分为0或1的节点（可能意味着这些节点不适合作为备选节点）。
        if mc_score in [0, 1]:
            continue
        # 评估未访问的展开序列（rollouts表示节点的已知可能展开序列）
        for idx, rollout in enumerate(node.rollouts):
            if node.visited_rollouts[idx]:
                continue
            # 计算两种分值
            q_val = compute_q_value(rollout, mc_score)
            u_val = compute_u_value(node, nodes)
            # 分值求和
            qu_value = q_val + u_val
            # 记录最大的qu value
            if qu_value > highest_qu_value:
                highest_qu_value = qu_value
                best_node = node
                best_rollout_idx = idx
    if best_rollout_idx != -1 and best_node is not None:
        best_node.visited_rollouts[best_rollout_idx] = True
        # 返回：最优节点best_node、最优展开序列best_node.rollouts[best_rollout_idx]、对应的QU值
        return best_node, best_node.rollouts[best_rollout_idx], highest_qu_value
    else:
        return None, None, None

# 用于将输入的文本字符串 text 分为大致相等的两部分
def split_text_middle(text):
    text = text.strip()
    mid_idx = len(text) // 2
    if text[mid_idx] != ' ':
        left_space = text.rfind(' ', 0, mid_idx)
        right_space = text.find(' ', mid_idx)
        if left_space == -1:
            split_idx = right_space
        elif right_space == -1:
            split_idx = left_space
        else:
            split_idx = (
                left_space if (mid_idx - left_space) <= (right_space - mid_idx) else right_space
            )
    else:
        split_idx = mid_idx
    part1 = text[:split_idx].strip()
    part2 = text[split_idx:].strip()
    return part1, part2

# 分析一个节点的展开序列，尝试找到合适的子节点，将错误定位在文本展开中
def locate_error(node, rollout, model):
    """
    node: 当前正在分析的节点对象，包含问题、已有的文本和正确答案等信息。
    rollout: 当前节点的文本展开序列（即要分析的文本部分）。
    model: 用于进行展开模拟和评估的模型，可能涉及语言生成任务。
    """
    current_span = rollout # 当前处理的文本
    previous_text = ""     # 之前处理过的文本
    nodes_to_expand = []   # 待进一步扩展的节点
    leaf_nodes = []        # 叶子节点
    while True:
        if len(current_span.split()) < 2:
            break
        # 用「split_text_middle」将「current_span」分为左右两部分left_part和right_part
        left_part, right_part = split_text_middle(current_span)
        print("----")
        print(" Left:", left_part)
        print(" Right:", right_part)
        new_node = Node(
            node.question, previous_text + left_part, node.correct_answer
        )
        # 用perform_rollouts对new_node进行模拟展开
        perform_rollouts(new_node, model)
        # 计算new_node的蒙特卡洛评分mc_score并赋值。
        mc_score = calculate_mc_score(new_node)
        # 赋值
        new_node.mc_score = mc_score
        if mc_score == 1:
            break
        elif mc_score > 0:
            current_span = right_part
            previous_text += left_part
            nodes_to_expand.append(new_node)
        else:
            current_span = left_part
            leaf_nodes.append(new_node)
    print("----")
    return nodes_to_expand, leaf_nodes

# 计算节点的质量值（Q值）
# 表示基于当前展开序列rollout_text和蒙特卡洛评分mc_score的质量评估
def compute_q_value(rollout_text, mc_score, alpha=0.5, beta=0.9, max_length=500):
    """
    rollout_text: 展开的文本序列。
    mc_score: 节点的蒙特卡洛评分。
    alpha: 控制与蒙特卡洛评分的关系，默认值为0.5。
    beta: 控制与文本长度的关系，默认值为0.9。
    max_length: 用于尺度化文本长度的最大值，默认值为500。
    """
    # 与 mc_score 相关，alpha ** (1 - mc_score)，mc_score 越接近1，part1 越小。
    part1 = alpha ** (1 - mc_score)
    # 与 rollout_text 的长度成反比，beta ** (len(rollout_text) / max_length)，卷展文本越长，part2 越小
    part2 = beta ** (len(rollout_text) / max_length)
    # 通过将 part1 与 part2 相乘获得最终的 Q值。这种设计倾向于选择评分高且文本较短的展开序列。
    return part1 * part2

# 计算节点的探索偏移值（U值）
# 鼓励对尚未被充分探索的节点进行更多的访问。
def compute_u_value(node, all_nodes, exploration_param=0.125):
    """
    node: 当前节点。
    all_nodes: 所有节点列表，用于计算总访问量。
    exploration_param: 探索参数，决定探索的力度，默认值为0.125。
    """
    total_visits = sum(n.visits for n in all_nodes)
    numerator = math.sqrt(total_visits)
    denominator = 1 + node.visits
    return exploration_param * (numerator / denominator)

# xxw core
# 基于mcts方法自动构造prm训练数据
def process_annotations(question, nodes, model: LM, filename='nodes_data.json', max_iterations=100):
    """
    用于基于蒙特卡洛树搜索（MCTS）方法来处理注释，并自动构造概率图模型（PRM）训练数据
    question: 当前处理的问题。
    nodes: 初始节点列表。
    model: 一个LM模型（通常是语言模型），用于帮助定位错误。
    filename: 存储节点数据的文件名，默认是 nodes_data.json。
    max_iterations: 最大迭代次数的限制，默认是100。
    """

    print("++++++")
    iteration = 0
    leaf_nodes = [] # 用于存储找到的叶子节点
    while True:
        # 最优节点node、最优展开序列rollout、对应的QU值
        node, rollout, max_qu = select_best_node(nodes)
        # 将节点的信息转换为字典（包括问题、部分答案和蒙特卡洛评分），并追加到文件 filename 中。
        if node is not None and node.partial_answer != '':
            new_entry = {
                "question": question,
                "partial_answer": node.partial_answer,
                "mc_score": node.mc_score,
            }
            append_to_json(filename, new_entry)
            # 判断是否已达到最大迭代次数，如果是则中断循环
            iteration += 1
            if iteration > max_iterations:
                break
        if node is None:
            break
        print()
        print("[Selected Node]")
        print(node)
        print("  Rollout:", rollout, " || QU Value:", max_qu)
        node.increment_visits()
        # 调用「locate_error」查找错误，获取「扩展的节点expanded_nodes」和「新的叶子节点leaves」
        expanded_nodes, leaves = locate_error(node, rollout, model)
        # 检查是否有扩展的节点，如果没有则继续下一个循环。
        if not expanded_nodes:
            continue
        # 将所有非空且部分答案非空的新节点加入到节点列表 nodes 中。
        nodes.extend(
            n for n in expanded_nodes if n is not None and n.partial_answer != ''
        )
        # 把新的叶子节点加入到leaf_nodes列表中。
        leaf_nodes.extend(leaves)
    # 把每个节点的信息记录到 JSON 文件中。
    for leaf_node in leaf_nodes:
        new_entry = {
            "question": question,
            "partial_answer": leaf_node.partial_answer,
            "mc_score": leaf_node.mc_score,
        }
        append_to_json(filename, new_entry)
    print("++++++")

# Utils
def append_to_json(filename, data_entry):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        data = []
    data.append(data_entry)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data appended to {filename}")
