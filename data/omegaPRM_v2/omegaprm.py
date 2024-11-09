import heapq
import math
import random
import re
import json
from typing import List, Tuple, Dict, Any, Optional
import itertools
from llm_utils import LLMService

# Helper function to separate reasoning steps
def separate_steps(steps: List[str], mode: str = 'join') -> Any:
    delimiter = "\n\n"
    if mode == 'join':
        if not isinstance(steps, list):
            raise TypeError("For 'join' mode, 'steps' must be a list of strings.")
        return delimiter.join(steps)
    elif mode == 'split':
        if not isinstance(steps, str):
            raise TypeError("For 'split' mode, 'steps' must be a string.")
        return steps.split(delimiter)
    else:
        raise ValueError("Mode should be either 'join' or 'split'.")

# Helper function to check correctness of a generated response
def check_correctness(generated_response: str, expected_answer: str) -> bool:
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_response.strip()
    )
    last_sentence = sentences[-1] if sentences else ''
    return expected_answer.strip() in last_sentence.strip()

# 
class LanguageModel:
    def __init__(self, model_name="/root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct",
                 device="cuda", max_new_tokens=512, temperature=0.7, top_k=30, top_p=0.9):
        """
        Initialize the LanguageModel with parameters for the LLM service.

        Parameters:
        model_name：模型的路径或名称，默认值为 /root/.cache/modelscope/hub/Qwen/Qwen2___5-Math-7B-Instruct。
        device：计算设备，默认值为 cuda（GPU），也可以是 cpu。
        max_new_tokens：生成的响应中最大新生成的 tokens 数量，默认值为 512。
        temperature：采样温度，用于控制生成的多样性，默认值为 0.7。
        top_k：Top-K 采样，用于控制生成的多样性，默认值为 30。
        top_p：Top-P 采样（Nucleus 采样），用于控制生成的多样性，默认值为 0.9。
        """
        self.llm_service = LLMService(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        self.default_prompt = (
            "Please complete the answer for the question based on the given steps without generating existing steps again, "
            "and separate your following steps using \\n\\n.\n\n"
        )
        self.llm_service.start_service()

    # 生成一个rollout
    def generate_rollout(self, state_prefix: str) -> str:
        """
        state_prefix：当前解决方案的前缀。
        prompt：将默认提示与当前解决方案前缀合并，形成完整的提示。
        response：调用 LLMService 的 generate_response 方法生成响应。
        返回值：返回生成响应中的 content 部分，假设响应格式包含多个 ['role'] 条目，并且助手的响应在 response[1]['content'] 中。
        """
        prompt = self.default_prompt + state_prefix
        response = self.llm_service.generate_response(prompt)
        return response[1]['content']  # Assuming the response format has ['role'] entries and 'assistant' response

    # 更新prompt
    def update_prompt(self, new_prompt: str):
        """
        更新提示模板
        """
        self.default_prompt = new_prompt

    # 评估正确性
    def evaluate_correctness(self, response: str, expected_answer: str) -> bool:
        """
        response：完整的生成响应。
        expected_answer：期望的答案。
        返回值：返回一个布尔值，表示期望答案是否在生成响应的最后部分。这里假设 check_correctness 是一个外部定义的函数，用于比较生成的响应和期望的答案。
        """
        return check_correctness(response, expected_answer)


# Define the State class
class State:
    def __init__(self, solution_prefix: str, parent: Optional['State'] = None):
        self.solution_prefix = solution_prefix  # Solution prefix as a single string
        self.parent = parent  # Reference to the parent state
        self.N = 0  # Visit count (number of times selected)
        self.total_rollouts = 0  # Total number of rollouts generated from this state
        self.correct_rollouts = 0  # Number of correct rollouts
        self.MC: Optional[float] = None  # Monte Carlo estimation (c/k)
        self.Q: Dict[str, float] = {}  # Q(s, r): estimated value for each rollout
        self.R: List[str] = []  # Set of all rollouts from this state
        self.incorrect_rollouts: List[str] = []  # List of incorrect rollouts

    def add_rollout(self, rollout: str):
        self.R.append(rollout)

    def add_incorrect_rollout(self, rollout: str):
        if rollout not in self.incorrect_rollouts:
            self.incorrect_rollouts.append(rollout)

    def get_full_solution(self) -> str:
        # Return the complete solution from the root to this state
        if self.parent:
            return self.parent.get_full_solution() + '\n\n' + self.solution_prefix
        else:
            return self.solution_prefix

# xxw tree结构
class SearchTree:
    def __init__(self):
        self.root: Optional[State] = None
        self.nodes: List[State] = []  # List of all states

    def add_state(self, state: State):
        self.nodes.append(state)

# xxw 优先队列
class CandidatePool:
    """
    用于管理和储存「状态state」和「对应的rollout字符串」
    """
    def __init__(self):
        self.heap: List[Tuple[float, int]] = []                 # 一个堆（优先队列），存储元组 (-priority, unique_id)。使用负数优先级是为了实现最大堆
        self.entry_finder: Dict[int, Tuple[float, int]] = {}    # 一个字典，用于快速查找堆中的条目，键为 unique_id，值为堆中的元组 (-priority, unique_id)
        self.counter = itertools.count()                        # 一个生成唯一 ID 的计数器，使用 itertools.count() 实现。
        self.id_to_rollout: Dict[int, Tuple[State, str]] = {}   # 一个字典，映射 unique_id 到 (state, rollout)，方便快速查找状态和 rollout。
        self.latest_id_per_rollout: Dict[Tuple[int, str], int] = {}  # 一个字典，映射 (state_id, rollout) 到最新的 unique_id，用于识别和更新现有的 rollout。

    # 添加或更新方法, 用于添加一个新的 rollout 或者更新已存在的 rollout 的优先级。
    def add_or_update(self, state: State, rollout: str, priority: float):
        state_id = id(state)  # 获取状态对象的唯一标识符
        rollout_key = (state_id, rollout)

        # 检查 rollout 是否已存在于池中
        if rollout_key in self.latest_id_per_rollout:
            # 之前的 unique_id 仍然存在；它现在已过期
            old_unique_id = self.latest_id_per_rollout[rollout_key]
            # 通过从 entry_finder 中删除旧条目来标记其为无效
            if old_unique_id in self.entry_finder:
                del self.entry_finder[old_unique_id]
                del self.id_to_rollout[old_unique_id]

        # 为更新的 rollout 分配一个新的 unique_id
        unique_id = next(self.counter)
        self.latest_id_per_rollout[rollout_key] = unique_id

        # 将新条目添加到堆和映射中
        heapq.heappush(self.heap, (-priority, unique_id))  # 使用负数优先级实现最大堆
        self.entry_finder[unique_id] = (-priority, unique_id)
        self.id_to_rollout[unique_id] = (state, rollout)

    # 弹出优先级最高的 rollout
    def pop(self) -> Tuple[Optional[State], Optional[str]]:
        while self.heap:
            neg_priority, unique_id = heapq.heappop(self.heap)
            # 检查此 unique_id 是否仍然有效
            if unique_id in self.entry_finder:
                # 有效条目
                state, rollout = self.id_to_rollout.pop(unique_id)
                del self.entry_finder[unique_id]
                # 从 latest_id_per_rollout 中删除
                state_id = id(state)
                rollout_key = (state_id, rollout)
                if self.latest_id_per_rollout.get(rollout_key) == unique_id:
                    del self.latest_id_per_rollout[rollout_key]
                return state, rollout
            # 否则，过期条目；跳过
        return None, None
    # 判空  
    def is_empty(self) -> bool:
        return not self.entry_finder

# Define the OmegaPRM algorithm
class OmegaPRM:
    def __init__(self, LM: LanguageModel,  c_puct: float, alpha: float, beta: float, L: int, k: int, N: int,
                 rollout_budget: int):
        """
        Initialize the OmegaPRM algorithm.

        Parameters:
        - LM (LanguageModel): The language model instance.
        - expected_answer (str): The expected answer for correctness checking.
        - c_puct (float): Exploration constant.
        - alpha (float): Weight for MC(s).
        - beta (float): Length penalty.
        - L (int): Maximum solution length.
        - k (int): Number of rollouts for Monte Carlo estimation.
        - N (int): Maximum search count.
        """
        self.LM = LM                    # 一个语言模型实例，用于生成和评估相关的文本操作。
        self.expected_answer = None     
        self.c_puct = c_puct            # 探索常数：用于在搜索过程中控制探索与利用的平衡。
        self.alpha = alpha              # 用于蒙特卡洛估计的权重
        self.beta = beta                # 长度惩罚系数：与生成的文本长度相关的计算有关。
        self.L = L                      # 最大解决方案长度：用于限制生成的答案长度等相关计算。
        self.k = k                      # 蒙特卡洛估计的滚动次数：决定了进行多少次随机模拟来估计某些值。
        self.N = N                      # 最大搜索计数：限制了整个搜索过程的迭代次数。
        self.rollout_budget = rollout_budget # 滚动预算，限制总的滚动操作次数。

        self.T = SearchTree()           # 定义了一个树结构，有add操作
        self.C = CandidatePool()        # 一个优先队列，有add(update)操作，有pop操作

        self.n = 0  # 记录当前的搜索计数
        self.total_rollouts = 0  # 记录总的滚动操作次数。
        self.collected_data = [] # 空列表，用于收集搜索过程中的相关数据，可能是不同状态下的解决方案前缀及其对应的蒙特卡洛估计值等

    def reset(self):
        """
        用于重置类实例的内部状态变量
        以便为新的一轮运行做准备。
        """
        self.expected_answer = None
        self.T = SearchTree()  
        self.C = CandidatePool()  
        self.n = 0
        self.total_rollouts = 0
        self.collected_data = [] 

    def run(self, question: str, answer: str) -> List:
        """
        执行OmegaPRM算法的主要流程
        它接受一个问题字符串和对应的答案字符串作为参数，并返回一个列表
        """
        self.reset()

        print(f"Running OmegaPRM for question: '{question}'\n")
        # Initialization
        initial_state = State(solution_prefix=question, parent=None) # 创建一个初始状态对象initial_state，其solution_prefix为传入的问题字符串，父状态为None
        self.expected_answer = answer # 将传入的答案字符串赋值给self.expected_answer，用于后续的正确性评估。
        self.T.root = initial_state   # 将初始状态设置为搜索树self.T的根节点，并将其添加到搜索树中。
        self.T.add_state(initial_state) # 将搜索计数self.n重置为 0。
        self.n = 0

        # 对初始状态进行蒙特卡洛估计
        # 通过调用self.monte_carlo_estimation方法来生成多个滚动并评估其正确性，以获取初始状态的相关估计值。
        self.monte_carlo_estimation(initial_state)

        # 进入主循环，只要满足以下三个条件就会继续循环：
            # 1. 当前搜索计数self.n小于最大搜索计数self.N
            # 2. 总的滚动操作次数self.total_rollouts小于滚动预算self.rollout_budget
            # 3. 候选池self.C不为空，即还有候选的状态和滚动可供选择和处理
        while self.n < self.N and self.total_rollouts < self.rollout_budget and not self.C.is_empty():
            # 首先进入选择阶段
            # 获取一个状态和rollout
            selected_state, selected_rollout = self.selection_phase()
            if selected_state is None or selected_rollout is None:
                print("No more candidates to explore. Terminating search.\n")
                break

            # Log the selected rollout
            state_id = self.T.nodes.index(selected_state)
            print(f"Selection Phase: Selected rollout from State ID {state_id}")
            print(f"Selected Rollout:\n{selected_rollout}\n")

            # 然后进入扩展阶段
            # 执行二分查找以找到不正确的步骤
            self.expansion_phase_binary_search(selected_state, selected_rollout)

            # 更新状态State所有不正确的rollout的统计信息和候选
            self.maintenance_phase(selected_state)

            # Increment search count
            self.n += 1

        self.collect_solution_prefixes()
        return self.collected_data
    
    
    # xxw 执行蒙特卡洛估计
    def monte_carlo_estimation(self, state: State):
        """
        接受的State就是问题prompt
        """
        c = 0  # 正确 rollout 的计数

        # 存储不正确和正确的 rollout。
        incorrect_rollouts = []
        correct_rollouts = []

        # 循环生成 k 个 rollout
        for i in range(self.k):
            self.total_rollouts += 1
            # 使用 self.LM.generate_rollout(state.solution_prefix) 生成一个 rollout。
            rollout = self.LM.generate_rollout(state.solution_prefix)
            # 将生成的 rollout 添加到状态对象中。
            state.add_rollout(rollout)

            # 将state.solution_prefix和生成的rollout组合成完整的「当前步骤思考」
            full_solution = (state.solution_prefix + '\n\n' + rollout).strip() if state.solution_prefix else rollout
            # 使用self.LM.evaluate_correctness(full_solution, self.expected_answer) 评估「当前步骤思考」的正确性
            is_correct = self.LM.evaluate_correctness(full_solution, self.expected_answer)

            print(f"Rollout {i + 1} Correctness: {'Correct' if is_correct else 'Incorrect'}\n")

            # 根据is_correct 的结果，更新 c 和 correct_rollouts 或 incorrect_rollouts 列表。
            if is_correct:
                c += 1
                correct_rollouts.append(rollout)
            else:
                incorrect_rollouts.append(rollout)
                state.add_incorrect_rollout(rollout)  # Track incorrect rollouts

        # 更新状态的蒙特卡洛估计
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = state.correct_rollouts / state.total_rollouts if state.total_rollouts > 0 else 0

        """
        如果 state.MC == 1.0，所有正确的 rollout 都将被添加到树中作为新的状态。
        如果 state.MC == 0.0，状态被认为是不正确的，不再进一步操作。
        如果 0 < state.MC < 1.0，将正确的 rollout 添加到树中，将不正确的 rollout 添加到候选池中，并计算其优先级。
        """
        if state.MC == 1.0:
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
        elif state.MC == 0.0:
            return
        else:
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)

            for rollout in incorrect_rollouts:

                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)

    # Q评估函数，长度分
    def compute_Q(self, state: State, rollout: str) -> float:
        # Count words in the rollout
        word_count = len(rollout.split())
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta ** length_penalty)
        return Q_value

    # U评估函数，acc分数
    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s

    # QU函数合并在一起考虑
    def compute_selection_score(self, state: State, rollout: str) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score
    
    # 选择阶段
    # 调用pop，从候选池中选择一个状态和对应的rollout
    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_solution_prefix = (parent_state.solution_prefix + '\n\n' + rollout).strip() if parent_state.solution_prefix else rollout
        new_state = State(solution_prefix=new_solution_prefix, parent=parent_state)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        state_id_new = len(self.T.nodes) - 1
        # print(f"Added Correct Rollout as State ID {state_id_new}:")
        # print(f"Solution Prefix:\n{new_solution_prefix}\n")

    # 执行二分查找以找到不正确的步骤
    def expansion_phase_binary_search(self, parent_state: State, rollout: str):
        """
        该方法在扩展阶段（Expansion phase）中将一个rollout作为新的状态添加
        使用二分查找（Binary Search）来高效地找到rollout中不正确的步骤。这个方法主要用于处理从父状态选择的不正确rollout。
        parent_state：一个 State 对象，表示从哪个状态选择的 rollout。
        rollout：一个字符串，表示选择的不正确的 rollout。
        """
        # 调用一个外部函数separate_steps，将 rollout 字符串分离成单独的步骤列表 steps
        steps = separate_steps(rollout, mode='split')

        # 二分法找到不正确的步骤
        self.binary_search_incorrect_step(parent_state, steps, 0, len(steps) - 1)

    # 二分法
    def binary_search_incorrect_step(self, s_ast: State, steps: List[str], left: int, right: int):
        """
        通过递归的方式执行二分查找，以找到 rollout 中所有不正确的步骤。
        具体来说，它会在给定的步骤列表中通过二分查找逐步缩小范围，找到不正确的步骤，并基于 Monte Carlo 方法评估这些步骤的正确性
        """
        # 终止条件
        if left > right:
            return
        # 计算中间索引
        mid = (left + right) // 2

        # 创建prefix_solution
        # 如果 s_ast.solution_prefix 存在，将其与 new_steps 组合成一个新的前缀解决方案 prefix_solution。
        # 否则，直接使用 new_steps 生成前缀解决方案。
        new_steps = steps[:mid + 1]
        prefix_solution = (s_ast.solution_prefix + '\n\n' + separate_steps(new_steps, mode='join')).strip() if s_ast.solution_prefix else separate_steps(new_steps, mode='join').strip()
        
        # 创建新状态：基于prefix_solution创建一个新的State对象s_new
        s_new = State(solution_prefix=prefix_solution, parent=s_ast)

        # 将新状态 s_new 添加到状态树 self.T 中。
        # state_id_new：获取新状态的 ID，它是状态树中节点的索引。
        self.T.add_state(s_new)
        state_id_new = len(self.T.nodes) - 1

        # 执行 Monte Carlo 估计
        self.monte_carlo_estimation(s_new)

        # 判断Monte Carlo估计结果
        # 如果 s_new.MC 为 0，表示当前步骤组合 new_steps 中存在不正确的步骤。
        if s_new.MC == 0:
            print(f"State ID {state_id_new} has MC == 0. Incorrect step found. Searching earlier steps.\n")
            # 调用 binary_search_incorrect_step 方法，继续在左半部分（left 到 mid - 1）搜索不正确的步骤
            self.binary_search_incorrect_step(s_ast, steps, left, mid - 1)
        # 如果 s_new.MC 不为 0，表示当前步骤组合 new_steps 是正确的
        else:
            print(f"State ID {state_id_new} has MC == {s_new.MC:.2f}. Steps up to Step {mid + 1} are correct. Searching later steps.\n")
            # 调用 binary_search_incorrect_step 方法，继续在右半部分（mid + 1 到 right）搜索不正确的步骤。
            self.binary_search_incorrect_step(s_new, steps, mid + 1, right)
    
    # xxw 维护阶段
    def maintenance_phase(self, state: State):
        """
        维护阶段（Maintenance phase），更新状态 State 所有不正确的 rollout 的统计信息和候选池
        具体来说，该方法会遍历状态的所有不正确的 rollout，并重新计算这些 rollout 的优先级，然后更新候选池中的相应 rollout。
        接受一个 State 对象作为参数。
        state：表示需要更新其不正确 rollout 的状态。
        """

        # state.incorrect_rollouts：一个包含所有不正确 rollout 的列表。
        # 使用 for 循环遍历 state.incorrect_rollouts 中的每一个 rollout。
        for rollout in state.incorrect_rollouts:
            # 计算不正确rollout的优先级
            priority = self.compute_selection_score(state, rollout)
            # 更新候选池中的rollout优先级
            self.C.add_or_update(state, rollout, priority)

    # 该方法用于从搜索树中收集所有状态的前缀解决方案及其对应的 Monte Carlo (MC) 值
    def collect_solution_prefixes(self):
        """
        该方法用于从搜索树中收集所有状态的前缀解决方案及其对应的 Monte Carlo (MC) 值
        具体来说，它会遍历状态树 self.T 中的所有节点，提取每个节点的 solution_prefix 和 MC 值，并将这些信息存储在一个列表 self.collected_data 中
        """
        for node in self.T.nodes:
            solution_prefix = node.solution_prefix
            mc_value = node.MC
            self.collected_data.append({"solution_prefix": solution_prefix, "mc_value": mc_value})


# Example usage
if __name__ == "__main__":
    # Initialize the Language Model
    LM = LanguageModel(

        device="cuda",
        max_new_tokens=2048
    )

    # Define the question and expected answer
    question = "我在一个袋子里有5个编号为1到5的弹珠。假设我随机取出两个不同的弹珠，取出的弹珠上的数字之和的期望值是多少？"
    expected_answer = "6"

    # Initialize OmegaPRM with parameters
    omega_prm = OmegaPRM(
        LM=LM,
        c_puct=0.125,
        alpha=0.5,
        beta=0.9,
        L=500,
        k=16,
        N=10,
        rollout_budget=100,
    )

    # Run the OmegaPRM algorithm
    collected_data = omega_prm.run(question, expected_answer)

    # Save the collected solutions to a JSON file
    with open("collected_solutions2.json", "w") as f:
        json.dump(collected_data, f, indent=4)

    # Print the collected states and their Monte Carlo estimations
    print("\nFinal Collected States:")
    for idx, state in enumerate(search_tree.nodes):
        print(f"State {idx}:")
        print(f"Solution Prefix:\n{state.solution_prefix}")
        print(f"MC: {state.MC}, N: {state.N}, Total Rollouts: {state.total_rollouts}, Correct Rollouts: {state.correct_rollouts}\n")

    # Print collected solutions with MC values
    print("\nCollected Solution Prefixes and MC Values:")
    for solution in omega_prm.collected_data:
        print(f"Solution Prefix: {solution['solution_prefix']}")
        print(f"MC Value: {solution['mc_value']}\n")
