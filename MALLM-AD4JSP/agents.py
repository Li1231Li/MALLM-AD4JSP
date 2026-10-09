from llm_client import LLMTools


PROBLEM_DESCRIPTION = """
# 问题描述
这是一个经典的车间调度问题(Job Shop Scheduling Problem, JSSP)，静态、单目标版本。
- 有 num_jobs 个作业(job)和 num_machines 台机器(machine)。
- 每个作业包含一个固定的工序(operation)序列，每道工序必须在指定的机器上加工指定的时间，
  且同一作业内的工序必须严格按给定顺序完成(不能跳过或颠倒，这是工艺路线约束)。
- 每台机器同一时刻只能加工一道工序，且加工过程不可中断(非抢占式调度)。
- 所有作业在时刻0即可开始加工(这是"静态"问题的含义，不存在作业中途随机到达)。
- 优化目标：最小化 makespan（即所有作业全部加工完成的最大完工时间），这是单一目标(single-objective)。
- JSSP 是NP-Hard组合优化问题。
"""

DATASET_EXPLANATION = """
# 数据集说明
输入数据 jobs 是一个列表：jobs[i] 表示第 i 个作业的工序序列，
其中每个元素是二元组 (machine_id, processing_time)，按加工顺序排列。
num_jobs 表示作业(job)数量，num_machines 表示机器(machine)数量。
"""

GLOBAL_TASK = """
# 你需要做的事(总体任务)
你需要为上述JSSP问题设计/优化/实现一个调度算法，其目标是给出一个"分派顺序"，
使得按照该顺序解码得到的 makespan 尽可能小。
"""

TASK_REQUIREMENTS = """
# 对你的要求
1. 算法必须能够处理任意规模的JSSP算例(不同的 num_jobs 和 num_machines)，不要硬编码规模。
2. 最终代码必须是可以直接运行的Python代码，且必须定义如下签名的函数：
       def solve(jobs, num_jobs, num_machines):
           ...
           return schedule
   其中 schedule 是一个 list，表示按"决策/分派"顺序排列的作业编号(job_id)序列，
   例如 [0, 1, 0, 2, 1, ...]；每个 job_id 出现的次数必须等于该作业的工序数
   (因为每次出现代表"该作业的下一道待加工工序被分派执行一次")。
   注意：你不需要自己计算每道工序的开始/结束时间，外部的解码器会根据工艺约束
   和机器占用情况，按照你给出的分派顺序自动计算 makespan。
3. 代码中不允许使用需要联网、读写外部文件、等待用户输入等操作。
4. Use only brief English comments for non-obvious steps.
5. 只允许使用Python标准库(如 random、heapq、itertools 等)，如需数值计算可使用 numpy。
"""

ROBUSTNESS_REQUIREMENTS = """
# 鲁棒构造要求
不要只返回单一 SPT 或 LPT 序列。为了避免单一规则在不同 JSSP 实例上失效，算法应在 `solve` 内部构造 FIFO、最早完成时间（ECT）、最长剩余工作量（MWKR）和短工时优先（SPT）四种确定性的候选序列。FIFO 必须按轮次交替加入每个未完成作业的下一道工序，不能将一个作业的所有工序连续放入序列。ECT 每次应从未完成作业中选择按当前 `machine_free_time` 与 `job_free_time` 计算后最早完成下一道工序的作业。

算法必须在 `solve` 内部实现一个与外部评价器一致的 makespan 计算函数：对每个候选序列依次维护 `next_op_idx`、`machine_free_time` 和 `job_free_time`，将当前作业的下一道工序安排在作业与机器均可用的最早时刻，并以所有作业完成时间的最大值作为 makespan。返回候选集中 makespan 最小的序列。这样候选规则的选择只依赖当前输入实例，而不依赖外部文件或隐藏数据。

可以在最佳候选序列上执行有限次数的交换或插入邻域改进，但必须使用固定随机种子（如 `rng = random.Random(0)`）以保证重复可复现，并保留原始最佳候选作为回退解；任何改进只有在内部评分严格降低 makespan 时才允许采纳。不得在没有内部比较的情况下直接返回单一 SPT、LPT 或随机规则。

每个候选序列都必须恰好包含 `total_ops = sum(len(job) for job in jobs)` 个元素，并且每个作业编号出现次数恰好等于其工序数。构造序列时应执行 `for _ in range(total_ops)`，每轮只从尚有未完成工序的作业中选择一个作业；返回前必须显式检查 `len(schedule) == total_ops`。不得使用会在部分作业完成后提前终止、从而返回不完整序列的循环条件。
"""

INITIAL_PROMPT = (
    PROBLEM_DESCRIPTION
    + DATASET_EXPLANATION
    + GLOBAL_TASK
    + TASK_REQUIREMENTS
    + ROBUSTNESS_REQUIREMENTS
)


class BaseAgent:
    role_name = "BaseAgent"
    system_prompt = ""

    def __init__(self, llm: LLMTools = None, temperature: float = 0.7):
        self.llm = llm or LLMTools()
        self.temperature = temperature

    def act(self, user_prompt: str, history: list = None) -> str:

        return self.llm.chat(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            history=history,
            temperature=self.temperature,
        )


class AlgorithmDesigner(BaseAgent):
    role_name = "AD"
    system_prompt = (
        "你是AgentAD多智能体系统中的【算法设计师(Algorithm Designer)】。\n"
        "我非常熟悉组合优化与调度算法，包括优先级派工规则(SPT/LPT/MWKR等)、"
        "构造式启发式算法、遗传算法、模拟退火、禁忌搜索、变邻域搜索等元启发式算法。\n"
        "我的职责是：根据用户提供的车间调度问题(JSSP)场景描述和数据说明，"
        "提出一个具体的、可执行的、分步骤描述的调度算法框架，用于最小化makespan。\n"
        "请用清晰的分步骤自然语言(不要输出代码)描述算法逻辑，"
        "以便算法程序员能据此直接转写为Python代码。"
    )


class AlgorithmProgrammer(BaseAgent):
    role_name = "AP"
    system_prompt = (
        "你是AgentAD多智能体系统中的【算法程序员(Algorithm Programmer)】。\n"
        "我的职责是：把上游给出的自然语言算法步骤描述，准确地转写成可以直接运行的"
        "Python代码。\n"
        "硬性要求：\n"
        "1. 必须定义函数 def solve(jobs, num_jobs, num_machines): 并返回一个"
        "分派顺序列表(job_id组成的list)。\n"
        "2. 不允许调用 input()、读写文件、网络请求等操作。\n"
        "3. Use only brief English comments where the logic is non-obvious.\n"
        "4. 只输出一个```python代码块```，不要输出多余的解释文字。"
        "\n5. 对于调度算法，优先实现多个候选派工序列及其内部 makespan 评分，再返回候选中的最优序列。"
        "不要把单一 SPT/LPT 规则当作默认答案；任何局部搜索必须保留可行候选作为回退。"
        "\n6. 设 total_ops=sum(len(job) for job in jobs)。每个候选和最终返回的 schedule 必须长度恰好为 total_ops；"
        "请使用固定次数循环构造序列并在 return 前校验该条件。"
    )


class AlgorithmOptimizer(BaseAgent):
    role_name = "AO"
    system_prompt = (
        "你是AgentAD多智能体系统中的【算法优化师(Algorithm Optimizer)】。\n"
        "我擅长分析现有调度算法的瓶颈，并提出具体的改进策略，例如：\n"
        "- 改进优先级排序规则(结合剩余加工时间、机器负载、关键路径长度等启发式信息)；\n"
        "- 在初始解基础上引入局部搜索/邻域结构(如交换、插入)进行改进；\n"
        "- 引入随机重启、多次采样取最优解等策略；\n"
        "- 调整算法内部的参数配置。\n"
        "我会收到当前算法的自然语言描述、以及它在测试算例上的makespan表现，"
        "我需要分析当前算法的不足，并给出具体的、可操作的改进步骤描述。\n"
        "请只输出改进后的【完整】算法步骤描述(自然语言，不是代码)，"
        "描述要完整到可以让算法程序员直接据此重新实现，不要只写差异部分。"
        "改进时必须保留至少一个可行的构造性候选作为回退，并要求程序员用内部 makespan 评分验证"
        "新策略优于该候选后才返回。"
    )


class CodeTester(BaseAgent):
    role_name = "CT"
    system_prompt = (
        "你是AgentAD多智能体系统中的【代码测试员(Code Tester)】。\n"
        "我的职责是：当代码执行报错或结果不满足约束时，分析报错信息/异常原因，"
        "并给出精确的修复方案。\n"
        "我会收到：出错的代码 和 报错/异常信息。\n"
        "请仔细定位错误原因(常见问题包括：solve函数签名不对、返回值格式不对、"
        "对同一作业重复调度、下标越界、死循环等)，"
        "并直接输出修复后的【完整】可执行Python代码(用```python代码块```包裹)。\n"
        "代码要求：必须定义 solve(jobs, num_jobs, num_machines) 函数，"
        "并返回分派顺序列表；仅使用必要的英文注释；不要输出多余的解释文字。"
    )


class ProjectManager(BaseAgent):
    role_name = "PM"
    system_prompt = (
        "你是AgentAD多智能体系统中的【项目经理(Project Manager)】。\n"
        "我负责统筹整个工作流程：记录每一步的关键信息(记忆流)，"
        "评估每次算法迭代的效果，并在流程结束后根据整个流程的记录撰写一份简洁的"
        "中文技术文档，总结最终得到的调度算法的设计思路、关键改进点和最终效果(makespan)。\n"
        "生成文档时请使用Markdown格式，包含：算法思路、关键优化点、"
        "迭代优化过程摘要、最终结果。"
    )
