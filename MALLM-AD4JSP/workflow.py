from jssp_instance import JSSPInstance, evaluate_schedule
from code_runner import run_generated_code
from llm_client import extract_code_block
from agents import (
    AlgorithmDesigner, AlgorithmProgrammer, AlgorithmOptimizer,
    CodeTester, ProjectManager, INITIAL_PROMPT,
)


class MemoryStream:
    def __init__(self):
        self.events = []

    def add(self, phase: str, role: str, content: str):
        self.events.append({"phase": phase, "role": role, "content": content})

    def as_text(self, max_chars: int = 6000) -> str:
        text = "\n\n".join(
            f"[{e['phase']}][{e['role']}]\n{e['content']}" for e in self.events
        )
        return text[-max_chars:]


class AgentADWorkflow:
    def __init__(self, instance: JSSPInstance,
                 max_design_trials: int = 3,
                 max_optim_iters: int = 10,
                 max_debug_attempts: int = 5,
                 exec_timeout: int = 30,
                 llm_temperature: float = 0.7,
                 verbose: bool = True,
                 use_memory_stream: bool = True,
                 use_code_tester: bool = True):
        self.instance = instance
        self.max_design_trials = max_design_trials
        self.max_optim_iters = max_optim_iters
        self.max_debug_attempts = max_debug_attempts
        self.exec_timeout = exec_timeout
        self.verbose = verbose
        self.use_memory_stream = use_memory_stream
        self.use_code_tester = use_code_tester

        self.ad = AlgorithmDesigner(temperature=llm_temperature)
        self.ap = AlgorithmProgrammer(temperature=llm_temperature)
        self.ao = AlgorithmOptimizer(temperature=llm_temperature)
        self.ct = CodeTester(temperature=llm_temperature)
        self.pm = ProjectManager(temperature=llm_temperature)

        self.memory = MemoryStream()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


    def _transfer_to_code_and_test(self, algo_description: str, phase: str):
        raw_reply = self.ap.act(
            f"请将以下调度算法描述转写为Python代码：\n\n{algo_description}\n\n"
            f"{self._memory_context()}"
            f"目标算例规模参考: num_jobs={self.instance.num_jobs}, "
            f"num_machines={self.instance.num_machines}(但代码需支持任意规模)"
        )
        code = extract_code_block(raw_reply)
        self.memory.add(phase, "AP", f"生成代码：\n{code}")

        instance_dict = self.instance.to_dict()

        for attempt in range(self.max_debug_attempts):
            success, schedule, error, _stdout = run_generated_code(
                code, instance_dict, timeout=self.exec_timeout
            )

            if success:
                try:
                    makespan = evaluate_schedule(self.instance, schedule)
                except ValueError as e:
                    success = False
                    error = f"结果不满足约束: {e}"

            if success:
                self.memory.add(phase, "CT", f"测试通过，makespan={makespan}")
                return code, makespan, True

            self.memory.add(phase, "CT", f"第{attempt + 1}次测试失败: {error}")
            self._log(f"  [调试第{attempt + 1}次] 失败原因: {error}")

            if not self.use_code_tester:
                break
            if attempt == self.max_debug_attempts - 1:
                break

            fix_reply = self.ct.act(
                f"以下代码在测试中出现问题，请修复：\n\n```python\n{code}\n```\n\n"
                f"报错/问题信息：\n{error}\n\n"
                f"{self._memory_context()}"
                f"算例规模参考: num_jobs={self.instance.num_jobs}, "
                f"num_machines={self.instance.num_machines}"
            )
            code = extract_code_block(fix_reply)
            self.memory.add(phase, "CT", f"修复后代码：\n{code}")

        return code, None, False


    def design_phase(self):
        self._log("===== [阶段1/4] 设计阶段 (Design) =====")
        best = {"description": None, "code": None, "makespan": float("inf")}

        for trial in range(self.max_design_trials):
            self._log(f"-- 设计尝试 {trial + 1}/{self.max_design_trials} --")
            desc = self.ad.act(
                f"{INITIAL_PROMPT}\n\n"
                f"这是第{trial + 1}次尝试，请给出一种调度算法"
                f"(可以是启发式优先级规则、构造式算法，也可以是简单的(元)启发式搜索算法)"
                f"的分步骤描述。"
            )
            self.memory.add("设计阶段", "AD", desc)

            code, makespan, ok = self._transfer_to_code_and_test(desc, "设计阶段")
            if ok:
                self._log(f"  -> 可行，makespan = {makespan}")
                if makespan < best["makespan"]:
                    best.update(description=desc, code=code, makespan=makespan)
            else:
                self._log("  -> 该候选算法始终未能通过测试，放弃")

        if best["code"] is None:
            raise RuntimeError(
                "设计阶段未能得到任何可行的初始算法，"
                "请检查LLM返回内容格式，或增大 max_design_trials / max_debug_attempts。"
            )

        self._log(f"[设计阶段完成] 初始算法 makespan = {best['makespan']}")
        return best["description"], best["code"], best["makespan"]


    def optimization_testing_loop(self, init_description, init_code, init_makespan):
        self._log("===== [阶段2-3/4] 优化与测试阶段 (Optimization <-> Testing) =====")

        best_description = init_description
        best_code = init_code
        best_makespan = init_makespan

        for i in range(self.max_optim_iters):
            self._log(f"-- 优化轮次 {i + 1}/{self.max_optim_iters} "
                      f"(当前最优makespan={best_makespan}) --")

            optim_prompt = (
                f"当前算法描述：\n{best_description}\n\n"
                f"当前算法在测试算例(num_jobs={self.instance.num_jobs}, "
                f"num_machines={self.instance.num_machines})上的makespan为: {best_makespan}\n\n"
                f"{self._memory_context()}"
                f"这是第{i + 1}轮优化(共{self.max_optim_iters}轮)。"
                f"请基于以上信息分析瓶颈，并提出具体的改进方案，"
                f"输出改进后的【完整】算法步骤描述。"
            )
            new_description = self.ao.act(optim_prompt)
            self.memory.add("优化阶段", "AO", new_description)

            new_code, new_makespan, ok = self._transfer_to_code_and_test(
                new_description, "测试阶段"
            )

            if ok and new_makespan < best_makespan:
                self.memory.add(
                    "优化阶段", "PM",
                    f"第{i + 1}轮优化被采纳：makespan由 {best_makespan} 降为 {new_makespan}"
                )
                self._log(f"  -> 采纳新算法！makespan: {best_makespan} -> {new_makespan}")
                best_description, best_code, best_makespan = (
                    new_description, new_code, new_makespan
                )
            else:
                reason = "未通过测试" if not ok else f"新makespan({new_makespan})未优于当前最优"
                self.memory.add(
                    "优化阶段", "PM",
                    f"第{i + 1}轮优化未被采纳({reason})，保留原算法"
                )
                self._log(f"  -> 不采纳({reason})，保留原算法")

        return best_description, best_code, best_makespan


    def documentation_phase(self, description, code, makespan):

        self._log("===== [阶段4/4] 文档阶段 (Documentation) =====")

        doc_prompt = (
            f"以下是整个算法设计/优化流程的记忆流记录(可能因篇幅原因只保留了最近部分)：\n\n"
            f"{self.memory.as_text()}\n\n"
            f"最终得到的算法描述：\n{description}\n\n"
            f"最终代码：\n```python\n{code}\n```\n\n"
            f"最终在测试算例(num_jobs={self.instance.num_jobs}, "
            f"num_machines={self.instance.num_machines})上的makespan为: {makespan}\n\n"
            f"请撰写一份简洁的中文Markdown技术文档，总结算法设计思路、"
            f"关键优化点，以及最终效果。"
        )
        doc = self.pm.act(doc_prompt)
        return doc


    def run(self):

        desc0, code0, ms0 = self.design_phase()
        desc, code, makespan = self.optimization_testing_loop(desc0, code0, ms0)
        doc = self.documentation_phase(desc, code, makespan)
        return {
            "description": desc,
            "code": code,
            "makespan": makespan,
            "document": doc,
        }

    def _memory_context(self, max_chars: int = 3000) -> str:
        if not self.use_memory_stream:
            return ""
        memory = self.memory.as_text(max_chars=max_chars)
        if not memory.strip():
            return ""
        return f"以下是此前设计、测试和采纳记录，可用于避免重复错误并指导改进：\n{memory}\n\n"
