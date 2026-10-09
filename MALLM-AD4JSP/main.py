from jssp_instance import JSSPInstance
from workflow import AgentADWorkflow


def main():
    instance = JSSPInstance.random_instance(num_jobs=6, num_machines=5, seed=42)


    print(f"算例规模: {instance.num_jobs} jobs x {instance.num_machines} machines")


    workflow = AgentADWorkflow(
        instance=instance,
        max_design_trials=3,
        max_optim_iters=10,
        max_debug_attempts=5,
        exec_timeout=30,
        llm_temperature=0.7,
        verbose=True,
    )

    result = workflow.run()


    print("\n===================== 最终结果 =====================")
    print(f"最优 makespan: {result['makespan']}")

    out_path = "agentad_jssp_result.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# AgentAD × JSSP 求解结果\n\n")
        f.write(f"- 算例规模: {instance.num_jobs} jobs x {instance.num_machines} machines\n")
        f.write(f"- 最终makespan: {result['makespan']}\n\n")
        f.write(result["document"])
        f.write("\n\n## 最终算法代码\n\n```python\n")
        f.write(result["code"])
        f.write("\n```\n")

    print(f"完整技术文档与代码已保存到: {out_path}")


if __name__ == "__main__":
    main()
