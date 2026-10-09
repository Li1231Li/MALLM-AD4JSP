import random


class JSSPInstance:
    def __init__(self, jobs, num_jobs, num_machines, name="instance"):
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.name = name

    def to_dict(self):

        return {
            "name": self.name,
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "jobs": self.jobs,
        }

    @staticmethod
    def random_instance(num_jobs=6, num_machines=5, min_time=1, max_time=20, seed=None):
        rng = random.Random(seed)
        jobs = []
        for _ in range(num_jobs):
            machine_order = list(range(num_machines))
            rng.shuffle(machine_order)
            job = [(m, rng.randint(min_time, max_time)) for m in machine_order]
            jobs.append(job)
        return JSSPInstance(jobs, num_jobs, num_machines,
                             name=f"rand_{num_jobs}x{num_machines}")

    @staticmethod
    def from_taillard_text(text: str, name="taillard"):
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        num_jobs, num_machines = map(int, lines[0].split())
        jobs = []
        for i in range(1, num_jobs + 1):
            nums = list(map(int, lines[i].split()))
            job = [(nums[k], nums[k + 1]) for k in range(0, len(nums), 2)]
            jobs.append(job)
        return JSSPInstance(jobs, num_jobs, num_machines, name=name)


def evaluate_schedule(instance: JSSPInstance, schedule):
    jobs = instance.jobs
    num_jobs = instance.num_jobs
    num_machines = instance.num_machines

    if not isinstance(schedule, (list, tuple)):
        raise ValueError("算法返回结果必须是作业分派顺序的list，例如 [0,1,0,2,...]")

    total_ops = sum(len(j) for j in jobs)
    if len(schedule) != total_ops:
        raise ValueError(f"分派序列长度({len(schedule)})与总工序数({total_ops})不一致")

    next_op_idx = [0] * num_jobs
    machine_free_time = [0.0] * num_machines
    job_free_time = [0.0] * num_jobs
    op_count = [len(j) for j in jobs]

    # Decode each dispatch decision at the earliest feasible start.
    for job_id in schedule:
        if not isinstance(job_id, int) or job_id < 0 or job_id >= num_jobs:
            raise ValueError(f"非法的job_id: {job_id}")
        idx = next_op_idx[job_id]
        if idx >= op_count[job_id]:
            raise ValueError(f"job {job_id} 的工序已全部处理完毕，分派序列不合法(重复调度)")

        machine_id, proc_time = jobs[job_id][idx]
        start = max(machine_free_time[machine_id], job_free_time[job_id])
        end = start + proc_time

        machine_free_time[machine_id] = end
        job_free_time[job_id] = end
        next_op_idx[job_id] += 1

    if any(next_op_idx[j] != op_count[j] for j in range(num_jobs)):
        raise ValueError("并非所有工序都被调度，分派序列不完整")

    makespan = max(job_free_time)
    return makespan
