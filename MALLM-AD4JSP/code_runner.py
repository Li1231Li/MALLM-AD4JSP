import json
import subprocess
import sys
import tempfile
import os


RUNNER_TEMPLATE = '''
import json

INSTANCE = {instance_json}

{user_code}

if __name__ == "__main__":
    jobs = INSTANCE["jobs"]
    num_jobs = INSTANCE["num_jobs"]
    num_machines = INSTANCE["num_machines"]
    result = solve(jobs, num_jobs, num_machines)
    print("###RESULT_START###")
    print(json.dumps(result))
    print("###RESULT_END###")
'''


def run_generated_code(code: str, instance_dict: dict, timeout: int = 30):
    script = RUNNER_TEMPLATE.format(
        instance_json=json.dumps(instance_dict, ensure_ascii=False),
        user_code=code,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        tmp_path = f.name

    # A subprocess timeout limits hangs; it is not a security sandbox.
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, None, f"代码执行超时(超过{timeout}秒)，可能存在死循环或性能问题", ""
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if proc.returncode != 0:

        return False, None, proc.stderr[-3000:], proc.stdout

    stdout = proc.stdout
    if "###RESULT_START###" not in stdout or "###RESULT_END###" not in stdout:
        return False, None, "代码未按约定格式输出结果(未找到RESULT标记，请检查是否正确return)", stdout

    payload = stdout.split("###RESULT_START###")[1].split("###RESULT_END###")[0].strip()
    try:
        schedule = json.loads(payload)
    except json.JSONDecodeError as e:
        return False, None, f"solve()返回结果不是合法的JSON可序列化对象: {e}", stdout

    return True, schedule, None, stdout
