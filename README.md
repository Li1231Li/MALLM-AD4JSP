# AgentAD-style multi-agent workflow for static JSSP

This repository adapts the design, optimization, testing, and documentation stages of an AgentAD-style workflow to the static, single-objective job-shop scheduling problem (JSSP). The objective is to minimize makespan. An LLM proposes and revises scheduling algorithms; generated Python code is executed on a JSSP instance and evaluated against the same dispatch-sequence interface.

## Files

| File | Purpose |
| --- | --- |
| `agents.py` | Prompts and roles for the designer, programmer, optimizer, tester, and project manager |
| `workflow.py` | Multi-agent workflow and memory stream |
| `llm_client.py` | Volcano Ark chat-completions client and Markdown code extraction |
| `jssp_instance.py` | JSSP instances, random generator, text parser, and makespan evaluator |
| `code_runner.py` | Subprocess execution of LLM-generated code with a timeout |
| `main.py` | Example entry point |

The prompts remain in Chinese to preserve the original experiment's behavior. They request only brief English comments in generated code. Source comments are limited to the few English notes needed to explain the scheduler and execution boundary.

## Scheduling interface

Generated algorithms must define `solve(jobs, num_jobs, num_machines)` and return a list of job IDs. Each occurrence of a job ID dispatches that job's next operation. Every job must appear exactly once per operation. `evaluate_schedule()` decodes this sequence using job precedence and machine availability, then returns the makespan.

## Configuration

Use Python 3.10 or newer. The client uses `requests` if it is installed and otherwise falls back to Python's standard library.

1. Set `ARK_API_KEY` to your own Volcano Ark API key. Never commit the key.
2. Optionally set `ARK_MODEL_ID` and `ARK_API_ENDPOINT`. The defaults match the original project configuration and may need updating for your account.
3. Run `python main.py` from this directory.

PowerShell:

```powershell
$env:ARK_API_KEY = "your-own-key"
python main.py
```

Bash:

```bash
export ARK_API_KEY="your-own-key"
python main.py
```

The example generates a 6-job, 5-machine instance and writes `agentad_jssp_result.md`. Each workflow run makes paid API calls. The number of design trials, optimization iterations, and repair attempts can be adjusted in `main.py`.

## Input format

`JSSPInstance.from_taillard_text()` accepts a simplified format: the first line contains `num_jobs num_machines`, followed by one line per job with alternating `machine_id processing_time` pairs. It does not parse the official Taillard two-matrix format directly.

## Execution boundary

`code_runner.py` runs LLM-generated Python in a subprocess with a timeout. This limits hangs and isolates ordinary exceptions from the workflow, but it is **not a security sandbox**. Review generated code before using this workflow with untrusted prompts or data.
