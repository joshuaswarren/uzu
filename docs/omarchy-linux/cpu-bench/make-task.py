"""Write the shared prompt and the uzu `cli bench` task for the CPU comparison.
Usage: make-task.py OUTDIR PROMPT_WORDS TOKENS_LIMIT RUNS
The same prompt text goes to uzu (chat message) and llama.cpp (-f file); the chat templates differ, so token counts are
reported by each engine and not assumed equal."""
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
words = int(sys.argv[2])
limit = int(sys.argv[3])
runs = int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
base = ("The committee reviewed the quarterly logistics report and noted that regional warehouses had reduced delivery "
        "times while fuel costs rose. Members asked for a breakdown by carrier, a comparison with last year, and an "
        "estimate of how seasonal demand would change the staffing plan for the next two quarters. ").split()
text = " ".join((base * (words // len(base) + 1))[:words]) + "\n\nSummarize the report in two sentences."
(out / "prompt.txt").write_text(text)
task = {
    "identifier": "cpu-compare",
    "repo_id": "alibaba:qwen3.5:0.8b:mirai:mirai-m:4",
    "number_of_runs": runs,
    "tokens_limit": limit,
    "messages": [{"role": "user", "content": text, "reasoning_content": None}],
    "greedy": True,
}
(out / "task.json").write_text(json.dumps(task, indent=1))
print(f"prompt words={len(text.split())} tokens_limit={limit} runs={runs}")
