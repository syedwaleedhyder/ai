# AI Coding Agent

Inspired by Claude Code. A minimal, from-scratch coding agent: a CLI that sends your prompt to an LLM (via OpenRouter) and lets it iteratively call tools — list files, read files, write files, run Python — to complete a task, printing its plan and function calls as it goes.

## How it works

`main.py` runs an agentic loop (up to 20 iterations): it sends the conversation to the model along with a set of available tools, executes any tool calls the model requests, feeds the results back, and repeats until the model returns a plain-text final answer.

Available tools (defined in `functions/`):

- `get_files_info` — list files/directories with size info
- `get_file_content` — read a file's contents (truncated at `MAX_CHARS`, see `config.py`)
- `run_python_file` — execute a Python file with optional args (30s timeout)
- `write_file` — create or overwrite a file

All tool calls are sandboxed to a fixed working directory — currently hardcoded to `./calculator` in `call_function.py` — so the agent can only read/write/execute files inside that folder, regardless of what path the model requests.

The `calculator/` directory is a small sample Python project that the agent operates on by default.

## Setup

Requires Python >=3.11 and [uv](https://docs.astral.sh/uv/).

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file in this directory with your [OpenRouter](https://openrouter.ai/) API key:

   ```
   OPENROUTER_API_KEY=sk-or-...
   ```

## Usage

```bash
uv run main.py "<your prompt>" [--verbose] [--log]
```

Example:

```bash
uv run main.py "Identify the bug in the calculator, then fix it, then run tests to verify it." --log
```

Flags:

- `--verbose` — print each function call with its arguments, and token usage per LLM call
- `--log` — write a full JSONL trace of the session (LLM requests/responses, tool calls/results) to `logs/<timestamp>_<session_id>.jsonl`, useful for debugging

## Notes

- The model is hardcoded to `openrouter/free` in `main.py`.
- If the agent doesn't converge on a final answer within 20 iterations, it exits with an error.
