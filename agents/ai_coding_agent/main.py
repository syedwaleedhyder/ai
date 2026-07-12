import os
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from call_function import available_functions, call_function
from logger import new_session_id, setup_session_logger, log_event


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if api_key is None:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Please set it in your .env file."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def main():
    print("Hello from ai-coding-agent!")

    session_id = new_session_id()
    session_logger = setup_session_logger(session_id, enabled=args.log)
    if args.log:
        print(f"Logging this session to logs/*_{session_id}.jsonl")

    system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read file contents
- Execute Python files with optional arguments
- Write or overwrite files

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.

Work step by step: call whatever functions you need to gather information or make changes, look at the results, and keep going until the task is actually done. Only give a final plain-text response (with no function calls) once you're confident the task is complete.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    if args.verbose:
        print("Messages:", messages)
    log_event(session_logger, session_id, "session_start", {"user_prompt": args.user_prompt})

    MAX_ITERATIONS = 20
    for _ in range(MAX_ITERATIONS):
        log_event(session_logger, session_id, "llm_request", {"model": "openrouter/free", "messages": messages})
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            tools=available_functions,
            # temperature=0,
        )
        log_event(session_logger, session_id, "llm_response", response)
        message = response.choices[0].message
        messages.append(message)

        usage = response.usage
        if args.verbose and usage is not None:
            print(f"Prompt tokens: {usage.prompt_tokens}")
            print(f"Response tokens: {usage.completion_tokens}")

        if not message.tool_calls:
            print("Final response:")
            print(message.content)
            log_event(session_logger, session_id, "session_end", {"final_response": message.content})
            return

        for tool_call in message.tool_calls:
            result_message = call_function(
                tool_call, verbose=args.verbose, session_logger=session_logger, session_id=session_id
            )
            if not result_message["content"]:
                raise Exception(f"Fatal: function {tool_call.function.name} returned empty content")
            if args.verbose:
                print(f"-> {result_message['content']}")
            messages.append(result_message)

    print(f"Error: reached max iterations ({MAX_ITERATIONS}) without a final response.")
    log_event(session_logger, session_id, "session_end", {"error": "max_iterations_reached"})
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log every LLM call, tool call, and tool result to a JSONL file under logs/ for debugging",
    )
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    main()
