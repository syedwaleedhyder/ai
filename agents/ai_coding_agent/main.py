import os
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from call_function import available_functions, call_function


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

    MAX_ITERATIONS = 20
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            tools=available_functions,
            # temperature=0,
        )
        message = response.choices[0].message
        messages.append(message)

        usage = response.usage
        if args.verbose and usage is not None:
            print(f"Prompt tokens: {usage.prompt_tokens}")
            print(f"Response tokens: {usage.completion_tokens}")

        if not message.tool_calls:
            print("Final response:")
            print(message.content)
            return

        for tool_call in message.tool_calls:
            result_message = call_function(tool_call, verbose=args.verbose)
            if not result_message["content"]:
                raise Exception(f"Fatal: function {tool_call.function.name} returned empty content")
            if args.verbose:
                print(f"-> {result_message['content']}")
            messages.append(result_message)

    print(f"Error: reached max iterations ({MAX_ITERATIONS}) without a final response.")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    main()
