import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI


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
    messages = [
        {
            "role": "user",
            "content": args.user_prompt,
        }
    ]
    if args.verbose:
        print("User prompt:", args.user_prompt)

    response = client.chat.completions.create(model="openrouter/free", messages=messages)
    print(response.choices[0].message.content)
    usage = response.usage
    if args.verbose and usage is not None:
        print(f"Prompt tokens: {usage.prompt_tokens}")
        print(f"Response tokens: {usage.completion_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    main()
