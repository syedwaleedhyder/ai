import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)




def main():
    print("Hello from ai-coding-agent!")
    messages: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=args.user_prompt)])
    ]
    if args.verbose:
        print("User prompt:", args.user_prompt)

    response = client.models.generate_content(model="gemini-2.5-flash", contents=messages)
    print(response.text)
    usage_metadata = response.usage_metadata
    prompt_token_count = usage_metadata.prompt_token_count
    candidates_token_count = usage_metadata.candidates_token_count
    if args.verbose:
        print(f"Prompt tokens: {prompt_token_count}")
        print(f"Response tokens: {candidates_token_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    # Now we can access `args.user_prompt`
    main()
