import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # so × and ° print correctly on Windows

client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=os.getenv("BASE_URL"))
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def ask(messages):
    # Free models are often overloaded and return no choices, so retry a few times.
    for attempt in range(5):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, parallel_tool_calls=True
        )
        if response.choices:
            print(json.dumps(response.model_dump(exclude_none=True), indent=2, ensure_ascii=False))
            return response.choices[0]
        time.sleep(2**attempt)
    raise RuntimeError(f"OpenRouter error: {response.error}")


def calculator(a: float, b: float, operation: str) -> float:
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[operation]


def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 22, "unit": "celsius", "condition": "sunny"}  # fake


TOOL_FUNCTIONS = {"calculator": calculator, "get_weather": get_weather}

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Do basic arithmetic on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                },
                "required": ["a", "b", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools.\n"
        "When the user asks several independent things, request ALL the tool calls "
        "you need in a single response instead of one at a time.",
    },
    {"role": "user", "content": input("You: ")},
]

# Keep calling the LLM until it answers with text instead of asking for tools.
# A plain question ("hi") -> 1 call. A question needing tools -> 2+ calls.
call = 1
while True:
    choice = ask(messages)
    print(f"\nCALL {call} finish_reason: {choice.finish_reason}")

    if not choice.message.tool_calls:
        print("Final answer:", choice.message.content)
        break

    # The LLM asked for tools: run each one and add the results to the history
    messages.append(choice.message.model_dump(exclude_none=True))
    for tool_call in choice.message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = TOOL_FUNCTIONS[tool_call.function.name](**args)
        print(f"-> ran {tool_call.function.name}({args}) = {result}")
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
    call += 1
