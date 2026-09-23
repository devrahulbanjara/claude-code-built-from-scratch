import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")  # so × and ° print correctly on Windows

client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=os.getenv("BASE_URL"))
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


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


def ask_stream(messages):
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, parallel_tool_calls=True, stream=True
    )

    content = ""
    tool_calls = {}  # index -> tool call being built up piece by piece
    finish_reason = None

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        # Show the raw piece that arrived in this chunk
        piece = delta.model_dump(exclude_none=True, exclude={"reasoning", "reasoning_details"})
        if piece.get("content") or piece.get("tool_calls") or choice.finish_reason:
            print("CHUNK:", json.dumps(piece), "| finish_reason:", choice.finish_reason)

        # Text arrives as small pieces: just join them
        if delta.content:
            content += delta.content

        # Tool calls arrive as pieces too. `index` says which tool call a piece belongs to.
        # The first piece has the id and name, later pieces only add more of the arguments.
        for tc in delta.tool_calls or []:
            call = tool_calls.setdefault(
                tc.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc.id:
                call["id"] = tc.id
            if tc.function and tc.function.name:
                call["function"]["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                call["function"]["arguments"] += tc.function.arguments

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    return content, list(tool_calls.values()), finish_reason


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to tools.\n"
        "When the user asks several independent things, request ALL the tool calls "
        "you need in a single response instead of one at a time.",
    },
    {"role": "user", "content": input("You: ")},
]

call = 1
while True:
    print(f"\n========== CALL {call} ==========")
    content, tool_calls, finish_reason = ask_stream(messages)
    print("finish_reason:", finish_reason)

    if not tool_calls:
        print("Final answer:", content)
        break

    # Only now, after the stream ended, are the arguments complete JSON
    print("Assembled tool_calls:", json.dumps(tool_calls, indent=2))

    messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})
    for tool_call in tool_calls:
        args = json.loads(tool_call["function"]["arguments"])
        result = TOOL_FUNCTIONS[tool_call["function"]["name"]](**args)
        print(f"-> ran {tool_call['function']['name']}({args}) = {result}")
        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result)})
    call += 1
