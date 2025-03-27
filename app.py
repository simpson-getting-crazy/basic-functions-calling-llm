import json
import os
import requests
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Check for required environment variables
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL')

if not api_key or not base_url:
    print("Error: Missing required environment variables.")
    print("Please ensure OPENAI_API_KEY and OPENAI_BASE_URL are set in your .env file.")
    sys.exit(1)

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

user_input = input("Welcome to the Weather Assistant! Please enter a prompt: \n")

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

messages = [
    {"role": "system", "content": "You are a helpful weather assistant"},
    {"role": "user", "content": user_input},
]

completion = client.chat.completions.create(
    model="openai/gpt-4o-2024-11-20",
    messages=messages,
    tools=tools,
)

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    
for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current temperature in celsius.")
    response: str = Field(description="The response from the weather API.")

final_completion = client.beta.chat.completions.parse(
    model="openai/gpt-4o-2024-11-20",
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

final_response = final_completion.choices[0].message.parsed
print(final_response)