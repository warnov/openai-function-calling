import os
import json
import datetime
import random
import json
from typing import List


from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt




client = AzureOpenAI(
api_key=os.getenv("AI_ROADSHOW_AOAI_KEY"),  
api_version="2024-02-01",
azure_endpoint=os.getenv("AI_ROADSHOW_AOIA_ENDPOINT"))

deployment = "ai-roadshow" #gpt-3.5-turbo-0613
messages = []


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city for which to get the weather",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                        "description": "The temperature unit to use. Default is fahrenheit.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location the user wants to know its weather",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Default is fahrenheit.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]







def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "bogota" in location.lower():
        return json.dumps({"location": "Bogota", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
    
def get_n_day_weather_forecast(location, unit="fahrenheit", num_days=3):
    current_date = datetime.date.today()
    forecast = []
    
    for i in range(num_days):
        date = current_date + datetime.timedelta(days=i)
        temperature = random.randint(10, 20)
        forecast.append(f"{date.strftime('%B %d')}: {temperature}")
    
    return json.dumps({"location": location, "forecast": forecast, "unit": unit})


def pretty_print_chat_completion_message(chat_message: ChatCompletionMessage):
    content = chat_message.content
    print("Content:")
    print(json.dumps(content, indent=2))

    if chat_message.tool_calls:
        function_name = chat_message.tool_calls[0].function.name
        arguments = chat_message.tool_calls[0].function.arguments
        print("Function Name:", function_name)
        print("Arguments:")
        print(json.dumps(json.loads(arguments), indent=2))
    else:
        print("Functions: None")


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=deployment):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e       


def function_caller(tool_calls: List[ChatCompletionMessageToolCall]):
    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
            "get_n_day_weather_forecast": get_n_day_weather_forecast,	
        }  
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "get_current_weather":
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"))
            elif function_name == "get_n_day_weather_forecast":
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                    num_days=function_args.get("num_days"))
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  
        second_response = client.chat.completions.create(
            model=deployment,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content


# Testing specifying location
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "I'm visiting San Francisco. What will be the weather today?"})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
pretty_print_chat_completion_message(assistant_message)
print(function_caller(assistant_message.tool_calls))
input("Press any key to continue with the next sample...\n\n")


# Testing specifying the location and the preferred units
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Please tell me the weather in Bogota, in celsius"})
chat_response = chat_completion_request(
    messages, tools=tools
)
assistant_message = chat_response.choices[0].message
messages.append(assistant_message)
pretty_print_chat_completion_message(assistant_message)
print(function_caller(assistant_message.tool_calls))