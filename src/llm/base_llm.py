import requests
import json
import os
from typing import List, Dict
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def generate_response(self, instructions: str, prompt: str) -> str | None:
        pass


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "Qwen/Qwen3.5-9B",
    together_api_key=None,
    **kwargs,
):
    # Remove None parameters for Together API - don't set to string 'none'
    if top_p is None:
        payload_top_p = None
    else:
        payload_top_p = top_p
    if temperature is None:
        payload_temperature = None
    else:
        payload_temperature = temperature

    payload = {
        "model": model,
        "messages": [{"role": role, "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        **kwargs,
    }
    # Only add temperature and top_p if they're not None
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
        url = os.path.join(get_proxy_url(), "v1/chat/completions")
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(
                f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}"
            )
    else:
        if together_api_key is None:
            together_api_key = os.environ["TOGETHER_API_KEY"]
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict["choices"][-1]["message"]["role"] = json_dict["choices"][-1][
            "message"
        ]["role"].name.lower()
    try:
        output_dict = {
            "role": json_dict["choices"][-1]["message"]["role"],
            "content": json_dict["choices"][-1]["message"]["content"],
        }
    except Exception as e:
        raise Exception(
            f"Failed to get correct output dict. Please try again. Error: {e}"
        )
    return output_dict


def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = 500,
    model: str = "Qwen/Qwen3.5-9B",
    together_api_key=None,
    **kwargs,
):
    # Remove None parameters for Together API
    if top_p is None:
        payload_top_p = None
    else:
        payload_top_p = top_p
    if temperature is None:
        payload_temperature = None
    else:
        payload_temperature = temperature

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
        **kwargs,
    }
    # Only add temperature and top_p if they're not None
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    if (not together_api_key) and ("TOGETHER_API_KEY" not in os.environ):
        url = os.path.join(get_proxy_url(), "v1/chat/completions")
        response = requests.post(url, json=payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(
                f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}"
            )
    else:
        if together_api_key is None:
            together_api_key = os.environ["TOGETHER_API_KEY"]
        client = Together(api_key=together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict["choices"][-1]["message"]["role"] = json_dict["choices"][-1][
            "message"
        ]["role"].name.lower()
    try:
        output_dict = {
            "role": json_dict["choices"][-1]["message"]["role"],
            "content": json_dict["choices"][-1]["message"]["content"],
        }
    except Exception as e:
        raise Exception(
            f"Failed to get correct output dict. Please try again. Error: {e}"
        )
    return output_dict


def generate_params_dict(
    prompt: str,
    temperature: float = None,
    role="user",
    top_p: float = None,
    max_tokens: int = 500,
    model: str = "Qwen/Qwen3.5-9B",
):
    """
    Call an LLM with different sampling parameters to observe their effects.

    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use

    Returns:
        The LLM response
    """

    # Create the dictionary with the necessary parameters
    kwargs = {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model": model,
    }

    return kwargs


def call_llm_with_context(prompt: str, context: list, role: str = "user", **kwargs):
    """
    Calls a language model with the given prompt and context to generate a response.

    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - role (str): The role of the participant in the conversation, e.g., "user" or "assistant".
    - context (list): A list representing the conversation history, to which the new input is added.
    - **kwargs: Additional keyword arguments for configuring the language model call (e.g., top_k, temperature).

    Returns:
    - response (str): The generated response from the language model based on the provided prompt and context.
    """

    # Append the dictionary {'role': role, 'content': prompt} into the context list
    context.append({"role": role, "content": prompt})

    # Call the llm with multiple input passing the context list and the **kwargs
    response = generate_with_multiple_input(context, **kwargs)

    # Append the LLM response in the context dict
    context.append(response)

    return response


def generate_system_call(command):
    PROMPT = f"""
You are an assistant program that converts natural language commands into structured JSON for controlling smart home devices. The JSON should conform to a specific format describing the device, action, and parameters. Here's how you can do it:

**Available Devices and Actions:**

1. **Light**
   - Actions: "turn on", "turn off"
   - Parameters: color, intensity (percentage)

2. **Automatic Lock**
   - Actions: "lock", "unlock"
   - Parameters: None

3. **Sound System (Speaker)**
   - Actions: "play", "pause", "stop", "set volume"
   - Parameters: volume (integer), track (string), playlist_style (string)

4. **TV**
   - Actions: "turn on", "turn off", "change channel", "adjust volume"
   - Parameters: channel (string), volume (integer)

5. **Air Conditioner**
   - Actions: "turn on", "turn off", "set temperature", "adjust fan speed"
   - Parameters: temperature (integer), fan_speed (low/medium/high)

**Rooms and Devices:**
- **Office**
  - Lights: "office_light_1" (ID: 123), "office_light_2" (ID: 321)
  - Automatic Lock: "office_door_lock" (ID: 111)

- **Living Room**
  - Light: "living_room_light" (ID: 222)
  - Speaker: "living_room_speaker" (ID: 223)
  - Air Conditioner: "living_room_airconditioner" (ID: 556)

- **Kitchen**
  - Light: "kitchen_light" (ID: 333)

- **Bedroom**
  - Light: "bedroom_light" (ID: 444)
  - TV: "bedroom_tv" (ID: 445)

- **Bathroom**
  - Light: "bathroom_light" (ID: 555)

**Task:**
Convert the following natural language command into the structured JSON format based on the available devices:

**Input Examples:**

1. "Turn on the office light with ID 123 with blue color and 50% intensity."
   - JSON:
     [
     {{
       "room": "office",
       "object_id": "123",
       "object_name": "office_light_1",
       "action": "turn on",
       "parameters": {{"color": "blue", "intensity": "50%"}}
     }}
     ]

2. "Lock the office door."
   - JSON:
   [
     {{
       "room": "office",
       "object_id": "111",
       "object_name": "office_door_lock",
       "action": "lock",
       "parameters": {{}}
     }}
    ]

2. "Make my living room a cheerful place"
   - JSON:
   [
     {{
       "room": "living_room",
       "object_id": "222",
       "object_name": "living_room_light",
       "action": "turn on",
       "parameters": {{'intensity': '80%', 'color':'yellow'}}
     }},
     {{
       "room": "living_room",
       "object_id": "223",
       "object_name": "living_room_speaker",
       "action": "turn on",
       "parameters": {{'volume': '100', 'playlist_style':'party'}}
     }},
     
   ]

**Note:**
- Ensure that each JSON object correctly maps the natural command to the appropriate device and action using the listed device ID.
- Use the object ID to differentiate between devices when the room contains multiple similar items.
- You can add more than one parameter in the parameters dictionary.

Using this information, translate the following command into JSON: "{command}". Output a list with all the necessary JSONs. 
Always output a list even if there is only one command to be applied, do not output anything else but the desired structure.
"""
    kwargs = generate_params_dict(PROMPT, temperature=0.4, top_p=0.1)
    result = generate_with_single_input(**kwargs)
    return result["content"]
