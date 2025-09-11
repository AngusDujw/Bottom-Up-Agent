import re
import json
import time
import math
import base64
import io
from typing import Optional, Dict, Any, List, Tuple

from PIL import Image
from openai import OpenAI

# 全局常量
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 16384 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def parse_action(action_str: str) -> Optional[Dict[str, Any]]:
    action = action_str.strip()
    
    # Special case for drag with box syntax
    if action.lower().startswith("drag") and ("start_box" in action and "end_box" in action):
        # First try handling bbox format
        bbox_matches = re.findall(r"<bbox>(\d+)\s+(\d+)(?:\s+\d+\s+\d+)?</bbox>", action)
        if len(bbox_matches) == 2:
            (x1, y1), (x2, y2) = [(int(px), int(py)) for px, py in bbox_matches]
            return {"name": "Drag", "input": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
        
        # Then try box_start/box_end format with potential 'x' prefix
        box_coords = re.findall(r"<\|box_start\|>\((?:x)?(\d+)\s+(\d+)\)<\|box_end\|>", action)
        if len(box_coords) == 2:
            (x1, y1), (x2, y2) = [(int(px), int(py)) for px, py in box_coords]
            return {"name": "Drag", "input": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
        
        # Extract coordinate pairs directly as a fallback
        coord_pairs = re.findall(r"\((?:x)?(\d+)\s+(\d+)\)", action)
        if len(coord_pairs) == 2:
            (x1, y1), (x2, y2) = [(int(px), int(py)) for px, py in coord_pairs]
            return {"name": "Drag", "input": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
        
        # Last resort: just extract all numbers and use first 4
        nums = re.findall(r"\d+", action)
        if len(nums) >= 4:
            x1, y1, x2, y2 = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
            return {"name": "Drag", "input": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
            
        return None
    
    # 通用回退：匹配 op(arg1, arg2, ...)
    m = re.match(r"^([a-zA-Z_]+)\s*\((.*)\)$", action, re.DOTALL)
    if not m:
        return None
    op = m.group(1).lower()
    args_block = m.group(2)

    # 提取所有整数，并处理可能带有 'x' 前缀的坐标
    nums = re.findall(r"(?:x)?(\d+)", args_block)
    coords = [int(n) for n in nums]

    if op == "click" and len(coords) >= 2:
        x, y = coords[0], coords[1]
        return {"name": "Click", "input": {"x": x, "y": y}}
    if op in ("right_single", "right_single_click") and len(coords) >= 2:
        x, y = coords[0], coords[1]
        return {"name": "RightSingle", "input": {"x": x, "y": y}}
    if op == "drag" and len(coords) == 4:
        x1, y1, x2, y2 = coords
        return {"name": "Drag", "input": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}

    return None

def parse_response(text: str) -> Dict[str, Any]:
    reflection = None
    thought = ""
    m = re.search(r"Reflection:\s*(.+?)(?=\n(Thought:|Action_Summary:|Action:))", text, re.DOTALL)
    if m:
        reflection = m.group(1).strip()
    m2 = re.search(r"Thought:\s*(.+?)(?=\nAction:)", text, re.DOTALL)
    if m2:
        thought = m2.group(1).strip()
    else:
        m3 = re.search(r"Action_Summary:\s*(.+?)(?=\nAction:)", text, re.DOTALL)
        if m3:
            thought = m3.group(1).strip()

    # Updated pattern to capture complex action calls with nested parentheses
    action_lines = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
    actions: List[Dict[str, Any]] = []
    
    for action_line in action_lines:
        action_line = action_line.strip()
        if action_line:
            parsed = parse_action(action_line)
            if not parsed:
                raise ValueError(f"can not parse Action: {action_line}")
            actions.append(parsed)
    
    return {"reflection": reflection, "thought": thought, "actions": actions}

class UI_TARS:
    def __init__(self, model_name, api_key=None):
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key
        )
        self.model_name = "doubao-1.5-ui-tars-250428"


    def call_text_images(self, text_prompt: str, imgs: List[str],
                         tools=None, max_iterations=5, pre_knowledge=None) -> Dict[str, Any]:

        messages=  []

        if pre_knowledge is not None:
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": pre_knowledge
                    }
                ]
            })

        message = {
            "role": "user",
            "content": [{"type":"text","text": text_prompt}]
        }

        img_data = base64.b64decode(imgs[0])
        img = Image.open(io.BytesIO(img_data))
        orig_w, orig_h = img.size
        # res_h, res_w = smart_resize(
        #     height=orig_h, width=orig_w,
        #     factor=IMAGE_FACTOR,
        #     min_pixels=MIN_PIXELS,
        #     max_pixels=MAX_PIXELS
        # )
        

        for img_b64 in imgs:

            message['content'].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        messages.append(message)

        for _ in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=2000
                )

                content = response.choices[0].message.content
                print(content)
                parsed = parse_response(content)
                action = parsed["actions"][0] if parsed["actions"] else None

                if action:
                    params = action["input"]

                    # Click and RightSingle
                    if action["name"] in ("Click", "RightSingle"):
                        x = params["x"]
                        y = params["y"]
                        orig_x = int(x * orig_w  / 1000.0)
                        orig_y = int(y * orig_h  / 1000.0)
                        action["input"]["x"] = orig_x
                        action["input"]["y"] = orig_y

                    # Drag
                    elif action["name"] == "Drag":
                        x1 = params["x1"]; y1 = params["y1"]
                        x2 = params["x2"]; y2 = params["y2"]
                        action["input"]["x1"] = int(x1 * orig_w  / 1000.0)
                        action["input"]["y1"] = int(y1 * orig_h  / 1000.0)
                        action["input"]["x2"] = int(x2 * orig_w  / 1000.0)
                        action["input"]["y2"] = int(y2 * orig_h  / 1000.0)

                return {
                    "message": content,
                    "function": action,
                    "usage": {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    }
                }

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)

        raise Exception("OpenAI API call failed after multiple attempts.")

