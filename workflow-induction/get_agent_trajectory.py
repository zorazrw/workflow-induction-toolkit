import os
import json
import argparse
from language import State, Time, ActionNode, SequenceNode

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# %% Path Utils

def get_task_name(data_dir: str) -> str:
    return data_dir.split("/")[-3]

def get_default_screenshot_url(task_name: str, expr_name: str = "20250614_OpenHands-Versa-claude-sonnet-4") -> str:
    return f"https://github.com/TheAgentCompany/experiments/raw/main/evaluation/1.0.0/{expr_name}/screenshots/{task_name}-image/"

# %% Parse Agent Trajectory

# Action
def get_action_detail(step: dict) -> str:
    # high_level_action = step["action"]  # ["browser_interactive", "run_ipython", "run", "edit", "read", "finish"]
    tool_calls = step["tool_call_metadata"]["model_response"]["choices"][0]["message"]["tool_calls"]
    actions = [tc["function"]["arguments"] for tc in tool_calls]
    return ' | '.join(actions)

# State 
def format_state_url(url: str, name: str) -> str:
    return f"{url}/{name}"

def format_state_name(index: int, use_som: bool = True) -> str:
    return f"{index}_som.png" if use_som else f"{index}.png"

def download_file(url: str, to_dir: str):
    cmd = f"wget -P {to_dir} {url}"
    os.system(cmd)

def get_state(index: int, screenshot_dict: dict, use_som: bool = True) -> str:
    if index in screenshot_dict:
        print(f"Using cached screenshot {index} ...")
        return screenshot_dict[index], screenshot_dict
    else:
        print(f"Downloading screenshot {index} ...")
        name = format_state_name(index=index, use_som=use_som)
        path = os.path.join(args.screenshot_dir, name)
        if not os.path.exists(path):
            url = format_state_url(url=args.screenshot_url, name=name)
            download_file(url, args.screenshot_dir)
        if not os.path.exists(path):  # download failed, often file not found
            return None, screenshot_dict
        screenshot_dict[index] = path
        return path, screenshot_dict



# Goal
PROMPT = """Your task is to extract the goal for this step, do NOT include any summary of previous steps, do NOT include plans for future steps. Only return the goal, no other text such as 'The goal for this step is:'."""

def get_goal(thought: str, llm_refine: bool = False, debug: bool = False) -> str:
    if not llm_refine: return thought
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": thought}
    ]
    response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=messages,
		temperature=0.0,
	)
    goal = response.choices[0].message.content
    if debug: 
        feedback = input(f"Thought: {thought}\nGoal: {goal}\nFeedback: ")
        if feedback == "":
            return goal
        else:
            return feedback
    return goal



# %% Main

def main():
    raw_traj = json.load(open(args.input_path))
    print("Task Instruction: ", raw_traj[0]["message"])

    screenshot_dict = {}

    node_list = []
    for i, raw_step in enumerate(raw_traj):
        if (raw_step["source"] == "agent") and ("action" in raw_step) and (raw_step["action"] != "finish"):
            before_state_path, screenshot_dict = get_state(index=i-1, screenshot_dict=screenshot_dict, use_som="claude" in args.data_dir)
            after_state_path, screenshot_dict = get_state(index=i+1, screenshot_dict=screenshot_dict, use_som="claude" in args.data_dir)
            state = State(before=before_state_path, after=after_state_path)
            time = Time(before=raw_step["timestamp"], after=raw_step["timestamp"])

            try:
                action_detail = get_action_detail(raw_step)
                action = f"{raw_step['action']} | {action_detail}"
            except:
                action = raw_step["action"]

            if i == 0: thought = action
            else: 
                if "claude" in args.data_dir:
                    thought = raw_step["args"]["thought"]
                else:
                    thought = raw_step["message"]
                
            goal = get_goal(thought, llm_refine=args.llm_refine, debug=args.debug)
            node = ActionNode(action=action, state=state, goal=goal, time=time)

            node_list.append(node)

    seq_node = SequenceNode(nodes=node_list)
    seq_node.to_json(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The directory containing the raw trajectory data.")
    parser.add_argument("--input_path", type=str, default="raw_trajectory.json",
                        help="The path to the raw trajectory data.")
    parser.add_argument("--output_path", type=str, default="processed_trajectory.json",
                        help="The path to the processed trajectory data.")
    parser.add_argument("--screenshot_dir", type=str, default="screenshots",
                        help="The directory containing or to save the screenshots.")
    parser.add_argument("--screenshot_url", type=str, default=None,
                        help="The URL to the screenshots.")
    parser.add_argument("--expr_name", type=str, default="20250614_OpenHands-Versa-claude-sonnet-4",
                        choices=["20250614_OpenHands-Versa-claude-sonnet-4", "20241217_OpenHands-0.14.2-gpt-4o-2024-08-06"],
                        help="The name of the expression.")
    parser.add_argument("--llm_refine", action="store_true",
                        help="Whether to refine the action-wise goal using LLM.")
    parser.add_argument("--debug", action="store_true",
                        help="Whether to debug the goal refinement.")
    args = parser.parse_args()

    args.input_path = os.path.join(args.data_dir, args.input_path)
    args.output_path = os.path.join(args.data_dir, args.output_path)
    args.screenshot_dir = os.path.join(args.data_dir, args.screenshot_dir)
    if not os.path.exists(args.screenshot_dir):
        os.makedirs(args.screenshot_dir)

    if args.screenshot_url is None:
        task_name = get_task_name(data_dir=args.data_dir)
        print(f"Using task name {task_name} to construct default screenshot url.")
        args.screenshot_url = get_default_screenshot_url(task_name=task_name, expr_name=args.expr_name)

    main()
