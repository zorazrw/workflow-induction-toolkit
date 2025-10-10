"""Validate Workflow Quality, Specifically On Each Workflow Step:
- If the action sequence aligns with the step subgoal
- If the states are consistent
"""

import os
import json
import argparse
from utils import encode_image, call_openai
from language import ActionNode, SequenceNode, get_last_action

PROMPT_GOAL = "Your task is to determine if the action sequence aligns with the task goal. The actions may not have achieved the goal yet, return 'YES' as long as it attempts to progress towards the goal. If no action sequence is provided, return 'YES'."
PROMPT_MODULARITY = "Evaluate whether the current task-solving step is clearly focused on causally consistent procedures or achieving the same final goal, instead of a concatenation of multiple distinct topics or interfaces."
OUTPUT = "Return 'YES' or 'NO', optionally followed by your reasoning process."

PROMPT_DICT = {
    "goal": PROMPT_GOAL + '\n' + OUTPUT,
    "modularity": PROMPT_MODULARITY + '\n' + OUTPUT,
}

# %% Goal-Action/State Consistency

def format_chunk(chunk: dict) -> str:
    annotation = "# " + chunk["text"]
    actions = [step["action"] for step in chunk["steps"]]
    return '\n'.join([annotation] + actions)


def eval_goal(
    node: ActionNode | SequenceNode, 
    prompt: str,
    verbose: bool = True
) -> bool:
    goal = "Task Goal: " + str(node.goal)
    content = [{"type": "text", "text": goal}]

    if isinstance(node, ActionNode):
        content.append({"type": "text", "text": node.action})
        state = node.state.get_state()
        if state is not None:
            state_url = encode_image(state, return_url=True)
            content.append({"type": "image_url", "image_url": {"url": state_url}})

    else:  # SequenceNode
        for n in node.nodes:
            content.append({"type": "text", "text": str(n.get_semantic_repr())})
        if len(node.nodes) > 0:
            state = get_last_action(node.nodes[-1]).state.get_state()
            if state is not None:
                state_url = encode_image(state, return_url=True)
                content.append({"type": "image_url", "image_url": {"url": state_url}})

    # Truncate the content to 10 images
    state_indices = [i for i,c in enumerate(content) if "image_url" in c]
    if len(state_indices) > 10:
        rate = len(state_indices) // 10
        indices_to_remove = [i for i in state_indices if i % rate != 0]
        content = [c for i,c in enumerate(content) if i not in indices_to_remove]

    text = call_openai(prompt=prompt, content=content)
    if verbose: print(text)
    return "YES" in text, text


# %% State Consistency

def load_valid_state(step: dict) -> dict:
    if step["state"]["after"] is not None:
        return step["state"]["after"]
    elif step["state"]["before"] is not None:
        return step["state"]["before"]
    else:
        return None


def eval_step_goal(
    goal: str, 
    prev_goals: list[str] | None, 
    next_goals: list[str] | None, 
    prompt: str,
    verbose: bool = True,
) -> bool:
    content = [{"type": "text", "text": "Current Step: " + goal}]
    if prev_goals is not None:
        content.append({"type": "text", "text": "Previous Steps: " + "\n".join(prev_goals)})
    if next_goals is not None:
        content.append({"type": "text", "text": "Next Steps: " + "\n".join(next_goals)})
    text = call_openai(prompt=prompt, content=content)
    if verbose: print(text)
    return "YES" in text, text

def eval_step_goals(step_goals: list[str], prompt: str, model: str, verbose: bool = True) -> list[bool]:
    step_scores, step_contents = [], []
    for i, goal in enumerate(step_goals):
        can_separate, content = eval_step_goal(goal, step_goals[:i], step_goals[i+1:], prompt, model, verbose)
        step_scores.append(can_separate)
        step_contents.append(content)
    return step_scores, step_contents


def eval_step_goals_separate(step_goals: list[str], prompt: str, model: str, verbose: bool = True) -> list[bool]:
    step_scores, step_contents = [], []
    for i, goal in enumerate(step_goals):
        can_separate, content = eval_step_goal(goal, None, None, prompt, model, verbose)
        step_scores.append(can_separate)
        step_contents.append(content)
    return step_scores, step_contents


# %% Main

def main():
    workflow_root = SequenceNode.from_json(args.workflow_path)
    prompt = PROMPT_DICT[args.metric]

    if args.metric == "goal":
        step_scores, step_contents = [], []
        for i, node in enumerate(workflow_root.nodes):
            i_score, i_content = eval_goal(node, prompt, args.model_name, args.verbose)
            step_scores.append(i_score)
            step_contents.append(i_content)
            print(i_score)
        print(args.metric.capitalize(), ":", f"{sum(step_scores)}/{len(step_scores)} = {sum(step_scores)/len(step_scores):.1f}")
        # Save the step contents
        with open(os.path.join(args.output_dir, f"{args.metric}.json"), "w") as f:
            json.dump(step_contents, f)
    elif args.metric == "modularity":
        step_goals = []
        for node in workflow_root.nodes:
            if node.status.value == "success":
                step_goals.append(node.goal)
            else:
                step_goals.append(node.goal + " (Attempt Failed)")
        if args.metric == "concision":
            step_scores, step_contents = eval_step_goals(step_goals, prompt, args.model_name, args.verbose)
        else:  # modularity
            step_scores, step_contents = eval_step_goals_separate(step_goals, prompt, args.model_name, args.verbose)
        step_scores = [int(s) for s in step_scores]
        print(args.metric.capitalize(), ":", f"{sum(step_scores)}/{len(step_scores)} = {sum(step_scores)/len(step_scores):.1f}")
        # Save the step contents
        with open(args.output_path, "w") as f:
            json.dump(step_contents, f)
    else:
        raise ValueError(f"Invalid metric: {args.metric}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The directory containing the `workflow.json` file.")
    parser.add_argument("--output_dir", type=str, default="validation",
                        help="The directory to save the evaluation results.")
    parser.add_argument("--metric", type=str, default="goal",
                        choices=["goal", "modularity"],
                        help="The metric to evaluate the workflow.")

    parser.add_argument("--model_name", type=str, default="gpt-4o",)
    
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.workflow_path = os.path.join(args.data_dir, "workflow.json")
    args.output_dir = os.path.join(args.data_dir, args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_path = os.path.join(args.output_dir, f"{args.metric}.json")

    main()
