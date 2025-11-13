"""Actively listens to new actions and screenshots in real-time and induces workflows accordingly."""

import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from language import *

# %% Action/Screenshot Detection

from utils import (
	is_keyboard_action, is_scroll_action,
	get_key_input, compose_key_input
)

def load_actions_from_db(db_path: str) -> list[str]:
	"""Load the actions from the database. """
	engine = create_engine(f"sqlite:///{db_path}")

	with engine.connect() as connection:
		query = text("SELECT * from observations")
		df = pd.read_sql_query(query, connection)
	return df["content"].to_list()

def hotkey_in_action(action: str) -> bool:
	"""Check if the action contains a hotkey."""
	return any(hotkey in action for hotkey in [".cmd", ".enter", ".tab", ".up", ".down"])

def trigger_close_buffer(action: str, buffer_actions: list[str], enable_hotkey: bool = False) -> bool:
	"""Time to close the buffer: 
	- Current buffer is non-empty
	- Next new key/scroll action is different from the last action in the buffer.
	"""
	if len(buffer_actions) == 0:
		return False
	if is_keyboard_action(buffer_actions[-1]) and (not is_keyboard_action(action)):
		return True
	if is_scroll_action(buffer_actions[-1]) and (not is_scroll_action(action)):
		return True
	if enable_hotkey and is_keyboard_action(action) and hotkey_in_action(action):
		return True
	return False

def trigger_add_buffer(action: str, buffer_actions: list[str]) -> bool:
	"""Should add the new action to the buffer.
	- Is keyboard or scroll action
	- (i) buffer is empty; (ii) last action in buffer is the same type as the new action.
	"""
	if not (is_keyboard_action(action) or is_scroll_action(action)):
		return False
	if len(buffer_actions) == 0:
		return True
	if is_keyboard_action(action) and is_keyboard_action(buffer_actions[-1]):
		return True
	if is_scroll_action(action) and is_scroll_action(buffer_actions[-1]):
		return True
	return False

def process_actions(actions: list[str], enable_hotkey: bool = False) -> list[str]:
	"""Merge adjacent keyboard and scrolling actions into a single action."""
	original_actions, merged_actions = [], []
	buffer_actions = []
	for action in actions:
		close_buffer_flag = trigger_close_buffer(action, buffer_actions, enable_hotkey=enable_hotkey)
		if close_buffer_flag:
			if buffer_actions and is_keyboard_action(buffer_actions[0]):  # keypress buffer
				assert all([is_keyboard_action(action) for action in buffer_actions])
				original_actions.append({"before": buffer_actions[0], "after": buffer_actions[-1]})
				
				buffer_values = [get_key_input(action) for action in buffer_actions]
				keyboard_input = compose_key_input(buffer_values)
				merged_actions.append(f"key_press('{keyboard_input}')")
			elif buffer_actions and is_scroll_action(buffer_actions[0]):  # scroll buffer
				assert all([is_scroll_action(action) for action in buffer_actions])
				for ba in buffer_actions:
					if len(merged_actions) == 0 or ba != merged_actions[-1]:
						original_actions.append({"before": ba, "after": ba})
						merged_actions.append(ba)
			buffer_actions = []

		add_buffer_flag = trigger_add_buffer(action, buffer_actions)
		if add_buffer_flag:
			buffer_actions.append(action)
		else:
			merged_actions.append(action)
			original_actions.append({"before": action, "after": action})

	return original_actions, merged_actions


def detect_new_actions(action_path: str, last_count: int) -> tuple[list[dict], int]:
    """Detect new actions since the last check.
    Returns a list of new actions and the updated action count.
    """
    actions = load_actions_from_db(action_path)
    updated_count = len(actions)
    processed_actions, _ = process_actions(actions[last_count:], enable_hotkey=True)
    return processed_actions, updated_count


def find_screenshot(screenshot_paths: list[str], action: str, suffix: str) -> tuple[str, list[str]]:
	"""Find the screenshot path for the given action and suffix.
	Return the screenshot path and the remaining screenshot paths."""
	for i, sp in enumerate(screenshot_paths):
		if action in sp and sp.endswith(suffix):
			return screenshot_paths[i], screenshot_paths[: i] + screenshot_paths[i+1:]
	return None, screenshot_paths

def fill_in_states(states: list[dict]) -> list[dict]:
    """Fill in missing before/after states."""
    for i, state in enumerate(states):
        if state["before"] is None:
            # fill in from the previous state's after
            if i > 0:
                state["before"] = states[i-1]["after"]
        if state["after"] is None:
            # fill in from the next state's before
            if i < len(states) - 1:
                state["after"] = states[i+1]["before"]
    return states


def get_states(actions: list[dict], screenshot_dir: str) -> list[dict[str, str]]:
	"""Get before/after states (screenshots) associate with each action.
	"""
	screenshot_paths = os.listdir(screenshot_dir)
	screenshot_paths = sorted(screenshot_paths, key=lambda x: x.split('_')[0]) # sort by timestamp
	screenshot_paths = [os.path.join(screenshot_dir, p) for p in screenshot_paths]

	states = []
	for action_dict in actions:
		# print(action_dict)
		if is_keyboard_action(action_dict["before"]):
			suffix_before = "_session_start.jpg"
			suffix_after = "_session_end.jpg"
			action_dict["before"] = action_dict["before"].replace("''", "'")  # normalize the action string
		else:
			suffix_before = ".jpg"
			suffix_after = ".jpg"
		before_path, screenshot_paths = find_screenshot(screenshot_paths, action_dict["before"], suffix_before)
		after_path, screenshot_paths = find_screenshot(screenshot_paths, action_dict["after"], suffix_after)
		
		state = {"before": before_path, "after": after_path}
		# print("Action Dict:", action_dict)
		# print("State: ", state)
		# cont = input("Continue? (y/n)")
		states.append(state)

	states = fill_in_states(states)
	return states


def parse_time_from_path(path: str) -> float:
	"""Parse the time from the path."""
	return float(path.split('/')[-1].split('_')[0])

def measure_time_from_states(states: list[dict]) -> list[dict]:
	"""Measure the time from the states."""
	time_list = []
	for i, state in enumerate(states):
		# calculate time range
		try:
			before_time = parse_time_from_path(state["before"])
			after_time = parse_time_from_path(state.get("after", state["before"]))
		except:
			# print(f"Error parsing time from path: {state}")
			# cont = input("Continue? (y/n)")
			before_time = 0
			after_time = 0
		time_range = after_time - before_time

		# calculate time diff
		if i == 0:
			time_diff = 0
		else:
			try:
				last_time = parse_time_from_path(states[i-1].get("after", states[i-1]["before"]))
				time_diff = before_time - last_time
			except:
				# print(f"Error parsing time from path: {states[i-1]}")
				time_diff = 0

		time_list.append({
			"before": before_time, "after": after_time,
			"range": time_range, "diff": time_diff,
		})
	return time_list

def get_new_states(screenshots_dir: str, action_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """Detect new screenshots since the last check.
    Returns a list of new screenshot paths.
    """
    states = get_states(action_list, screenshots_dir)
    time_list = measure_time_from_states(states)
    return states, time_list


# %% MSE-based Segmentation

MAX_DIFF = 100000.0

def calc_diff_scores(old_nodes: list[ActionNode], new_nodes: list[ActionNode]) -> list[float]:

    def mse(image_before: str | None, image_after: str | None) -> float:
        """Calculate the mean squared error between two images."""
        # print(f"Calculating MSE between {image_before} and {image_after}")
        if image_before is None or image_after is None:
            return MAX_DIFF
        image1 = cv2.imread(image_before)
        image2 = cv2.imread(image_after)
        if image1.shape != image2.shape:
            # print(f"Image shapes do not match: {image1.shape} != {image2.shape}")
            return MAX_DIFF
        err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
        err /= float(image1.shape[0] * image1.shape[1])
        return err

    diff_scores = []
    for i, new_node in enumerate(new_nodes):
        if i == 0: 
            diff_score = mse(
                image_before=old_nodes[-1].state.after if old_nodes else None,
                image_after=new_node.state.before,
            )
        else:
            diff_score = mse(
                image_before=new_nodes[i-1].state.after,
                image_after=new_node.state.before,
            )
        diff_scores.append(diff_score)
    return diff_scores

def get_consistent_ranges(
    scores: list[float],
    threshold: float = 8000.0,
    min_steps: int = 5,
) -> list[tuple[int, int]]:
    """
    Find all ranges (start_index, end_index) where all scores are below threshold.
    
    Args:
        scores: List of numeric scores
        threshold: Threshold value to compare against
        
    Returns:
        List of tuples (start_index, end_index) where all scores in the range
        [start_index, end_index] are below threshold. both ends inclusive.
    """
    ranges = []
    start = None
    
    for i, score in enumerate(scores):
        if score < threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                ranges.append((start, i - 1))
                start = None
    
    # Handle case where the last range extends to the end
    if start is not None:
        ranges.append((start, len(scores) - 1))
    
    ranges = [r for r in ranges if (r[1] - r[0] + 1)>= min_steps]
    return ranges


def get_intervals_per_step(
    diff_scores: list[float],
    threshold: float = 8000.0,
    verbose: bool = False
) -> list[tuple[int, int]]:
    """Segment the trajectory at actions with above-threshold state differences."""
    intervals = []
    s = 0
    for i, diff_score in enumerate(diff_scores):
        # clean the current segment if a new high-diff step is found
        if (diff_score > threshold):
            intervals.append((s, i))
            s = i + 1

    # add the last segment if it exists
    if s < len(diff_scores): 
        intervals.append((s, len(diff_scores) - 1))
    if verbose:
        print(f"Found {len(intervals)} segments via mse diff threshold {threshold}.")
    return intervals


def get_intervals(
    ranges: list[tuple[int, int]],
    diff_scores: list[float],
    threshold: float = 10000.0, 
    min_steps: int = 5,
    verbose: bool = False
) -> list[tuple[int, int]]:
    intervals = []

    # add the first range
    s, e = ranges[0]
    if s <= min_steps:
        ranges = [(0, e)] + ranges[1: ]
    else:
        intervals.append((0, s-1))
    if verbose:
        print(f"Found {len(ranges)} ranges: ", ranges)
        cont = input("Continue? (Y/n)")

    # add the rest of the ranges
    i, L = 0, len(ranges)
    while i < (L - 1):
        # calculate gap with next range
        curr_range = ranges[i]
        gap = (ranges[i][1]+1, ranges[i+1][0]-1)  # both sides inclusive
        step_diff = gap[1] - curr_range[1]
        if step_diff >= min_steps: # add as two ranges
            intervals.append(curr_range)
            intervals.append(gap)
            if verbose:
                print(f"Added range {curr_range} and gap {gap} (step_diff: {step_diff})")
                # cont = input("Continue? (Y/n)")
        else:  # merge as one range
            intervals.append((curr_range[0], gap[1]))
            if verbose:
                print(f"Added merged range {intervals[-1]} (curr_range: {curr_range}) (gap: {gap}) (step_diff: {step_diff})")
                # cont = input("Continue? (Y/n)")
        i += 1
    
    # after the loop
    if i != L - 1:
        print("Suggestion: use `--default_segment` to use the default segmentation method. Do you want to run it now? (y/n)")
        # cont = input()
        if cont == "y":
            segments = get_intervals_per_step(diff_scores, threshold, verbose)
            return segments
        else:
            raise ValueError("i != L - 1" + f"(i: {i}, L-1: {L-1})")

    N = len(diff_scores)
    if ranges[i][1] < N-1:
        curr_range = ranges[i]
        gap = (ranges[i][1]+1, N-1)
        step_diff = gap[0] - curr_range[1]
        if step_diff >= min_steps:
            intervals.append(curr_range)
            intervals.append(gap)
        else:
            intervals.append((curr_range[0], gap[1]))
    else:
        assert ranges[i][1] == N-1
        intervals.append(ranges[i])
    return intervals


def trigger_segmentation(
    old_nodes: list[ActionNode], new_nodes: list[ActionNode],
    threshold: float = 8000.0, min_steps: int = 5, verbose: bool = False,
) -> list[tuple[int, int]]:
    """Determine if segmentation should be triggered based on new screenshots.
    Returns index in the `new_screenshots` list where segmentation should occur.
    Otherwise, returns -1.
    """
    diff_scores = calc_diff_scores(old_nodes, new_nodes)
    ranges = get_consistent_ranges(diff_scores, threshold=threshold, min_steps=min_steps)  # inclusive at both ends
    print(f"Found {len(ranges)} ranges via mse diff threshold {threshold}: {ranges}")
    if len(ranges) == 0:
        return []
    intervals = get_intervals(
        ranges, 
        diff_scores=diff_scores, 
        threshold=threshold, min_steps=min_steps, verbose=verbose
    )
    if verbose:
        print(f"Found {len(intervals)} segments via mse diff threshold {threshold}: {intervals}")

    n = len(old_nodes)
    if n > 0:
        intervals = [(s+n, e+n) for (s, e) in intervals]
    all_nodes = old_nodes + new_nodes
    segments = [all_nodes[s:e+1] for (s, e) in intervals]
    return segments


# %% Workflow Induction

def induce_workflow(segments: list[dict]) -> list[ActionNode | SequenceNode]:
    """Induce workflow steps based on the provided actions and screenshots.
    Returns a list of induced workflow steps.
    """
    nodes = []
    for i, seg in enumerate(segments):
        if len(seg) == 1:
            node = ActionNode(action=seg[0].action, state=seg[0].state, time=seg[0].time)
            node.get_goal()
        else:
            node = SequenceNode(nodes=seg)
            node = annotate_high_level_nodes(node, verbose=args.verbose)
        print(f"[{node.node_type.value}] Goal: {node.goal}")
        nodes.append(node)
    return nodes


# %% Main Loop

def main():
    workflow_steps = []
    curr_action_nodes = []

    action_count = 0
    while True:
        new_actions, action_count = detect_new_actions(args.action_path, last_count=action_count)
        new_states, new_times = get_new_states(args.screenshot_dir, action_list=new_actions)
        new_action_nodes = [ActionNode(action=a["before"], state=s, time=t) for a,s,t in zip(new_actions, new_states, new_times)]

        if (len(curr_action_nodes) + len(new_action_nodes)) > args.min_workflow_steps:
            print(f"Detected {len(curr_action_nodes)} current action nodes + {len(new_action_nodes)} new action nodes")
            segments = trigger_segmentation(
                curr_action_nodes, new_action_nodes, 
                threshold=args.segment_threshold,
                min_steps=args.min_interval_steps,
                verbose=args.verbose,
            )
            if len(segments) > 0:
                print(f"[Segmentation Triggered] {len(segments)}")
                curr_steps = induce_workflow(
                    segments=segments,
                )
                workflow_steps.extend(curr_steps)

                curr_action_nodes = []
            else:
                time.sleep(args.update_interval)  # wait before checking again
        else:
            curr_action_nodes.extend(new_action_nodes)
            time.sleep(args.update_interval)  # wait before checking again


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Induce workflows in real-time based on new actions and screenshots.")
    parser.add_argument("--data_dir", type=str, default="~/Downloads/records", help="Recording data directory.")
    parser.add_argument("--action_path", type=str, default="actions.db", help="Path to the actions database.")
    parser.add_argument("--screenshot_dir", type=str, default="screenshots", help="Directory to the screenshots.")

    parser.add_argument("--update_interval", type=int, default=10, help="Interval in seconds to check for new actions and screenshots.")
    parser.add_argument("--segment_threshold", type=float, default=8000.0, help="MSE threshold for segmentation.")
    parser.add_argument("--min_interval_steps", type=int, default=2, help="Minimum number of steps in a segment.")
    parser.add_argument("--min_workflow_steps", type=int, default=20, help="Minimum number of steps to induce a workflow.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)
    args.action_path = os.path.join(args.data_dir, args.action_path)
    args.screenshot_dir = os.path.join(args.data_dir, args.screenshot_dir)

    main()
