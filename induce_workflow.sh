#!/bin/bash
# Quick Start Script for the Workflow Induction Toolkit

DATA_DIR=$1

# go to the workflow-induction directory
cd workflow-induction

# preprocess recorded human activities
python get_human_trajectory.py --data_dir $DATA_DIR

# segment the trajectory based on state transitions
python segment.py --data_dir $DATA_DIR

# perform semantic-based segment merging
python induce.py --data_dir $DATA_DIR --auto

# the workflow will be saved in the `${DATA_DIR}/workflow.json` as a JSON file.
# the high-level step description will be saved in the `${DATA_DIR}/workflow.txt` as plain texts.