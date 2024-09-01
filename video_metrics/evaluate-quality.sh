#!/bin/bash

# Define the model list
paths=("YOUR_PATH/data/720p_result" "YOUR_PATH/data/1_5k_result" "YOUR_PATH/data/2k_result")

# Define the dimension list
dimensions=("aesthetic_quality" "imaging_quality")

# Loop over each model
for path in "${paths[@]}"; do
    # Loop over each dimension
    for i in "${!dimensions[@]}"; do
        # Get the dimension and corresponding folder
        dimension=${dimensions[i]}

        # Construct the video path
        videos_path=${path[@]}
        echo "$dimension $videos_path"

        # Run the evaluation script
        python3 evaluate.py --videos_path $videos_path --dimension $dimension --mode=custom_input
    done
done
