import sys
from run_net import main

# Simulate command-line arguments
sys.argv = [
    "run_net.py",  # Script name (required by argparse)
    "--cfg", "D:/Github/slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml",  # Path to your config file
    # Add other arguments if needed, e.g., --init_method, etc.
]

# Call the main function
main()