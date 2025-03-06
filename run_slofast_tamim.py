import sys
from run_net import main
from multiprocessing import freeze_support
# Simulate command-line arguments
sys.argv = [
    "run_net.py",  # Script name (required by argparse)
    "--cfg", "D:/Github/slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml",  # Path to your config file
    # Add other arguments if needed, e.g., --init_method, etc.
]

if __name__ == "__main__":
# Call the main function
    freeze_support()
    main()