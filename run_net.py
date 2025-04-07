import sys
#from test_net import main  # Updated to import from test_net.py
#from faster_rcnn_slowFast import main  # Updated to import from test_net.py
from faster_rcnn_slowfast_finetune import main

from multiprocessing import freeze_support

# Simulate command-line arguments
sys.argv = [
    "run_net.py",  # Script name (required by argparse)
    "--cfg", "D:/Github/slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml",  # Path to your config file
]

if __name__ == "__main__":
    freeze_support()
    main()