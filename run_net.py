#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import parse_args
from slowfast.config.defaults import get_cfg  # Import get_cfg directly

from test_net import test


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = get_cfg()  # Use get_cfg directly
    cfg.merge_from_file(args.cfg_files[0])  # Load the config file

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()