#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Modified to process one video segment at a time for SlowFast feature extraction with proper feature map visualization

import numpy as np
import torch
import os
import time
from tqdm import tqdm
import glob
import pickle
from PIL import Image
import torchvision.transforms as transforms
import warnings
import matplotlib.pyplot as plt
import cv2

# Suppress the torchvision warning
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._transforms_video")

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.utils.parser import parse_args
from slowfast.config.defaults import get_cfg
from slowfast.utils.misc import launch_job

from models import build_model

logger = logging.get_logger(__name__)

# Custom Dataset for a Single Video Segment
class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, cfg):
        self.frames = frames  # Frames for a single video segment
        self.video_id = video_id
        self.cfg = cfg

        # Define transform to match SlowFast input requirements
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],  # SlowFast default normalization
                std=[0.225, 0.225, 0.225]
            )
        ])

    def __len__(self):
        return 1  # Only one sample (the current video segment)

    def __getitem__(self, idx):
        # Determine number of frames for slow and fast pathways
        num_frames_fast = self.cfg.DATA.NUM_FRAMES  # e.g., 8 for SlowFast 8x8
        alpha = self.cfg.SLOWFAST.ALPHA  # Typically 4
        num_frames_slow = num_frames_fast // alpha  # e.g., 8 // 4 = 2
        print(f"num_frames_fast: {num_frames_fast}, num_frames_slow: {num_frames_slow}")  # Debug

        # Sample frames for fast pathway (dense sampling)
        if len(self.frames) < num_frames_fast:
            fast_frames = self.frames + [self.frames[-1]] * (num_frames_fast - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_fast)
            fast_frames = [self.frames[i] for i in range(0, len(self.frames), step)][:num_frames_fast]

        # Sample frames for slow pathway (sparse sampling)
        if len(self.frames) < num_frames_slow:
            slow_frames = self.frames + [self.frames[-1]] * (num_frames_slow - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_slow)
            slow_frames = [self.frames[i] for i in range(0, len(self.frames), step)][:num_frames_slow]

        # Process fast pathway frames
        fast_processed = []
        fast_original = []  # Store original frames for visualization
        for frame in fast_frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.shape[0] == 0 or frame.shape[1] == 0 or frame.ndim != 3 or frame.shape[2] != 3:
                print(f"Warning: Invalid frame dimensions {frame.shape} for video {self.video_id}, skipping frame")
                continue

            pil_img = Image.fromarray(frame)
            fast_original.append(frame)  # Store original frame for visualization
            fast_processed.append(self.transform(pil_img))

        # Process slow pathway frames
        slow_processed = []
        slow_original = []  # Store original frames for visualization
        for frame in slow_frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.shape[0] == 0 or frame.shape[1] == 0 or frame.ndim != 3 or frame.shape[2] != 3:
                print(f"Warning: Invalid frame dimensions {frame.shape} for video {self.video_id}, skipping frame")
                continue

            pil_img = Image.fromarray(frame)
            slow_original.append(frame)  # Store original frame for visualization
            slow_processed.append(self.transform(pil_img))

        # Stack to create dual-pathway tensor (B, 2, C, T, H, W)
        if len(fast_processed) == 0 or len(slow_processed) == 0:
            print(f"Warning: No valid frames for video {self.video_id}, using dummy tensors")
            fast_tensor = torch.zeros((3, num_frames_fast, 224, 224))
            slow_tensor = torch.zeros((3, num_frames_slow, 224, 224))
            fast_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_fast
            slow_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_slow
        else:
            fast_tensor = torch.stack(fast_processed, dim=0)  # (T, C, H, W)
            slow_tensor = torch.stack(slow_processed, dim=0)  # (T, C, H, W)
            # Pad or truncate to match expected lengths
            if fast_tensor.shape[0] < num_frames_fast:
                fast_tensor = torch.cat([fast_tensor, fast_tensor[-1:].repeat(num_frames_fast - fast_tensor.shape[0], 1, 1, 1)], dim=0)
                fast_original.extend([fast_original[-1]] * (num_frames_fast - len(fast_original)))
            if slow_tensor.shape[0] < num_frames_slow:
                slow_tensor = torch.cat([slow_tensor, slow_tensor[-1:].repeat(num_frames_slow - slow_tensor.shape[0], 1, 1, 1)], dim=0)
                slow_original.extend([slow_original[-1]] * (num_frames_slow - len(slow_original)))
            fast_tensor = fast_tensor[:num_frames_fast]
            slow_tensor = slow_tensor[:num_frames_slow]
            fast_original = fast_original[:num_frames_fast]
            slow_original = slow_original[:num_frames_slow]
            # Transpose to (C, T, H, W)
            fast_tensor = fast_tensor.permute(1, 0, 2, 3)
            slow_tensor = slow_tensor.permute(1, 0, 2, 3)

        print(f"fast_tensor shape: {fast_tensor.shape}, slow_tensor shape: {slow_tensor.shape}")  # Debug

        # Create list of tensors for SlowFast input
        video_tensor = [slow_tensor, fast_tensor]  # List of tensors for slow and fast pathways

        return video_tensor, self.video_id, slow_original, fast_original

def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds

def extract_feature_maps(model, target_layer_slow, target_layer_fast, inputs):
    """
    Extract feature maps from the target layers of the SlowFast model.
    Args:
        model: The SlowFast model.
        target_layer_slow: Target layer for the slow pathway.
        target_layer_fast: Target layer for the fast pathway.
        inputs: Input tensors for the model.
    Returns:
        dict: Feature maps for slow and fast pathways.
    """
    activations = {'slow': None, 'fast': None}

    # Define hooks to capture feature maps
    def save_activation(pathway):
        def hook(module, input, output):
            activations[pathway] = output
        return hook

    # Register hooks
    handle_slow = target_layer_slow.register_forward_hook(save_activation('slow'))
    handle_fast = target_layer_fast.register_forward_hook(save_activation('fast'))

    # Forward pass
    with torch.no_grad():
        model.eval()
        preds, feat = model(inputs)

    # Remove hooks
    handle_slow.remove()
    handle_fast.remove()

    # Process feature maps
    feature_maps = {}
    for pathway in ['slow', 'fast']:
        feature_map = activations[pathway].data.cpu().numpy()  # Shape: (B, C, T, H, W)
        # Sum across channels to preserve spatial information
        feature_map = np.sum(feature_map, axis=1)  # Shape: (B, T, H, W)
        feature_map = np.maximum(feature_map, 0)  # ReLU-like operation
        # Normalize each frame's feature map independently
        for t in range(feature_map.shape[1]):  # Iterate over T dimension
            frame_map = feature_map[0, t]
            max_val = np.max(frame_map) + 1e-8
            if max_val > 0:
                feature_map[0, t] = frame_map / max_val  # Normalize to [0, 1]
        feature_maps[pathway] = feature_map

    return feature_maps

def visualize_feature_maps(feature_maps, slow_frames, fast_frames, video_id, output_dir):
    """
    Visualize feature maps and original frames in separate subplots.
    Args:
        feature_maps (dict): Feature maps for slow and fast pathways.
        slow_frames (list): Original frames for the slow pathway.
        fast_frames (list): Original frames for the fast pathway.
        video_id (str): Video ID for naming the output file.
        output_dir (str): Directory to save the visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Visualize slow pathway
    for t in range(len(slow_frames)):
        plt.figure(figsize=(10, 5))
        
        # Original frame subplot
        plt.subplot(1, 2, 1)
        frame = np.squeeze(slow_frames[t])
        plt.imshow(np.squeeze(frame))
        plt.title(f"Slow Frame {t+1}")
        plt.axis('off')

        # Feature map subplot
        plt.subplot(1, 2, 2)
        heatmap = feature_maps['slow'][0, t]  # Shape: (H, W)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))  # Upsample to match frame resolution
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Slow Feature Map {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_slow_frame_{t+1}.png"))
        plt.close()

    # Visualize fast pathway
    for t in range(len(fast_frames)):
        plt.figure(figsize=(10, 5))
        
        # Original frame subplot
        plt.subplot(1, 2, 1)
        frame = np.squeeze(fast_frames[t])
        plt.imshow(np.squeeze(frame))
        plt.title(f"Fast Frame {t+1}")
        plt.axis('off')

        # Feature map subplot
        plt.subplot(1, 2, 2)
        heatmap = feature_maps['fast'][0, t]  # Shape: (H, W)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))  # Upsample to match frame resolution
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Fast Feature Map {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_fast_frame_{t+1}.png"))
        plt.close()

@torch.no_grad()
def perform_inference(test_loader, model, cfg):
    """
    Perform feature extraction for a single video segment using the SlowFast model with feature map visualization.
    Args:
        test_loader (loader): DataLoader for the single video segment.
        model (model): The pretrained SlowFast model.
        cfg (CfgNode): Configs.
    """
    # Identify target layers for feature map extraction (use earlier layers for higher spatial resolution)
    target_layer_slow = model.s3.pathway0_res2  # Earlier layer for slow pathway (e.g., 28x28 spatial resolution)
    target_layer_fast = model.s3.pathway1_res2  # Earlier layer for fast pathway (e.g., 28x28 spatial resolution)

    for inputs, video_id, slow_frames, fast_frames in test_loader:
        print(f"Input types: {type(inputs)}")  # Debug: Check if inputs is a list
        print(f"Input slow shape: {inputs[0].shape}, fast shape: {inputs[1].shape}")  # Debug
        print(f"Input device before move: {inputs[0].device}")  # Debug: Check input device
        # Transfer the data to the current GPU device
        inputs = [inp.cuda(non_blocking=True) for inp in inputs]
        print(f"Input device after move: {inputs[0].device}")  # Debug: Verify input device

        # Extract feature maps
        feature_maps = extract_feature_maps(model, target_layer_slow, target_layer_fast, inputs)

        # Debug: Print feature map shapes
        print(f"Slow feature map shape: {feature_maps['slow'].shape}")
        print(f"Fast feature map shape: {feature_maps['fast'].shape}")

        # Visualize feature maps
        feature_map_dir = os.path.join(cfg.OUTPUT_DIR, "feature_maps")
        visualize_feature_maps(feature_maps, slow_frames, fast_frames, video_id[0], feature_map_dir)

        # Perform the forward pass for feature extraction
        model.eval()
        preds, feat = model(inputs)

        # Gather features across all devices
        if cfg.NUM_GPUS > 1:
            preds, feat = du.all_gather([preds, feat])

        feat = feat.cpu().numpy()

        # Save features for this video
        out_path = os.path.join(cfg.OUTPUT_DIR, "features")
        os.makedirs(out_path, exist_ok=True)
        out_file = f"{video_id[0]}_slowfast_features.npy"
        np.save(os.path.join(out_path, out_file), feat)
        print(f"Saved features for {video_id[0]} to {os.path.join(out_path, out_file)}")

        # Clear memory
        del inputs, preds, feat
        torch.cuda.empty_cache()

def test(cfg):
    """
    Perform feature extraction on video segments one at a time using the pretrained SlowFast model.
    Args:
        cfg (CfgNode): Configs.
    """
    # Set random seed from configs
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config
    logger.info("Test with config:")
    logger.info(cfg)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Debug: Confirm device
    model = build_model(cfg)
    model = model.to(device)  # Explicitly move model to the detected device
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    # Load pretrained checkpoint
    cu.load_test_checkpoint(cfg, model)

    # Directory containing pickle files
    pickle_dir = "D:/pickle_dir/fine_tune"
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))

    print(f"Found {len(pickle_files)} pickle files to process.")
    print("----------------------------------------------------------")

    start_time = time.time()
    processed_segments = 0

    # Process one video segment at a time
    for pkl_file in tqdm(pickle_files, desc="Processing pickle files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Iterate through segments in the pickle file
        for camera_id in data:
            if camera_id == 'cam3':
                for segments_group in data[camera_id]:
                    for segment in segments_group:
                        # Filter segments based on labels (only 2 and 3)
                        if 'segment_ratings' in segment:
                            rating = segment['segment_ratings'].get('t1', None)
                            try:
                                rating = int(rating)
                                if rating not in [2, 3]:  # Only include labels 2 and 3
                                    continue
                            except (ValueError, TypeError):
                                print(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                                continue
                        else:
                            print(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")
                            continue

                        # Create video ID
                        video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                                    f"{segment['CameraId']}_seg_{segment['segment_id']}")

                        # Check if features already exist
                        out_path = os.path.join(cfg.OUTPUT_DIR, "features")
                        out_file = f"{video_id}_slowfast_features.npy"
                        if os.path.exists(os.path.join(out_path, out_file)):
                            print(f"Features for {video_id} already exist, skipping...")
                            continue

                        print(f"Processing segment: {video_id}")

                        # Create dataset for this single segment
                        dataset = SinglePickleFrameDataset(
                            frames=segment['frames'],
                            video_id=video_id,
                            cfg=cfg
                        )

                        # Create DataLoader for this single segment
                        test_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=1,  # Process one segment at a time
                            shuffle=False,
                            num_workers=0,  # Set to 0 to avoid worker issues
                            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                            drop_last=False,
                        )

                        # Perform feature extraction for this segment with feature map visualization
                        perform_inference(test_loader, model, cfg)
                        processed_segments += 1

                        # Clear memory after processing each segment
                        del dataset, test_loader
                        torch.cuda.empty_cache()

    end_time = time.time()
    hours, minutes, seconds = calculate_time_taken(start_time, end_time)
    print(f"Processed {processed_segments} video segments.")
    print(
        "Time taken: {} hour(s), {} minute(s) and {} second(s)".format(
            hours, minutes, seconds
        )
    )
    print("----------------------------------------------------------")

def main():
    """
    Main function to spawn the test process.
    """
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_files[0])  # Load the config file

    # Perform feature extraction
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()