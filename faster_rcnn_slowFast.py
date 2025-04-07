#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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

def load_bboxes(bbox_dir, video_id):
    """
    Load bounding box data for a video segment.
    Args:
        bbox_dir (str): Directory containing bounding box files.
        video_id (str): Video identifier (e.g., 'patient_1_task_1_cam3_seg_0').
    Returns:
        list: List of bounding boxes, each an array of shape (4,) [x_min, y_min, x_max, y_max].
              Returns None if file not found or data is invalid.
    """
    bbox_file = os.path.join(bbox_dir, f"{video_id}_bboxes.pkl")
    if not os.path.exists(bbox_file):
        print(f"No bounding box file found for {video_id} at {bbox_file}")
        return None
    
    with open(bbox_file, 'rb') as f:
        bboxes = pickle.load(f)
    
    # Validate the loaded data
    if not isinstance(bboxes, list) or not bboxes:
        print(f"Invalid bounding box data for {video_id}: expected a non-empty list, got {type(bboxes)}")
        return None
    
    # Check each entry is an array with 4 values
    for i, box in enumerate(bboxes):
        if not isinstance(box, (list, np.ndarray)) or len(box) != 4:
            print(f"Invalid bounding box at frame {i} for {video_id}: expected 4 values, got {box}")
            return None
    
    # Ensure length matches expected frame count (optional, can remove if frame count varies)
    # if len(bboxes) != 20:
    #     print(f"Warning: Expected 20 frames for {video_id}, got {len(bboxes)} bounding boxes")
    
    return bboxes

# Custom Dataset with Cropped Frames for Inference and Visualization
class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, cfg, bbox_dir):
        self.frames = frames
        self.video_id = video_id
        self.cfg = cfg
        self.bbox_dir = bbox_dir
        self.bboxes = load_bboxes(bbox_dir, video_id)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        num_frames_fast = self.cfg.DATA.NUM_FRAMES  # e.g., 8
        alpha = self.cfg.SLOWFAST.ALPHA  # e.g., 4
        num_frames_slow = num_frames_fast // alpha  # e.g., 2
        print(f"num_frames_fast: {num_frames_fast}, num_frames_slow: {num_frames_slow}")

        # Sample indices
        if len(self.frames) < num_frames_fast:
            fast_indices = list(range(len(self.frames))) + [len(self.frames) - 1] * (num_frames_fast - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_fast)
            fast_indices = [i for i in range(0, len(self.frames), step)][:num_frames_fast]

        if len(self.frames) < num_frames_slow:
            slow_indices = list(range(len(self.frames))) + [len(self.frames) - 1] * (num_frames_slow - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_slow)
            slow_indices = [i for i in range(0, len(self.frames), step)][:num_frames_slow]

        # Helper function to crop frame with extended lower side
        def crop_frame(frame, bbox):
            if bbox is None or len(bbox) != 4:
                print(f"No valid bbox for frame in {self.video_id}, using full frame")
                return frame
            x_min, y_min, x_max, y_max = map(int, bbox)  # Original coordinates
            h, w = frame.shape[:2]
            
            # Extend y_max by 30 pixels
            y_max_extended = y_max + 30
            
            # Clamp coordinates to frame boundaries
            x_min, x_max = max(0, x_min), min(w, x_max)
            y_min, y_max_extended = max(0, y_min), min(h, y_max_extended)
            
            # Check for valid crop region
            if x_max <= x_min or y_max_extended <= y_min:
                print(f"Invalid crop coordinates [{x_min}, {y_min}, {x_max}, {y_max_extended}] for {self.video_id}, using full frame")
                return frame
            
            return frame[y_min:y_max_extended, x_min:x_max]
        # Process frames with cropping for inference and visualization
        fast_processed = []
        fast_original = []
        slow_processed = []
        slow_original = []

        # Fast pathway
        for i in fast_indices:
            frame = self.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                print(f"Warning: Invalid frame dimensions {frame.shape} for {self.video_id}, using dummy")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            # Crop frame before transformation
            bbox = self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None
            cropped_frame = crop_frame(frame, bbox)
            pil_img = Image.fromarray(cropped_frame)
            fast_processed.append(self.transform(pil_img))  # For inference
            fast_original.append(cropped_frame)  # For visualization

        # Slow pathway
        for i in slow_indices:
            frame = self.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                print(f"Warning: Invalid frame dimensions {frame.shape} for {self.video_id}, using dummy")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            # Crop frame before transformation
            bbox = self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None
            cropped_frame = crop_frame(frame, bbox)
            pil_img = Image.fromarray(cropped_frame)
            slow_processed.append(self.transform(pil_img))  # For inference
            slow_original.append(cropped_frame)  # For visualization

        # Stack tensors
        if not fast_processed or not slow_processed:
            print(f"Warning: No valid frames for {self.video_id}, using dummy tensors")
            fast_tensor = torch.zeros((3, num_frames_fast, 224, 224))
            slow_tensor = torch.zeros((3, num_frames_slow, 224, 224))
            fast_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_fast
            slow_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_slow
        else:
            fast_tensor = torch.stack(fast_processed, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]
            slow_tensor = torch.stack(slow_processed, dim=0).permute(1, 0, 2, 3)  # [C, T, H, W]
            if fast_tensor.shape[1] < num_frames_fast:
                fast_tensor = torch.cat([fast_tensor, fast_tensor[:, -1:].repeat(1, num_frames_fast - fast_tensor.shape[1], 1, 1)], dim=1)
                fast_original.extend([fast_original[-1]] * (num_frames_fast - len(fast_original)))
            if slow_tensor.shape[1] < num_frames_slow:
                slow_tensor = torch.cat([slow_tensor, slow_tensor[:, -1:].repeat(1, num_frames_slow - slow_tensor.shape[1], 1, 1)], dim=1)
                slow_original.extend([slow_original[-1]] * (num_frames_slow - len(slow_original)))
            fast_tensor = fast_tensor[:, :num_frames_fast]
            slow_tensor = slow_tensor[:, :num_frames_slow]
            fast_original = fast_original[:num_frames_fast]
            slow_original = slow_original[:num_frames_slow]

        print(f"fast_tensor shape: {fast_tensor.shape}, slow_tensor shape: {slow_tensor.shape}")
        video_tensor = [slow_tensor, fast_tensor]
        return video_tensor, self.video_id, slow_original, fast_original

def calculate_time_taken(start_time, end_time):
    hours = int((end_time - start_time) / 3600)
    minutes = int((end_time - start_time) / 60) - (hours * 60)
    seconds = int((end_time - start_time) % 60)
    return hours, minutes, seconds

def extract_feature_maps(model, target_layer_slow, target_layer_fast, inputs):
    activations = {'slow': None, 'fast': None}
    def save_activation(pathway):
        def hook(module, input, output):
            activations[pathway] = output
        return hook
    handle_slow = target_layer_slow.register_forward_hook(save_activation('slow'))
    handle_fast = target_layer_fast.register_forward_hook(save_activation('fast'))
    with torch.no_grad():
        model.eval()
        preds, feat = model(inputs)
    handle_slow.remove()
    handle_fast.remove()
    feature_maps = {}
    for pathway in ['slow', 'fast']:
        feature_map = activations[pathway].data.cpu().numpy()
        feature_map = np.sum(feature_map, axis=1)  # [B, T, H, W]
        feature_map = np.maximum(feature_map, 0)
        for t in range(feature_map.shape[1]):
            frame_map = feature_map[0, t]
            max_val = np.max(frame_map) + 1e-8
            if max_val > 0:
                feature_map[0, t] = frame_map / max_val
        feature_maps[pathway] = feature_map
    return feature_maps

def visualize_feature_maps(feature_maps, slow_frames, fast_frames, video_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for t in range(len(slow_frames)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(slow_frames[t])
        plt.imshow(np.squeeze(frame))
        plt.title(f"Slow Cropped Frame {t+1}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        heatmap = feature_maps['slow'][0, t]
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Slow Feature Map {t+1}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_slow_frame_{t+1}.png"))
        plt.close()
    for t in range(len(fast_frames)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(fast_frames[t])
        plt.imshow(np.squeeze(frame))
        plt.title(f"Fast Cropped Frame {t+1}")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        heatmap = feature_maps['fast'][0, t]
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        plt.imshow(heatmap, cmap='jet')
        plt.title(f"Fast Feature Map {t+1}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_fast_frame_{t+1}.png"))
        plt.close()

@torch.no_grad()
def perform_inference(test_loader, model, cfg):
    target_layer_slow = model.s3.pathway0_res2
    target_layer_fast = model.s3.pathway1_res2
    for inputs, video_id, slow_frames, fast_frames in test_loader:
        print(f"Input slow shape: {inputs[0].shape}, fast shape: {inputs[1].shape}")
        inputs = [inp.cuda(non_blocking=True) for inp in inputs]
        feature_maps = extract_feature_maps(model, target_layer_slow, target_layer_fast, inputs)
        print(f"Slow feature map shape: {feature_maps['slow'].shape}")
        print(f"Fast feature map shape: {feature_maps['fast'].shape}")
        feature_map_dir = os.path.join(cfg.OUTPUT_DIR, "feature_maps")
        visualize_feature_maps(feature_maps, slow_frames, fast_frames, video_id[0], feature_map_dir)
        model.eval()
        preds, feat = model(inputs)
        if cfg.NUM_GPUS > 1:
            preds, feat = du.all_gather([preds, feat])
        feat = feat.cpu().numpy()
        out_path = os.path.join(cfg.OUTPUT_DIR, "features")
        os.makedirs(out_path, exist_ok=True)
        out_file = f"{video_id[0]}_slowfast_features.npy"
        np.save(os.path.join(out_path, out_file), feat)
        print(f"Saved features for {video_id[0]} to {os.path.join(out_path, out_file)}")
        del inputs, preds, feat
        torch.cuda.empty_cache()

def test(cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Test with config:")
    logger.info(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = build_model(cfg)
    model = model.to(device)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    cu.load_test_checkpoint(cfg, model)

    pickle_dir = "D:/pickle_dir/fine_tune"
    bbox_dir = "D:/frcnn_bboxes/bboxes_top"
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))

    print(f"Found {len(pickle_files)} pickle files to process.")
    print("----------------------------------------------------------")

    start_time = time.time()
    processed_segments = 0

    for pkl_file in tqdm(pickle_files, desc="Processing pickle files"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        for camera_id in data:
            if camera_id == 'cam3':
                for segments_group in data[camera_id]:
                    for segment in segments_group:
                        if 'segment_ratings' in segment:
                            rating = segment['segment_ratings'].get('t1', None)
                            try:
                                rating = int(rating)
                                if rating not in [2, 3]:
                                    continue
                            except (ValueError, TypeError):
                                print(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                                continue
                        else:
                            print(f"Skipping segment in {pkl_file} with missing 'segment_ratings'")
                            continue

                        video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                                    f"{segment['CameraId']}_seg_{segment['segment_id']}")

                        out_path = os.path.join(cfg.OUTPUT_DIR, "features")
                        out_file = f"{video_id}_slowfast_features.npy"
                        if os.path.exists(os.path.join(out_path, out_file)):
                            print(f"Features for {video_id} already exist, skipping...")
                            continue

                        print(f"Processing segment: {video_id}")

                        dataset = SinglePickleFrameDataset(
                            frames=segment['frames'],
                            video_id=video_id,
                            cfg=cfg,
                            bbox_dir=bbox_dir
                        )

                        test_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                            drop_last=False,
                        )

                        perform_inference(test_loader, model, cfg)
                        processed_segments += 1

                        del dataset, test_loader
                        torch.cuda.empty_cache()

    end_time = time.time()
    hours, minutes, seconds = calculate_time_taken(start_time, end_time)
    print(f"Processed {processed_segments} video segments.")
    print(f"Time taken: {hours} hour(s), {minutes} minute(s) and {seconds} second(s)")
    print("----------------------------------------------------------")

def main():
    args = parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_files[0])
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

if __name__ == "__main__":
    main()