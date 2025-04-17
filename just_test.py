#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import torch
import os
import glob
import pickle
from PIL import Image
import torchvision.transforms as transforms
import warnings
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
# Suppress specific warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms._transforms_video")

# SlowFast imports
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
    """
    bbox_file = os.path.join(bbox_dir, f"{video_id}_bboxes.pkl")
    if not os.path.exists(bbox_file):
        logger.warning(f"No bounding box file found for {video_id} at {bbox_file}")
        return None

    try:
        with open(bbox_file, 'rb') as f:
            bboxes = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading bounding box file for {video_id}: {e}")
        return None

    if not isinstance(bboxes, list) or not bboxes:
        logger.error(f"Invalid bounding box data for {video_id}: expected a non-empty list, got {type(bboxes)}")
        return None

    for i, box in enumerate(bboxes):
        if not isinstance(box, (list, np.ndarray)) or len(box) != 4:
            logger.error(f"Invalid bounding box at frame {i} for {video_id}: expected 4 values, got {box}")
            return None

    return bboxes


class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, label, cfg, bbox_dir):
        self.frames = frames
        self.video_id = video_id
        self.label = label
        self.cfg = cfg
        self.bbox_dir = bbox_dir
        self.bboxes = load_bboxes(bbox_dir, video_id)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        num_frames_fast = self.cfg.DATA.NUM_FRAMES
        alpha = self.cfg.SLOWFAST.ALPHA
        num_frames_slow = num_frames_fast // alpha

        # Sample indices for fast pathway
        if len(self.frames) < num_frames_fast:
            fast_indices = list(range(len(self.frames))) + [len(self.frames) - 1] * (num_frames_fast - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_fast)
            fast_indices = [i for i in range(0, len(self.frames), step)][:num_frames_fast]

        # Sample indices for slow pathway
        if len(self.frames) < num_frames_slow:
            slow_indices = list(range(len(self.frames))) + [len(self.frames) - 1] * (num_frames_slow - len(self.frames))
        else:
            step = max(1, len(self.frames) // num_frames_slow)
            slow_indices = [i for i in range(0, len(self.frames), step)][:num_frames_slow]

        def crop_frame(frame, bbox):
            if bbox is None or len(bbox) != 4:
                return frame
            x_min, y_min, x_max, y_max = map(int, bbox)
            h, w = frame.shape[:2]
            y_max_ext = min(h, y_max + 30)
            x_min, x_max = max(0, x_min), min(w, x_max)
            if x_max <= x_min or y_max_ext <= y_min:
                return frame
            return frame[y_min:y_max_ext, x_min:x_max]

        fast_processed, fast_original = [], []
        slow_processed, slow_original = [], []

        # Process fast frames
        for i in fast_indices:
            frame = self.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            bbox = self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None
            cropped = crop_frame(frame, bbox)
            fast_original.append(cropped)
            fast_processed.append(self.transform(Image.fromarray(cropped)))

        # Process slow frames
        for i in slow_indices:
            frame = self.frames[i]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).numpy()
            elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame.squeeze(0)

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            if frame.ndim != 3 or frame.shape[2] != 3:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)

            bbox = self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None
            cropped = crop_frame(frame, bbox)
            slow_original.append(cropped)
            slow_processed.append(self.transform(Image.fromarray(cropped)))

        if fast_processed and slow_processed:
            fast_tensor = torch.stack(fast_processed, dim=0).permute(1, 0, 2, 3)
            slow_tensor = torch.stack(slow_processed, dim=0).permute(1, 0, 2, 3)
        else:
            fast_tensor = torch.zeros((3, num_frames_fast, 224, 224))
            slow_tensor = torch.zeros((3, num_frames_slow, 224, 224))
            fast_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_fast
            slow_original = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames_slow

        return [slow_tensor, fast_tensor], self.video_id, slow_original, fast_original, torch.tensor(self.label, dtype=torch.long)


def fastslow_collate_fn(batch):
    """
    Custom collate_fn: stacks only the Slow/Fast tensors & labels,
    leaving raw frames & IDs as lists.
    """
    slow_batch = torch.stack([item[0][0] for item in batch], dim=0)
    fast_batch = torch.stack([item[0][1] for item in batch], dim=0)
    inputs = [slow_batch, fast_batch]

    video_ids = [item[1] for item in batch]
    slow_raw = [item[2] for item in batch]
    fast_raw = [item[3] for item in batch]
    labels = torch.stack([item[4] for item in batch], dim=0)

    return inputs, video_ids, slow_raw, fast_raw, labels


def modify_slowfast_head(model, num_classes, device):
    in_features = model.head.projection.in_features
    model.head.projection = torch.nn.Linear(in_features, num_classes).to(device)
    logger.info(f"Modified SlowFast head to output {num_classes} classes on device {device}")
    return model


# Grad-CAM implementation for SlowFast
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self.save_activations))
        self.hook_handles.append(target_layer.register_backward_hook(self.save_gradients))

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, inputs, target_class=None):
        """
        Compute Grad-CAM heatmap for the target class.
        Args:
            inputs: Input tensors [slow_tensor, fast_tensor].
            target_class: Target class index for Grad-CAM. If None, use the predicted class.
        Returns:
            heatmap: Grad-CAM heatmap of shape (T, H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        outputs, _ = self.model(inputs)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        # Backward pass to get gradients
        outputs[:, target_class].backward()

        # Compute Grad-CAM
        gradients = self.gradients  # Shape: (B, C, T, H, W)
        activations = self.activations  # Shape: (B, C, T, H, W)

        # Average gradients over the channel dimension
        weights = torch.mean(gradients, dim=[0, 3, 4], keepdim=True)  # Shape: (1, C, T, 1, 1)
        heatmap = torch.sum(weights * activations, dim=1).squeeze()  # Shape: (T, H, W)

        # Apply ReLU
        heatmap = torch.maximum(heatmap, torch.tensor(0.0, device=heatmap.device))

        # Normalize the heatmap
        for t in range(heatmap.shape[0]):
            frame_heatmap = heatmap[t]
            max_val = torch.max(frame_heatmap) + 1e-8
            min_val = torch.min(frame_heatmap)
            if max_val > min_val:
                heatmap[t] = (frame_heatmap - min_val) / (max_val - min_val)
            else:
                heatmap[t] = torch.zeros_like(frame_heatmap)

        return heatmap.detach().cpu().numpy()

    def __del__(self):
        # Remove hooks when the object is deleted
        for handle in self.hook_handles:
            handle.remove()

# Function to extract Grad-CAM heatmaps
def extract_gradcam_heatmaps(model, target_layer_slow, target_layer_fast, inputs):
    """
    Extract Grad-CAM heatmaps from specified layers of the SlowFast model.
    Args:
        model: SlowFast model.
        target_layer_slow: Target layer in the slow pathway.
        target_layer_fast: Target layer in the fast pathway.
        inputs: Input tensors [slow_tensor, fast_tensor].
    Returns:
        dict: Grad-CAM heatmaps for slow and fast pathways.
    """
    gradcam_slow = GradCAM(model, target_layer_slow)
    gradcam_fast = GradCAM(model, target_layer_fast)

    # Compute Grad-CAM heatmaps
    heatmap_slow = gradcam_slow(inputs)  # Shape: (T_slow, H, W)
    heatmap_fast = gradcam_fast(inputs)  # Shape: (T_fast, H, W)

    return {'slow': heatmap_slow, 'fast': heatmap_fast}

# Function to visualize Grad-CAM heatmaps
def visualize_gradcam_heatmaps(heatmaps, slow_frames, fast_frames, video_id, output_dir):
    """
    Visualize Grad-CAM heatmaps alongside cropped frames.
    Args:
        heatmaps (dict): Grad-CAM heatmaps for slow and fast pathways.
        slow_frames (list): Original slow pathway frames.
        fast_frames (list): Original fast pathway frames.
        video_id (str): Video identifier.
        output_dir (str): Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Visualize slow pathway
    for t in range(len(slow_frames)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(slow_frames[t])
        plt.imshow(frame)
        plt.title(f"Slow Cropped Frame {t+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        heatmap = heatmaps['slow'][t]  # Shape: (H, W)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smooth to reduce noise
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.imshow(frame, alpha=0.5)  # Overlay the heatmap on the original frame
        plt.title(f"Slow Grad-CAM {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_slow_gradcam_{t+1}.png"))
        plt.close()

    # Visualize fast pathway
    for t in range(len(fast_frames)):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = np.squeeze(fast_frames[t])
        plt.imshow(frame)
        plt.title(f"Fast Cropped Frame {t+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        heatmap = heatmaps['fast'][t]  # Shape: (H, W)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smooth to reduce noise
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.imshow(frame, alpha=0.5)  # Overlay the heatmap on the original frame
        plt.title(f"Fast Grad-CAM {t+1}")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{video_id}_fast_gradcam_{t+1}.png"))
        plt.close()

def perform_inference(test_loader, model, cfg):
    """
    Run inference + Grad‑CAM on test_loader.
    We only disable grad where it’s not needed.
    """
    model.eval()

    layers_to_test = {
        'low':  {'slow': model.s2.pathway0_res0, 'fast': model.s2.pathway1_res0},
        'mid':  {'slow': model.s3.pathway0_res2, 'fast': model.s3.pathway1_res2},
        'deep': {'slow': model.s5.pathway0_res0, 'fast': model.s5.pathway1_res0},
    }

    for inputs, video_ids, slow_raw_list, fast_raw_list, labels in test_loader:
        vid = video_ids[0]
        # each of these is a list-of-arrays: we only want the inner list
        slow_frames = slow_raw_list[0]
        fast_frames = fast_raw_list[0]

        for depth, layers in layers_to_test.items():
            heatmaps = extract_gradcam_heatmaps(model, layers['slow'], layers['fast'], inputs)
            gradcam_dir = os.path.join(cfg.OUTPUT_DIR, f"gradcam_{depth}")
            visualize_gradcam_heatmaps(heatmaps, slow_frames, fast_frames, vid, gradcam_dir)


        # 2) Now disable grads while saving features:
        with torch.no_grad():
            preds, feat = model(inputs)
            if cfg.NUM_GPUS > 1:
                preds, feat = du.all_gather([preds, feat])
            feat = feat.cpu().numpy()
            out_path = os.path.join(cfg.OUTPUT_DIR, "features")
            os.makedirs(out_path, exist_ok=True)
            out_file = f"{video_id[0]}_slowfast_features.npy"
            np.save(os.path.join(out_path, out_file), feat)
            logger.info(f"Saved features for {video_id[0]} to {out_path}")
        torch.cuda.empty_cache()


def test(cfg):
    """
    - Collects EVERY segment (no train/val split)
    - Builds a batch_size=1 DataLoader over all segments
    - Loads your fine-tuned checkpoint
    - Runs perform_inference over the full set
    """
    # 1) Collect all segments exactly as before:
    pickle_dir = "D:/pickle_dir/fine_tune"
    bbox_dir   = "D:/frcnn_bboxes/bboxes_top"
    all_segments = []
    for pkl_file in glob.glob(os.path.join(pickle_dir, '*.pkl')):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {pkl_file}: {e}")
            continue

        for camera_id in data:
            if camera_id != 'cam3':
                continue
            for segments_group in data[camera_id]:
                for segment in segments_group:
                    if 'segment_ratings' not in segment:
                        continue
                    rating = segment['segment_ratings'].get('t1', None)
                    try:
                        rating = int(rating)
                        if rating not in [2, 3]:
                            continue
                    except:
                        continue

                    label = 0 if rating == 2 else 1
                    vid = (f"patient_{segment['patient_id']}_"
                           f"task_{segment['activity_id']}_"
                           f"{segment['CameraId']}_seg_{segment['segment_id']}")
                    all_segments.append({
                        'frames':    segment['frames'],
                        'video_id':  vid,
                        'label':     label
                    })

    # 2) Build full dataset + DataLoader (batch_size=1 for GradCAM)
    full_ds = torch.utils.data.ConcatDataset([
        SinglePickleFrameDataset(
            frames=seg['frames'],
            video_id=seg['video_id'],
            label=seg['label'],
            cfg=cfg,
            bbox_dir=bbox_dir
        ) for seg in all_segments
    ])
    full_loader = torch.utils.data.DataLoader(
        full_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn=fastslow_collate_fn
    )

    # 3) Build & load your fine-tuned model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model = modify_slowfast_head(model, num_classes=2, device=device)

    ckpt = os.path.join(cfg.OUTPUT_DIR, "slowfast_finetuned.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # 4) Run inference (Grad-CAM + feature save) on the full dataset
    perform_inference(full_loader, model, cfg)


def main():
    args = parse_args()
    cfg  = get_cfg()
    cfg.merge_from_file(args.cfg_files[0])

    # Make sure your config knows these:
    cfg.PICKLE_DIR = "D:/pickle_dir/fine_tune"
    cfg.BBOX_DIR   = "D:/frcnn_bboxes/bboxes_top"

    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
