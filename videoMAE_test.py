#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# Configuration flag:
# 0 = do fine-tuning and then testing
# 1 = load saved fine-tuned model and run testing only
USE_PRETRAINED_FINETUNED = 0
CLASS_IMBALANCE = 1  # 1 to enable class weight balancing
BALANCED_SAMPLING = 1  # 1 to enable balanced batch sampling
SAVE_HEATMAPS_FEATURES = 1  # 1 to save attention maps and features
num_class = 3  # set to 3 classes (0, 1, 2)

import numpy as np
import torch
import torch.nn as nn
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
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
from skimage.transform import resize
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, VideoMAEConfig

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class Config:
    def __init__(self):
        self.OUTPUT_DIR = "D:\\nature_everything\\output_videomae"
        self.DATA_LOADER_PIN_MEMORY = True
        self.NUM_WORKERS = 0
        self.TRAIN_BATCH_SIZE = 4
        self.TEST_BATCH_SIZE = 4
        self.NUM_FRAMES = 16  # VideoMAE standard
        self.SAMPLING_RATE = 4 # Stride
        self.RNG_SEED = 42

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Path to the CSV file with camera assignments
ipsi_contra_csv = "D:\\nature_everything\\camera_assignments.csv"
camera_box = "bboxes_ipsi"
_map = {1: 0, 2: 1, 3: 2}

def load_bboxes(bbox_dir, video_id):
    """ Load bounding box data for a video segment. """
    bbox_file = os.path.join(bbox_dir, f"{video_id}_bboxes.pkl")
    if not os.path.exists(bbox_file):
        return None
    try:
        with open(bbox_file, 'rb') as f:
            bboxes = pickle.load(f)
    except Exception as e:
        return None
    if not isinstance(bboxes, list) or not bboxes:
        return None
    return bboxes

class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, label, camera_id, cfg, bbox_dir, is_train=True, processor=None):
        self.frames = frames
        self.video_id = video_id
        self.label = label
        self.hand_id = camera_id
        self.cfg = cfg
        self.bbox_dir = bbox_dir
        self.is_train = is_train
        self.bboxes = load_bboxes(bbox_dir, video_id)
        
        self.processor = processor

        if is_train:
            self.spatial_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
            ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        target_frames = self.cfg.NUM_FRAMES
        stride = self.cfg.SAMPLING_RATE
        total_span = target_frames * stride
        
        # Temporal Sampling
        vlen = len(self.frames)
        if vlen >= total_span:
            if self.is_train:
                start_idx = np.random.randint(0, vlen - total_span + 1)
            else:
                start_idx = (vlen - total_span) // 2
            indices = np.arange(start_idx, start_idx + total_span, stride)
        else:
            indices = np.arange(0, vlen, stride)
            while len(indices) < target_frames:
                indices = np.concatenate([indices, [indices[-1]]])
            indices = indices[:target_frames]

        def crop_frame(frame, bbox):
            if bbox is None or len(bbox) != 4:
                return frame
            x_min, y_min, x_max, y_max = map(int, bbox)
            h, w = frame.shape[:2]
            
            if camera_box == 'bboxes_top':
                y_max_ext = min(h, y_max + 30)
                x_min, x_max = max(0, x_min - 10), min(w, x_max + 10)
                y_min = max(0, y_min)
            else:
                if self.hand_id == 'cam4':
                    y_max_ext = min(h, y_max - 20)
                    x_min, x_max = max(0, x_min - 30), min(w, x_max)
                    y_min = max(0, y_min)
                else:
                    y_max_ext = min(h, y_max - 20)
                    x_min, x_max = max(0, x_min), min(w, x_max + 30)
                    y_min = max(0, y_min)
            
            if x_max <= x_min or y_max_ext <= y_min:
                return frame
            return frame[y_min:y_max_ext, x_min:x_max]

        video_frames = []
        original_frames = []

        for i in indices:
            i = int(i)
            try:
                frame = self.frames[i]
                if isinstance(frame, torch.Tensor):
                    frame = frame.permute(1, 2, 0).numpy()
                elif isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                
                if frame.ndim != 3 or frame.shape[2] != 3:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                
                bbox = (self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None)
                cropped = crop_frame(frame, bbox)
                
                pil_img = Image.fromarray(cropped)
                transformed_img = self.spatial_transform(pil_img)
                
                video_frames.append(np.array(transformed_img))
                original_frames.append(cropped)
                
            except Exception as e:
                video_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                original_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        encoded_inputs = self.processor(list(video_frames), return_tensors="pt")
        pixel_values = encoded_inputs.pixel_values.squeeze(0)  # [16, 3, 224, 224]

        return pixel_values, self.video_id, original_frames, torch.tensor(self.label, dtype=torch.long)

def videomae_collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch], dim=0) 
    video_ids = [item[1] for item in batch]
    raw_frames = [item[2] for item in batch]
    labels = torch.stack([item[3] for item in batch], dim=0)
    return pixel_values, video_ids, raw_frames, labels

# --- HELPERS ---

def _safe_entropy_from_probs(p: np.ndarray) -> float:
    _EPS = 1e-12
    p = np.clip(p.astype(np.float64), _EPS, 1.0)
    return float(-(p * np.log(p)).sum())

def parse_video_id_umera(video_id: str) -> dict:
    parts = video_id.split("_")
    out = {
        "video_id": video_id,
        "patient_id": None, "activity_id": None, "camera_id": None,
        "segment_id": None, "task_key": None, "task_view_key": None,
    }
    try:
        pid = parts[1]
        activity = parts[3]
        cam = parts[4]
        out["patient_id"] = int(pid) if pid.isdigit() else pid
        out["activity_id"] = int(activity) if str(activity).isdigit() else activity
        out["camera_id"] = cam
        out["task_key"] = f"patient_{pid}_task_{activity}"
        out["task_view_key"] = f"patient_{pid}_task_{activity}_{cam}"
        if parts[-2] == "seg":
            out["segment_id"] = int(parts[-1])
    except:
        pass
    return out

def visualize_attention(model, pixel_values, original_frames, video_id, output_dir):
    # Visualization adapted for VideoMAE attention rollout
    with torch.no_grad():
        outputs = model(pixel_values.unsqueeze(0).to(model.device), output_attentions=True)
        last_attn = outputs.attentions[-1] 
        attn_map = torch.mean(last_attn, dim=1).squeeze(0) 
        cls_attn = attn_map[0, 1:] # Exclude CLS itself
        
        T = 16
        Grid = 14
        cls_attn = cls_attn.reshape(T, Grid, Grid)
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        cls_attn = cls_attn.cpu().numpy()
        
    os.makedirs(output_dir, exist_ok=True)
    
    for t in range(min(T, len(original_frames))):
        if t % 2 != 0: continue 
        
        frame = original_frames[t]
        attn = cls_attn[t]
        
        attn_resized = cv2.resize(attn, (frame.shape[1], frame.shape[0]))
        attn_heatmap = np.uint8(255 * attn_resized)
        attn_heatmap = cv2.applyColorMap(attn_heatmap, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(frame, 0.6, attn_heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, f"{video_id}_attn_{t}.jpg"), overlay)

# --- INFERENCE FUNCTION ---
@torch.no_grad()
def perform_inference(test_loader, model, cfg, inference_segments=None, export_for_controller=True):
    model.eval()
    device = model.device
    predictions = []
    controller_records = []
    
    logger.info("Starting Inference...")
    
    inference_map = {}
    if inference_segments:
        inference_map = {seg['video_id']: seg for seg in inference_segments}

    feat_save_dir = os.path.join(cfg.OUTPUT_DIR, "features_videomae")
    os.makedirs(feat_save_dir, exist_ok=True)
    
    for pixel_values, video_ids, raw_frames, labels in tqdm(test_loader, desc="Inference"):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        outputs = model(pixel_values, output_hidden_states=True)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        
        last_hidden = outputs.hidden_states[-1] 
        cls_tokens = last_hidden[:, 0, :] 
        
        batch_size = labels.size(0)
        
        for i in range(batch_size):
            vid = video_ids[i]
            pred_class = torch.argmax(probs[i]).item()
            conf = probs[i, pred_class].item()
            feat_vec = cls_tokens[i].cpu().numpy()
            
            if SAVE_HEATMAPS_FEATURES and i == 0: 
                vis_dir = os.path.join(cfg.OUTPUT_DIR, "vis_attention")
                visualize_attention(model, pixel_values[i], raw_frames[i], vid, vis_dir)

            meta = parse_video_id_umera(vid)
            t1_label = inference_map.get(vid, {}).get('t1_label')
            t2_label = inference_map.get(vid, {}).get('t2_label')

            record = {
                "video_id": vid,
                "patient_id": meta.get("patient_id"),
                "task_key": meta.get("task_key"),
                "pred": int(pred_class),
                "confidence": float(conf),
                "entropy": _safe_entropy_from_probs(probs[i].cpu().numpy()),
                "logits": logits[i].cpu().numpy(),
                "probs": probs[i].cpu().numpy(),
                "feat_global": feat_vec, 
                "t1_label": t1_label,
                "t2_label": t2_label
            }
            controller_records.append(record)
            
            np.save(os.path.join(feat_save_dir, f"{vid}_feat.npy"), feat_vec)

            is_correct = False
            if t1_label is not None and pred_class == t1_label: is_correct = True
            if t2_label is not None and pred_class == t2_label: is_correct = True
            
            predictions.append({
                'video_id': vid,
                'predicted': pred_class,
                'correct': is_correct,
                'confidence': conf
            })

    df = pd.DataFrame(predictions)
    df.to_csv(os.path.join(cfg.OUTPUT_DIR, "videomae_predictions.csv"), index=False)
    
    acc = df['correct'].mean()
    logger.info(f"Inference Accuracy: {acc:.4f}")

    if export_for_controller:
        pkl_path = os.path.join(cfg.OUTPUT_DIR, "videomae_controller_export.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"segment_records": controller_records}, f)
        logger.info(f"Exported controller data to {pkl_path}")

    return predictions

# --- TRAINING FUNCTION ---
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for pixel_values, _, _, labels in tqdm(loader, desc="Train"):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * labels.size(0)
        pred = outputs.logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pixel_values, _, _, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            pred = outputs.logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / total, correct / total
import re

def get_patient_id_from_video_id(video_id: str):
    """
    Extract patient id from strings like:
      patient_12_task_7_cam3_seg_4
    Returns int if possible, else string id.
    """
    m = re.search(r"patient_([^_]+)", video_id)
    if not m:
        return None
    pid = m.group(1)
    return int(pid) if pid.isdigit() else pid

# --- MAIN FLOW ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- CORRECTED MODEL ID ---
    # Using the standard V1 Base model from MCG-NJU (Original Authors)
    # This guarantees compatibility with 'VideoMAEForVideoClassification'
    model_name = "MCG-NJU/VideoMAE-Base"
    logger.info(f"Loading model: {model_name}")
    
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    
    config = VideoMAEConfig.from_pretrained(model_name)
    config.num_labels = num_class
    config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    config.label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
    
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name, 
        config=config, 
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Freeze encoder parameters if you want to only train the head
    # for param in model.videomae.parameters():
    #     param.requires_grad = False

    # --- DATA PREPARATION ---
    pickle_dir = "D:/nature_everything/nature_dataset"
    bbox_dir = "D:/nature_everything/frcnn_boxes/" + camera_box
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
    
    camera_df = pd.read_csv(ipsi_contra_csv)
    patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
    
    all_segments = []
    
    logger.info("Parsing Dataset...")
    for pkl_file in tqdm(pickle_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except: continue

        for camera_id in data:
            for segments_group in data[camera_id]:
                for segment in segments_group:
                    pid = segment['patient_id']
                    seg_cam = segment['CameraId']
                    
                    if camera_id == 'cam3' and camera_box == 'bboxes_top':
                        ipsi = camera_id
                    elif camera_id != 'cam3' and camera_box == 'bboxes_ipsi':
                        ipsi = patient_to_ipsilateral.get(pid)
                    else: continue
                    
                    if ipsi != seg_cam: continue
                    
                    ratings = segment.get('segment_ratings', {})
                    t1, t2 = ratings.get('t1'), ratings.get('t2')
                    
                    final_rating = None
                    if t1 == t2 and t1 in [1, 2, 3, '1', '2', '3']:
                        final_rating = int(t1)
                    elif t1 is not None and t1 in [1, 2, 3]: final_rating = int(t1)
                    elif t2 is not None and t2 in [1, 2, 3]: final_rating = int(t2)
                    
                    if final_rating is None: continue
                    
                    video_id = f"patient_{pid}_task_{segment['activity_id']}_{seg_cam}_seg_{segment['segment_id']}"
                    
                    all_segments.append({
                        'frames': segment['frames'],
                        'video_id': video_id,
                        'label': _map[final_rating], 
                        'hand_id': ipsi,
                        't1_label': _map.get(int(t1)) if t1 else None,
                        't2_label': _map.get(int(t2)) if t2 else None
                    })

    logger.info(f"Collected {len(all_segments)} segments.")
    groups = []
    k_folds = 5
    bad = 0
    for seg in all_segments:
        pid = get_patient_id_from_video_id(seg["video_id"])
        if pid is None:
            bad += 1
            pid = f"unknown_{bad}"  # keep split running, but you should inspect these
        groups.append(pid)

    groups = np.asarray(groups)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=k_folds)   
    if USE_PRETRAINED_FINETUNED:
        test_dataset = torch.utils.data.ConcatDataset([
            SinglePickleFrameDataset(s['frames'], s['video_id'], s['label'], s['hand_id'], cfg, bbox_dir, is_train=False, processor=processor)
            for s in all_segments
        ])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TEST_BATCH_SIZE, collate_fn=videomae_collate_fn)
        perform_inference(test_loader, model, cfg, all_segments)
    else:
        for fold, (train_idx, val_idx) in enumerate(gkf.split(all_segments, groups=groups)):
            logger.info(f"Starting fold {fold+1}/{k_folds}")

            train_segs = [all_segments[i] for i in train_idx]
            val_segs   = [all_segments[i] for i in val_idx]
            # sanity check: no patient overlap
            train_p = {get_patient_id_from_video_id(s["video_id"]) for s in train_segs}
            val_p   = {get_patient_id_from_video_id(s["video_id"]) for s in val_segs}
            overlap = train_p.intersection(val_p)
            logger.info(f"Fold {fold+1}: patient overlap = {len(overlap)}")            
            if CLASS_IMBALANCE:
                labels = [s['label'] for s in train_segs]
                counts = np.bincount(labels, minlength=3)
                weights = len(labels) / (3 * (counts + 1e-6))
                class_weights = torch.tensor(weights, dtype=torch.float).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
            scaler = GradScaler()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

            train_ds = torch.utils.data.ConcatDataset([
                SinglePickleFrameDataset(s['frames'], s['video_id'], s['label'], s['hand_id'], cfg, bbox_dir, is_train=True, processor=processor)
                for s in train_segs
            ])
            val_ds = torch.utils.data.ConcatDataset([
                SinglePickleFrameDataset(s['frames'], s['video_id'], s['label'], s['hand_id'], cfg, bbox_dir, is_train=False, processor=processor)
                for s in val_segs
            ])

            train_sampler = None
            if BALANCED_SAMPLING:
                lbls = [s['label'] for s in train_segs]
                cls_counts = np.bincount(lbls, minlength=3)
                sample_weights = [1.0/cls_counts[l] for l in lbls]
                train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
            
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=cfg.TRAIN_BATCH_SIZE, 
                sampler=train_sampler, shuffle=(train_sampler is None),
                collate_fn=videomae_collate_fn
            )
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.TEST_BATCH_SIZE, collate_fn=videomae_collate_fn)
            
            best_acc = 0
            for epoch in range(15):
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                scheduler.step()
                
                logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"best_model_fold_{fold+1}.pt"))
            
            # Reset model for next fold
            model = VideoMAEForVideoClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True).to(device)

if __name__ == "__main__":
    main()