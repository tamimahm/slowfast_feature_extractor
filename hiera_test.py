#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# Configuration flag:
# 0 = do fine-tuning and then testing
# 1 = load saved fine-tuned model and run testing only
USE_PRETRAINED_FINETUNED = 0
CLASS_IMBALANCE = 1  # 1 to enable class weight balancing
BALANCED_SAMPLING = 1  # 1 to enable balanced batch sampling
SAVE_FEATURES = 1  # 1 to save features (Heatmaps are disabled for Hiera as it requires specialized attention rollout)
num_class = 3  # Classes 0, 1, 2

import numpy as np
import torch
import torch.nn as nn
import os
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
from tqdm import tqdm

# Hiera Imports
try:
    import hiera
except ImportError:
    raise ImportError("Please install hiera: pip install hiera-transformer")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class Config:
    def __init__(self):
        self.OUTPUT_DIR = "D:\\nature_everything\\output_hiera"
        self.DATA_LOADER_PIN_MEMORY = True
        self.TRAIN_BATCH_SIZE = 4
        self.TEST_BATCH_SIZE = 4
        # Hiera Base 16x224 config
        self.NUM_FRAMES = 16 
        self.SAMPLING_RATE = 4 
        self.INPUT_SIZE = 224
        self.RNG_SEED = 42

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Camera/File Configuration
ipsi_contra_csv = "D:\\nature_everything\\camera_assignments.csv"
camera_box = "bboxes_ipsi"
_map = {1: 0, 2: 1, 3: 2}

def load_bboxes(bbox_dir, video_id):
    bbox_file = os.path.join(bbox_dir, f"{video_id}_bboxes.pkl")
    if not os.path.exists(bbox_file): return None
    try:
        with open(bbox_file, 'rb') as f: return pickle.load(f)
    except: return None

# --- DATASET ---
class SinglePickleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames, video_id, label, camera_id, cfg, bbox_dir, is_train=True):
        self.frames = frames
        self.video_id = video_id
        self.label = label
        self.hand_id = camera_id
        self.cfg = cfg
        self.bbox_dir = bbox_dir
        self.is_train = is_train
        self.bboxes = load_bboxes(bbox_dir, video_id)

        # Standard ImageNet stats for Hiera
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(cfg.INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(cfg.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Temporal Sampling
        vlen = len(self.frames)
        total_span = self.cfg.NUM_FRAMES * self.cfg.SAMPLING_RATE
        
        if vlen >= total_span:
            if self.is_train:
                start_idx = np.random.randint(0, vlen - total_span + 1)
            else:
                start_idx = (vlen - total_span) // 2
            indices = np.arange(start_idx, start_idx + total_span, self.cfg.SAMPLING_RATE)
        else:
            # Pad if too short
            indices = np.arange(0, vlen, self.cfg.SAMPLING_RATE)
            while len(indices) < self.cfg.NUM_FRAMES:
                indices = np.concatenate([indices, [indices[-1]]])
            indices = indices[:self.cfg.NUM_FRAMES]

        # Process Frames
        imgs = []
        raw_frames = [] # Keep for visualization if needed
        
        for i in indices:
            i = int(i)
            try:
                frame = self.frames[i]
                if isinstance(frame, torch.Tensor): frame = frame.permute(1,2,0).numpy()
                if frame.max() <= 1.0: frame = (frame*255).astype(np.uint8)
                else: frame = frame.astype(np.uint8)
                
                # BBox Cropping
                bbox = (self.bboxes[i] if self.bboxes and i < len(self.bboxes) else None)
                if bbox is not None:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    h, w = frame.shape[:2]
                    # Padding logic
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
                    
                    if x_max > x_min and y_max_ext > y_min:
                        frame = frame[y_min:y_max_ext, x_min:x_max]

                # Convert to PIL and Transform
                pil_img = Image.fromarray(frame)
                imgs.append(self.transform(pil_img))
                raw_frames.append(frame)

            except Exception as e:
                # Fallback blank frame
                imgs.append(torch.zeros(3, self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
                raw_frames.append(np.zeros((224,224,3), dtype=np.uint8))

        # Stack: (T, C, H, W) -> (C, T, H, W)
        # Hiera expects (B, 3, T, H, W)
        video_tensor = torch.stack(imgs).permute(1, 0, 2, 3) 
        
        return video_tensor, self.video_id, raw_frames, torch.tensor(self.label, dtype=torch.long)

def hiera_collate_fn(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    video_ids = [item[1] for item in batch]
    raw = [item[2] for item in batch]
    labels = torch.stack([item[3] for item in batch], dim=0)
    return inputs, video_ids, raw, labels

# --- HELPERS ---
def _safe_entropy_from_probs(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def parse_video_id_umera(video_id):
    parts = video_id.split("_")
    out = {"video_id": video_id, "patient_id": None, "task_key": None}
    try:
        pid = parts[1]
        act = parts[3]
        cam = parts[4]
        out.update({
            "patient_id": int(pid) if pid.isdigit() else pid,
            "activity_id": int(act) if act.isdigit() else act,
            "camera_id": cam,
            "segment_id": int(parts[-1]) if parts[-2] == "seg" else None,
            "task_key": f"patient_{pid}_task_{act}"
        })
    except: pass
    return out

# --- MODEL WRAPPER ---
class HieraWrapper(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Load 'hiera_base_16x224' from torch hub
        self.model = torch.hub.load('facebookresearch/hiera', model='hiera_base_16x224', pretrained=pretrained, checkpoint="mae_k400_ft_k400")
        
        # Replace Head
        in_features = self.model.head.projection.in_features
        self.model.head.projection = nn.Linear(in_features, num_classes)
        
        # Hook for features
        self.features = None
        self.model.head.projection.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # Input to head is the pooled feature (B, C)
        self.features = input[0].detach()

    def forward(self, x):
        return self.model(x)

# --- INFERENCE ---
@torch.no_grad()
def perform_inference(test_loader, model, cfg, inference_segments=None, export_for_controller=True):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    controller_records = []
    
    # Setup feature export dir
    feat_dir = os.path.join(cfg.OUTPUT_DIR, "features_hiera")
    os.makedirs(feat_dir, exist_ok=True)
    
    inference_map = {s['video_id']: s for s in inference_segments} if inference_segments else {}

    for inputs, video_ids, _, labels in tqdm(test_loader, desc="Inference"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward
        preds = model(inputs) # (B, num_classes)
        probs = F.softmax(preds, dim=1)
        
        # Get features captured by hook
        batch_feats = model.features # (B, 768)
        
        batch_size = labels.size(0)
        for i in range(batch_size):
            vid = video_ids[i]
            pred_cls = torch.argmax(probs[i]).item()
            conf = probs[i, pred_cls].item()
            
            # Metadata
            meta = parse_video_id_umera(vid)
            t1 = inference_map.get(vid, {}).get('t1_label')
            t2 = inference_map.get(vid, {}).get('t2_label')
            
            # Save Feature
            feat_vec = batch_feats[i].cpu().numpy()
            if SAVE_FEATURES:
                np.save(os.path.join(feat_dir, f"{vid}_feat.npy"), feat_vec)
            
            # Record for Controller
            controller_records.append({
                "video_id": vid,
                "patient_id": meta.get("patient_id"),
                "task_key": meta.get("task_key"),
                "pred": int(pred_cls),
                "confidence": float(conf),
                "logits": preds[i].cpu().numpy(),
                "probs": probs[i].cpu().numpy(),
                "entropy": _safe_entropy_from_probs(probs[i].cpu().numpy()),
                "feat_global": feat_vec,
                "t1_label": t1,
                "t2_label": t2
            })
            
            # Accuracy Check
            is_correct = False
            if t1 is not None and pred_cls == t1: is_correct = True
            if t2 is not None and pred_cls == t2: is_correct = True
            
            predictions.append({
                'video_id': vid,
                'predicted': pred_cls,
                'correct': is_correct
            })
            
    # Save CSV
    pd.DataFrame(predictions).to_csv(os.path.join(cfg.OUTPUT_DIR, "hiera_predictions.csv"), index=False)
    
    # Save Controller PKL
    if export_for_controller:
        pkl_path = os.path.join(cfg.OUTPUT_DIR, "hiera_controller_export.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"segment_records": controller_records}, f)
        logger.info(f"Exported to {pkl_path}")
        
    return predictions

# --- TRAINING ---
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for inputs, _, _, labels in tqdm(loader, desc="Train"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, _, _, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
    return running_loss/total, correct/total

# --- MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize Model
    model = HieraWrapper(num_classes=num_class).to(device)
    logger.info("Initialized Hiera-Base (16x224)")

    # Data Loading (Identical logic to user script)
    pickle_dir = "D:/nature_everything/nature_dataset"
    bbox_dir = "D:/nature_everything/frcnn_boxes/" + camera_box
    pickle_files = glob.glob(os.path.join(pickle_dir, '*.pkl'))
    
    camera_df = pd.read_csv(ipsi_contra_csv)
    patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
    
    all_segments = []
    logger.info("Loading Data...")
    
    for pkl_file in tqdm(pickle_files):
        try:
            with open(pkl_file, 'rb') as f: data = pickle.load(f)
        except: continue

        for camera_id in data:
            for seg in data[camera_id][0]: # Assuming structure
                # Filtering
                pid = seg['patient_id']
                if camera_id=='cam3' and camera_box=='bboxes_top': ipsi = camera_id
                elif camera_id!='cam3' and camera_box=='bboxes_ipsi': ipsi = patient_to_ipsilateral.get(pid)
                else: continue
                
                if ipsi != seg['CameraId']: continue
                
                # Rating
                ratings = seg.get('segment_ratings', {})
                t1, t2 = ratings.get('t1'), ratings.get('t2')
                
                valid_rating = None
                if t1 == t2 and t1 in [1,2,3]: valid_rating = int(t1)
                elif t1 in [1,2,3]: valid_rating = int(t1)
                elif t2 in [1,2,3]: valid_rating = int(t2)
                
                if valid_rating is None: continue
                
                all_segments.append({
                    'frames': seg['frames'],
                    'video_id': f"patient_{pid}_task_{seg['activity_id']}_{seg['CameraId']}_seg_{seg['segment_id']}",
                    'label': _map[valid_rating],
                    'hand_id': ipsi,
                    't1_label': _map.get(int(t1)) if t1 else None,
                    't2_label': _map.get(int(t2)) if t2 else None
                })
    
    logger.info(f"Found {len(all_segments)} segments.")

    if USE_PRETRAINED_FINETUNED:
        # Load weights and inference
        ckpt = os.path.join(cfg.OUTPUT_DIR, "best_fold_model.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt)['state_dict'])
            
        ds = SinglePickleFrameDataset([], "", 0, "", cfg, bbox_dir, is_train=False) # Dummy for class usage
        # Construct actual dataset
        datasets = [SinglePickleFrameDataset(s['frames'], s['video_id'], s['label'], s['hand_id'], cfg, bbox_dir, False) for s in all_segments]
        loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), 
                                           batch_size=cfg.TEST_BATCH_SIZE, 
                                           collate_fn=hiera_collate_fn)
        perform_inference(loader, model, cfg, all_segments)
        
    else:
        # K-Fold Training
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_segments)):
            logger.info(f"--- FOLD {fold+1} ---")
            train_segs = [all_segments[i] for i in train_idx]
            val_segs = [all_segments[i] for i in val_idx]
            
            # Class Weights
            if CLASS_IMBALANCE:
                lbls = [s['label'] for s in train_segs]
                cnts = np.bincount(lbls, minlength=3)
                w = len(lbls) / (3 * (cnts + 1e-6))
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))
            else:
                criterion = nn.CrossEntropyLoss()
                
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scaler = GradScaler()
            
            # Loaders
            train_ds = [SinglePickleFrameDataset(s['frames'], s['video_id'], s['label'], s['hand_id'], cfg, bbox_dir, True) for s in train_segs]
            val_ds = [SinglePickleFrameDataset(s['frames'], s['video_id'], s['label'], s['hand_id'], cfg, bbox_dir, False) for s in val_segs]
            
            # Sampler
            sampler = None
            if BALANCED_SAMPLING:
                lbls = [s['label'] for s in train_segs]
                c_counts = np.bincount(lbls, minlength=3)
                weights = [1.0/c_counts[l] for l in lbls]
                sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
            
            train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_ds), 
                                                     batch_size=cfg.TRAIN_BATCH_SIZE,
                                                     sampler=sampler,
                                                     shuffle=(sampler is None),
                                                     collate_fn=hiera_collate_fn)
                                                     
            val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_ds), 
                                                   batch_size=cfg.TEST_BATCH_SIZE,
                                                   collate_fn=hiera_collate_fn)
            
            best_acc = 0
            for epoch in range(15):
                t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
                v_loss, v_acc = validate(model, val_loader, criterion, device)
                logger.info(f"Ep {epoch+1}: Train {t_acc:.3f} | Val {v_acc:.3f}")
                
                if v_acc > best_acc:
                    best_acc = v_acc
                    torch.save({'state_dict': model.state_dict(), 'best_val_acc': best_acc}, 
                             os.path.join(cfg.OUTPUT_DIR, "best_fold_model.pt"))
            
            # Reset model for next fold
            model = HieraWrapper(num_classes=num_class).to(device)

if __name__ == "__main__":
    main()