#!/usr/bin/env python
import cv2, torch, numpy as np, logging, os, sys
from tqdm import tqdm
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.optimize import linear_sum_assignment
from model import BEVNetResNet101

# ───────── Config ────────────────────────────────────────────────────────
MODEL_ID = "facebook/mask2former-swin-large-coco-panoptic"
CATEGORY_IDS = {2, 5, 7}  # bicycle, airplane, truck (可改成 car, truck 等)
IOU_MATCH = 0.30
ALPHA = 0.60
LOST_TOL = 5

GRID_SIZE_X, GRID_SIZE_Y = 5, 7
GRID_CELL = 40
PROB_THRESHOLD = 0.60
BEV_MODEL_PATH = "bev_model.pth"
VIDEO_INPUT = "input_video_ets_test.mp4"
VIDEO_OUTPUT = "output_bev_grid_overlay.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.getLogger("transformers.image_processing_utils").setLevel("ERROR")

# Load icons
car_icon = cv2.imread("car_green.jpg", cv2.IMREAD_UNCHANGED)
car_icon = cv2.resize(car_icon, (GRID_CELL, GRID_CELL))
ego_icon = cv2.imread("car_black.jpg", cv2.IMREAD_UNCHANGED)
ego_icon = cv2.resize(ego_icon, (GRID_CELL, GRID_CELL))

# ───────── Utility ───────────────────────────────────────────────────────
def iou_xyxy(a, b):
    xx1, yy1 = np.maximum(a[:2], b[:2])
    xx2, yy2 = np.minimum(a[2:], b[2:])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union else 0.

def overlay(frame, pan_seg, infos, seg2tid):
    vis = frame.copy()
    for info in infos:
        if info["label_id"] not in CATEGORY_IDS:
            continue
        tid = seg2tid.get(info["id"], -1)
        if tid == -1:
            continue
        vis[pan_seg == info["id"]] = (1 - ALPHA) * vis[pan_seg == info["id"]] + ALPHA * np.array([0, 255, 0])
    return vis.astype(np.uint8)

# ───────── Main ──────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_ID)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID).to(DEVICE).eval()
    bev_model = BEVNetResNet101().to(DEVICE)
    bev_model.load_state_dict(torch.load(BEV_MODEL_PATH, map_location=DEVICE))
    bev_model.eval()

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        sys.exit("Cannot open input video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks, next_tid = {}, 1
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0, desc="Tracking")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = frame[:, :, ::-1]

        # ── Segmentation ──
        ins = processor(images=rgb, return_tensors="pt").to(DEVICE)
        outs = seg_model(**ins)
        post = processor.post_process_panoptic_segmentation(outs, target_sizes=[(h, w)], label_ids_to_fuse=[])[0]
        pan_seg = post["segmentation"].cpu().numpy()
        infos_all = [inf for inf in post["segments_info"] if inf["label_id"] in CATEGORY_IDS]

        boxes, seg_ids, infos = [], [], []
        for inf in infos_all:
            mask = pan_seg == inf["id"]
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            seg_ids.append(inf["id"])
            infos.append(inf)
        boxes = np.asarray(boxes, float)

        # ── Tracking ──
        track_ids = list(tracks)
        prev_boxes = np.array([tracks[t]["box"] for t in track_ids])
        assigned_prev, assigned_cur = set(), set()
        if len(prev_boxes) and len(boxes):
            ious = np.array([[iou_xyxy(pb, cb) for cb in boxes] for pb in prev_boxes])
            r, c = linear_sum_assignment(1 - ious)
            for i, j in zip(r, c):
                if ious[i, j] < IOU_MATCH:
                    continue
                tid = track_ids[i]
                tracks[tid]["box"] = boxes[j]
                tracks[tid]["label_id"] = infos[j]["label_id"]
                tracks[tid]["lost"] = 0
                assigned_prev.add(tid)
                assigned_cur.add(j)

        for tid in track_ids:
            if tid not in assigned_prev:
                tracks[tid]["lost"] += 1
        for i, box in enumerate(boxes):
            if i in assigned_cur:
                continue
            tracks[next_tid] = {"box": box, "lost": 0, "label_id": infos[i]["label_id"]}
            next_tid += 1
        tracks = {k: v for k, v in tracks.items() if v["lost"] < LOST_TOL}

        seg2tid = {}
        for sid, box in zip(seg_ids, boxes):
            best_tid, best_iou = -1, 0
            for tid, tr in tracks.items():
                i = iou_xyxy(box, tr["box"])
                if i > best_iou:
                    best_tid, best_iou = tid, i
            if best_tid != -1:
                seg2tid[sid] = best_tid

        vis = overlay(frame, pan_seg, infos, seg2tid)

        # ── BEV Input Prep ──
        vehicle_mask = np.zeros((h, w), dtype=np.uint8)
        for seg in post["segments_info"]:
            if seg["label_id"] in CATEGORY_IDS:
                vehicle_mask[pan_seg == seg["id"]] = 1

        resized_rgb = cv2.resize(rgb, (256, 256))
        resized_mask = cv2.resize(vehicle_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        rgb_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(resized_mask).unsqueeze(0).float()
        bev_input = torch.cat([rgb_tensor, mask_tensor], dim=0).unsqueeze(0).to(DEVICE)

        bev_output = bev_model(bev_input)
        bev_probs = torch.sigmoid(bev_output).squeeze(0).cpu().numpy()

        # ── BEV Grid Drawing ──
        canvas_h = GRID_CELL * GRID_SIZE_Y
        canvas_w = GRID_CELL * GRID_SIZE_X
        bev_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        for i in range(GRID_SIZE_Y):
            for j in range(GRID_SIZE_X):
                if i < bev_probs.shape[0] and j < bev_probs.shape[1]:
                    if bev_probs[i, j] > PROB_THRESHOLD:
                        y1, y2 = i * GRID_CELL, (i + 1) * GRID_CELL
                        x1, x2 = j * GRID_CELL, (j + 1) * GRID_CELL
                        bev_canvas[y1:y2, x1:x2] = car_icon[:, :, :3]

        # Ego car
        ego_i, ego_j = 6, 2
        y1, y2 = ego_i * GRID_CELL, (ego_i + 1) * GRID_CELL
        x1, x2 = ego_j * GRID_CELL, (ego_j + 1) * GRID_CELL
        bev_canvas[y1:y2, x1:x2] = ego_icon[:, :, :3]

        resized_bev = cv2.resize(bev_canvas, (150, 210))
        fy = vis.shape[0] - resized_bev.shape[0]
        fx = vis.shape[1] - resized_bev.shape[1]
        vis[fy:fy + resized_bev.shape[0], fx:fx + resized_bev.shape[1]] = resized_bev

        vw.write(vis)
        pbar.update(1)

    pbar.close()
    cap.release()
    vw.release()
    print(f"Done. Output saved to: {VIDEO_OUTPUT}")

# ───────── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
