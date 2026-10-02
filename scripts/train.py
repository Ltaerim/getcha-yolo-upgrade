"""YOLOv5s / YOLO26s 동일 조건 학습 + 검증.

usage:
    python scripts/train.py --data configs/data.yaml --project runs --epochs 30
"""
import argparse
import os
import re
import subprocess
import time

import pandas as pd
from ultralytics import YOLO

METRIC_KEYS = ['P', 'R', 'mAP50', 'mAP50-95']


def parse_v5_val(log, name):
    m = re.search(rf'^\s*{name}\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', log, re.M)
    return dict(zip(METRIC_KEYS, map(float, m.groups()))) if m else None


def train_v5(args):
    out_dir = os.path.join(args.project, 'v5')
    t0 = time.time()
    subprocess.run(['python', 'train.py', '--img', str(args.img), '--batch', str(args.batch),
                    '--epochs', str(args.epochs), '--data', os.path.abspath(args.data),
                    '--weights', 'yolov5s.pt', '--seed', str(args.seed), '--device', '0',
                    '--cache', 'ram', '--project', os.path.abspath(args.project), '--name', 'v5', '--exist-ok'],
                   cwd=args.yolov5_dir, check=True)
    train_min = (time.time() - t0) / 60
    best = os.path.join(out_dir, 'weights', 'best.pt')
    res = subprocess.run(['python', 'val.py', '--weights', os.path.abspath(best), '--data', os.path.abspath(args.data),
                          '--img', str(args.img), '--task', 'val', '--device', '0'],
                         cwd=args.yolov5_dir, capture_output=True, text=True)
    log = res.stdout + res.stderr
    allm, abn = parse_v5_val(log, 'all'), parse_v5_val(log, '이상') or {}
    return {'model': 'yolov5s', **allm, 'abnormal_AP50': abn.get('mAP50'),
            'abnormal_AP50-95': abn.get('mAP50-95'), 'train_min': train_min,
            'size_MB': os.path.getsize(best) / 1e6}


def train_ultralytics(args, weights):
    name = weights.replace('.pt', '')
    t0 = time.time()
    YOLO(weights).train(data=args.data, imgsz=args.img, batch=args.batch, epochs=args.epochs,
                        seed=args.seed, device=0, cache='ram', project=args.project, name=name, exist_ok=True)
    train_min = (time.time() - t0) / 60
    best = os.path.join(args.project, name, 'weights', 'best.pt')
    m = YOLO(best).val(data=args.data, imgsz=args.img, device=0)
    return {'model': name, 'P': m.box.mp, 'R': m.box.mr, 'mAP50': m.box.map50, 'mAP50-95': m.box.map,
            'abnormal_AP50': float(m.box.ap50[1]), 'abnormal_AP50-95': float(m.box.ap[1]),
            'train_min': train_min, 'size_MB': os.path.getsize(best) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='configs/data.yaml')
    ap.add_argument('--project', default='runs')
    ap.add_argument('--yolov5-dir', default='yolov5')
    ap.add_argument('--new', default='yolo26s.pt')
    ap.add_argument('--img', type=int, default=416)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/accuracy.csv')
    args = ap.parse_args()

    results = [train_v5(args), train_ultralytics(args, args.new)]
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == '__main__':
    main()
