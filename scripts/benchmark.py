"""YOLOv5s / YOLO26s 추론 속도 비교 (원본 크기 이미지 기준, end-to-end).

usage:
    python scripts/benchmark.py --v5 runs/v5/weights/best.pt --new runs/yolo26s/weights/best.pt \
        --images bench_original --yolov5-dir yolov5 --out results/speed.csv
"""
import argparse
import collections
import glob
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_v5(weights, yolov5_dir, imgs, device, imgsz):
    model = torch.hub.load(yolov5_dir, 'custom', path=weights, source='local').to(device)
    for p in imgs[:5]:
        model(Image.open(p), size=imgsz)  # warm-up
    times = []
    for p in imgs:
        im = Image.open(p)
        sync()
        s = time.perf_counter()
        model(im, size=imgsz)
        sync()
        times.append((time.perf_counter() - s) * 1000)
    return {'mean_ms': np.mean(times), 'p95_ms': np.percentile(times, 95)}


def bench_ultralytics(weights, imgs, device, imgsz):
    model = YOLO(weights)
    for p in imgs[:5]:
        model.predict(p, imgsz=imgsz, device=device, verbose=False)
    times, stages = [], collections.defaultdict(list)
    for p in imgs:
        sync()
        s = time.perf_counter()
        r = model.predict(p, imgsz=imgsz, device=device, verbose=False)[0]
        sync()
        times.append((time.perf_counter() - s) * 1000)
        for k, v in r.speed.items():
            stages[k].append(v)
    out = {'mean_ms': np.mean(times), 'p95_ms': np.percentile(times, 95)}
    out.update({f'{k}_ms': np.mean(v) for k, v in stages.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v5', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--new-name', default='yolo26s')
    ap.add_argument('--images', required=True)
    ap.add_argument('--yolov5-dir', default='yolov5')
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--out', default='results/speed.csv')
    args = ap.parse_args()

    imgs = sorted(glob.glob(f'{args.images}/*'))
    devices = (['cuda'] if torch.cuda.is_available() else []) + ['cpu']
    rows = []
    for dev in devices:
        rows.append({'model': 'yolov5s', 'device': dev,
                     **bench_v5(args.v5, args.yolov5_dir, imgs, dev, args.imgsz)})
        rows.append({'model': args.new_name, 'device': dev,
                     **bench_ultralytics(args.new, imgs, 0 if dev == 'cuda' else 'cpu', args.imgsz)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == '__main__':
    main()
