"""기존 Flask 서버(app.py) 처리 과정 단계별 시간 측정 (CPU).

- 모델 로드: torch.hub force_reload=True(기존) vs 로컬 저장소 로드
- 요청 1건: decode / inference / draw / save

usage:
    python scripts/profile_server.py --weights runs/v5/weights/best.pt --image sample.jpg --yolov5-dir yolov5
"""
import argparse
import collections
import io
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


def timed(fn, *a, **k):
    s = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - s) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--yolov5-dir', default='yolov5')
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--repeat', type=int, default=30)
    ap.add_argument('--out', default='results/server_profile.csv')
    args = ap.parse_args()

    _, t_reload = timed(torch.hub.load, 'ultralytics/yolov5', 'custom', path=args.weights, force_reload=True)
    _, t_local = timed(torch.hub.load, args.yolov5_dir, 'custom', path=args.weights, source='local')
    print(f'model load  force_reload: {t_reload:.0f} ms | local: {t_local:.0f} ms')

    model = torch.hub.load(args.yolov5_dir, 'custom', path=args.weights, source='local').to('cpu')
    raw = open(args.image, 'rb').read()
    stage = collections.defaultdict(list)
    for _ in range(args.repeat):
        img, t1 = timed(lambda b: Image.open(io.BytesIO(b)).convert('RGB'), raw)
        res, t2 = timed(model, img, size=args.imgsz)

        def draw():
            d = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype('arial.ttf', 30)
            except IOError:
                font = ImageFont.load_default()
            for b in res.xyxy[0]:
                d.rectangle(b[:4].tolist(), outline=(255, 0, 0), width=5)

        _, t3 = timed(draw)
        _, t4 = timed(img.save, 'predicted_image.jpg')
        for k, v in zip(['decode', 'inference', 'draw', 'save'], [t1, t2, t3, t4]):
            stage[k].append(v)

    df = pd.DataFrame({k: [np.mean(v)] for k, v in stage.items()}).T.rename(columns={0: 'mean_ms'})
    df['share_%'] = df['mean_ms'] / df['mean_ms'].sum() * 100
    df.loc['model_load_force_reload'] = [t_reload, None]
    df.loc['model_load_local'] = [t_local, None]
    df.index.name = 'stage'
    df.to_csv(args.out)
    print(df)


if __name__ == '__main__':
    main()
