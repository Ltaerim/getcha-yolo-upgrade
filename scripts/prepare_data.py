"""AIHub 전차선 애자 데이터셋 전처리.

1. 라벨 재정의: 줄 끝 상태 텍스트('이상 ...')를 기준으로 클래스를 0(정상) / 1(이상)으로 다시 매긴다.
   - train: 이상 객체에만 상태 텍스트가 있고, 텍스트가 없는 줄은 정상
   - val:   모든 줄에 '정상' 또는 '이상 ...' 텍스트가 있음
2. 이미지 사전 리사이즈: 긴 변 기준 416px로 한 번만 줄여 학습 시 디코딩/리사이즈 비용을 없앤다.
3. data.yaml 생성

usage:
    python scripts/prepare_data.py --root /content/getcha_data/dataset --img 416
"""
import argparse
import collections
import glob
import os
import shutil
import time
from multiprocessing import Pool

import cv2

CLASS_NAMES = ['정상', '이상']


def relabel_file(path):
    rows, counts = [], collections.Counter()
    with open(path, encoding='utf-8') as f:
        for line in f.read().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            cls = '1' if (len(t) > 5 and t[5].startswith('이상')) else '0'
            rows.append(' '.join([cls] + t[1:5]))
            counts[cls] += 1
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + ('\n' if rows else ''))
    return counts


def relabel(root):
    marker = os.path.join(root, '.relabeled')
    if os.path.exists(marker):
        print('labels already relabeled, skip')
        return
    for split in ['train', 'val']:
        total = collections.Counter()
        for path in glob.glob(f'{root}/labels/{split}/*.txt'):
            total += relabel_file(path)
        print(f'[{split}] normal(0): {total["0"]}, abnormal(1): {total["1"]}')
    open(marker, 'w').close()


def shrink(args):
    path, size = args
    img = cv2.imread(path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    r = size / max(h, w)
    if r < 1:
        img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return 1


def resize_images(root, size, keep_original=100, bench_dir=None):
    marker = os.path.join(root, '.resized')
    if os.path.exists(marker):
        print('images already resized, skip')
        return
    val_imgs = sorted(glob.glob(f'{root}/images/val/*'))
    if bench_dir:  # 속도 측정용 원본 이미지 보관
        os.makedirs(bench_dir, exist_ok=True)
        for p in val_imgs[:keep_original]:
            shutil.copy(p, bench_dir)
    paths = glob.glob(f'{root}/images/train/*') + val_imgs
    t0 = time.time()
    with Pool(os.cpu_count()) as pool:
        n = sum(pool.imap_unordered(shrink, [(p, size) for p in paths], chunksize=64))
    print(f'resized {n} images in {(time.time() - t0) / 60:.1f} min (cpu={os.cpu_count()})')
    open(marker, 'w').close()


def write_yaml(root, out):
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'train: {root}/images/train\n')
        f.write(f'val: {root}/images/val\n')
        f.write(f'nc: {len(CLASS_NAMES)}\n')
        f.write(f'names: {CLASS_NAMES}\n')
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='images/, labels/ 를 포함한 데이터 경로')
    ap.add_argument('--img', type=int, default=416)
    ap.add_argument('--yaml', default='configs/data.yaml')
    ap.add_argument('--bench-dir', default='bench_original')
    args = ap.parse_args()

    relabel(args.root)
    resize_images(args.root, args.img, bench_dir=args.bench_dir)
    for cache in glob.glob(f'{args.root}/labels/*.cache'):
        os.remove(cache)
    write_yaml(args.root, args.yaml)


if __name__ == '__main__':
    main()
