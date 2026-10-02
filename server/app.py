"""GETCHA 추론 서버.

변경 사항 (2026)
- 모델 로드: torch.hub force_reload=True 제거, 로컬 yolov5 저장소에서 로드 (서버 시작 124s -> 0.3s)
- MODEL_TYPE 환경변수로 YOLOv5 / YOLO26 선택
- 요청마다 결과 이미지를 고유 파일명으로 저장해 동시 요청 시 결과가 섞이지 않도록 수정
- Windows에서만 PosixPath 패치 적용 (Colab에서 학습한 가중치 로드용)
"""
import io
import os
import pathlib
import uuid

from flask import Flask, jsonify, render_template_string, request, send_file
from PIL import Image, ImageDraw, ImageFont

if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

MODEL_TYPE = os.getenv('MODEL_TYPE', 'yolov5')        # yolov5 | yolo26
MODEL_PATH = os.getenv('MODEL_PATH', 'weights/best.pt')
YOLOV5_DIR = os.getenv('YOLOV5_DIR', 'yolov5')         # git clone https://github.com/ultralytics/yolov5
IMG_SIZE = int(os.getenv('IMG_SIZE', 416))
RESULT_DIR = os.getenv('RESULT_DIR', 'results')
RED = (255, 0, 0)

os.makedirs(RESULT_DIR, exist_ok=True)
LATEST_ID = None
app = Flask(__name__)


def load_model():
    if MODEL_TYPE == 'yolo26':
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    import torch
    return torch.hub.load(YOLOV5_DIR, 'custom', path=MODEL_PATH, source='local')


model = load_model()

try:
    FONT = ImageFont.truetype('arial.ttf', 30)
except IOError:
    FONT = ImageFont.load_default()


def detect(img):
    """[(xmin, ymin, xmax, ymax, conf, class_name), ...]"""
    if MODEL_TYPE == 'yolo26':
        r = model.predict(img, imgsz=IMG_SIZE, verbose=False)[0]
        return [(*b.xyxy[0].tolist(), float(b.conf), r.names[int(b.cls)]) for b in r.boxes]
    r = model(img, size=IMG_SIZE)
    return [(*box[:4].tolist(), float(box[4]), r.names[int(box[5])]) for box in r.xyxy[0]]


def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    draw = ImageDraw.Draw(img)
    predictions = []
    for i, (x1, y1, x2, y2, conf, name) in enumerate(detect(img), start=1):
        draw.rectangle([x1, y1, x2, y2], outline=RED, width=5)
        draw.text((x1, y1), f'{i}. {name} {conf:.2f}', fill=RED, font=FONT)
        predictions.append({'object_id': i, 'class': name, 'confidence': conf,
                            'box': [x1, y1, x2, y2], 'color': RED})
    image_id = uuid.uuid4().hex
    img.save(os.path.join(RESULT_DIR, f'{image_id}.jpg'))
    global LATEST_ID
    LATEST_ID = image_id
    return image_id, predictions


@app.route('/')
def home():
    return render_template_string('''
    <html><head><title>GETCHA</title></head>
    <body>
      <h1>이미지 업로드 테스트</h1>
      <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*"><br><br>
        <input type="submit" value="예측">
      </form>
    </body></html>''')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400
    image_id, predictions = predict_image(request.files['image'].read())
    return jsonify({'image_url': f'/image/{image_id}', 'predictions': predictions})


@app.route('/image', methods=['GET'])
def get_latest_image():
    # 기존 클라이언트 호환용
    if LATEST_ID is None:
        return jsonify({'error': 'no prediction yet'}), 404
    return get_image(LATEST_ID)


@app.route('/image/<image_id>', methods=['GET'])
def get_image(image_id):
    path = os.path.join(RESULT_DIR, f'{os.path.basename(image_id)}.jpg')
    if not os.path.exists(path):
        return jsonify({'error': 'not found'}), 404
    return send_file(path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
