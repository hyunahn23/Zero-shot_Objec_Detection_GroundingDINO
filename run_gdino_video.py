import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import numpy as np  

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# 비디오 경로 설정 (영상 입력)
video_path = r'C:\Users\hyuna\OneDrive\바탕 화면\code\Pytorch\MyWork\multimodal\data\fish.mp4'  # 실제 비디오 파일 경로
output_video_path = './result/gdino_result_video.mp4'  # 출력 비디오 경로

# 비디오 불러오기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 비디오 writer 설정 (MP4로 저장)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

text = "a fish."  # 텍스트 프롬프트 (기존과 동일)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # OpenCV 프레임을 PIL 이미지로 변환
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 모델 처리 (inputs, outputs, results)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )
    
    # 결과를 프레임에 그리기 (matplotlib 대신 OpenCV로 직접 그리기 – 비디오 저장 효율 위해)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB로 변환
    for i, (score, box, label) in enumerate(zip(results[0]['scores'].cpu(), results[0]['boxes'].cpu(), results[0]['text_labels'])):
        x_min, y_min, x_max, y_max = map(int, box)  # 좌표 정수화
        # Bounding Box 그리기 (빨간색, 두께 2)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        # Label 추가 (위에 텍스트, 흰색 배경)
        text_label = f"{label} ({score:.2f})"
        cv2.putText(frame, text_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 처리된 프레임을 출력 비디오에 쓰기
    out.write(frame)
    
    frame_count += 1
    print(f"Processed frame {frame_count}")  # 진행 상황 출력 (옵션)

# 자원 해제
cap.release()
out.release()
print(f"Result video saved to {output_video_path}")