import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = r'C:\Users\hyuna\OneDrive\바탕 화면\code\Pytorch\MyWork\multimodal\data\picture_of_me_2.jpg'
#image = Image.open(requests.get(image_path, stream=True).raw)
image = Image.open(image_path)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a person. a fish. a shark. a paper bag."

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

fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)

# Bounding Box 추가
for i, (score, box, label) in enumerate(zip(results[0]['scores'].cpu(), results[0]['boxes'].cpu(), results[0]['text_labels'])):
    x_min, y_min, x_max, y_max = box
    width, height = x_max - x_min, y_max - y_min

    # 박스 그리기
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Label 추가
    text = f"{label}"
    if score:
        text += f" ({score:.2f})"
    
    ax.text(x_min, y_min - 5, text, fontsize=12, color='white',
            bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

# 결과 출력
plt.axis('off')
plt.savefig('./result/gdino_result3.png')

