import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import seaborn as sns
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dataset = load_dataset("clip-benchmark/wds_imagenetv2")
train_dataset = dataset['test']
sample_size = 10  # 사용할 샘플 개수
subset = train_dataset.shuffle().select(range(sample_size))


file_path = r'C:\Users\hyuna\OneDrive\바탕 화면\code\Pytorch\MyWork\multimodal\data\classnames.txt'
cls2label = open(file_path, 'r').readlines()
images = subset['webp']
label_texts = [cls2label[sub].rstrip() for sub in subset["cls"]]


inputs_image = processor(images=images, return_tensors="pt", padding=True)
inputs_text = processor(text=label_texts, return_tensors="pt", padding=True)

with torch.no_grad():
    image_embeddings = model.get_image_features(pixel_values=inputs_image["pixel_values"])
    text_embeddings = model.get_text_features(input_ids=inputs_text["input_ids"], attention_mask=inputs_text["attention_mask"]) # 이미지 데이터랑 텍스트 데이터를 임베딩 백터로 만드는 과정.

# 코사인 유사도 계산 (이미지 vs 라벨 텍스트)
image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)  # 정규화
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)  # 정규화
similarity_matrix = cosine_similarity(image_embeddings.cpu().numpy(), text_embeddings.cpu().numpy())

def create_thumbnail(img, size=(80,80)):
    return img.resize(size)

thumbnails = [create_thumbnail(img) for img in images]

# 시각화 (Heatmap + 이미지 썸네일 + 텍스트)
fig, ax = plt.subplots(figsize=(20, 20))

# Heatmap 그리기
sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=label_texts, yticklabels=False, ax=ax)

# 이미지 썸네일 추가 
for i, img in enumerate(thumbnails):
    imagebox = OffsetImage(img, zoom=1.0)  # 썸네일 크기 조정
    ab = AnnotationBbox(imagebox, (0, i+0.5), frameon=False, xybox=(-30, 0), xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)

plt.title("CLIP Image vs. Label Similarity (Cosine Similarity)")
plt.xlabel("Label Text")
plt.ylabel("Image")
plt.xticks(rotation=45, ha="right")  # x축 글자 기울이기


plt.savefig('./result/heatmap.png')
