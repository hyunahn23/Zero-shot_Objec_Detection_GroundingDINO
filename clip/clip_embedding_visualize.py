import torch
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embeddings(images, texts):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"]).cpu().numpy()
        text_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).cpu().numpy()
    return image_embeds, text_embeds

# COCO 데이터셋 로드 (샘플 데이터 사용)
dataset = load_dataset("clip-benchmark/wds_flickr8k")
# dataset = load_dataset("clip-benchmark/wds_imagenetv2")
train_dataset = dataset['test']
sample_size = 100  # 사용할 샘플 개수
subset = train_dataset.select(range(sample_size))

# 이미지와 캡션 추출
images = subset['jpg']
captions = subset["txt"]

# file_path = r'C:\Users\hyuna\OneDrive\바탕 화면\code\Pytorch\MyWork\multimodal\data\classnames.txt'
# cls2label = open(file_path, 'r').readlines()
# images = subset['webp']
# captions = [cls2label[sub].rstrip() for sub in subset["cls"]]


# CLIP 임베딩 계산
image_embeds, text_embeds = get_clip_embeddings(images, captions)

# UMAP 차원 축소
all_embeds = np.concatenate([image_embeds, text_embeds], axis=0)
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
reduced_embeds = umap_model.fit_transform(all_embeds)

# 이미지 및 텍스트 좌표 분리
image_coords = reduced_embeds[:sample_size]
text_coords = reduced_embeds[sample_size:]

# 시각화
fig, ax = plt.subplots(figsize=(30, 30))
# ax.scatter(image_coords[:, 0], image_coords[:, 1], color='blue', label='Images', alpha=0.5)
ax.scatter(text_coords[:, 0], text_coords[:, 1], color='red', label='Captions', alpha=0.5)

# 이미지 및 캡션 표시
def plot_with_images(ax, coords, images, captions, is_text=False):
    for i, (x, y) in enumerate(coords):
        if is_text:
            ax.text(x, y, captions[i][:30], fontsize=8, color='red', ha='right')  # 캡션 일부 표시
        else:
            img = images[i].resize((64, 64))  # 이미지 크기 조정
            imagebox = OffsetImage(img, zoom=1.0)  # 이미지 크기 축소 (zoom 조정 가능)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

# plot_with_images(ax, image_coords, images, captions, is_text=False)
plot_with_images(ax, text_coords, images, captions, is_text=True)

ax.legend()
plt.savefig('./result/test2.png')
