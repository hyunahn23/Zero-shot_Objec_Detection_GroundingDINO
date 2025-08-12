from transformers import CLIPModel, CLIPProcessor
from PIL import Image


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

txt = ['a piece of sushi','a dog', 'a banana']
img = Image.open(r'C:\Users\hyuna\OneDrive\바탕 화면\code\Pytorch\MyWork\multimodal\data\a.jpeg')



inputs = processor(text=txt, images=img, return_tensors="pt", padding=True)
print(inputs.keys())
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)

