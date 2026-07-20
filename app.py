import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.set_page_config(page_title="Phân loại rác AI", page_icon="♻️")
st.title("♻️ Trợ lý Phân loại Rác thải AI")
st.write("Hãy tải lên một bức ảnh rác, AI của chúng ta sẽ giúp bạn phân loại nó!")

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load('resnet50_eco_sorter.pth', map_location=torch.device('cpu')))
    model.eval() 
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Chọn một bức ảnh từ máy của bạn...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh bạn vừa tải lên', use_column_width=True)
    
    st.write("AI đang phân tích...")
    
    image_tensor = transform(image).unsqueeze(0) 
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]
        
    st.success(f"AI dự đoán đây là: **{result.upper()}**")
