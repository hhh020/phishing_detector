import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import torch
import torch_directml
from read_email import extract_body_from_eml
from training import LSTMClassifier
import joblib
import re
import os

dml = torch_directml.device()
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取数据
email_file_path = os.path.join(script_dir, 'test.eml')
email_body = extract_body_from_eml(email_file_path)

# 数据预处理
## Remove hyperlinks, punctuations, extra space
def preprocess_text(text):
    # Remove hyperlinks
    text = re.sub(r'http\S+', '', text)

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

text = preprocess_text(email_body)

# 文本转换为PyTorch张量
tf = joblib.load(os.path.join(script_dir, 'tfidf_vectorizer.pkl'))  # dimension reduction
text = tf.transform([text]).toarray()
input_tensor = torch.tensor(text, dtype=torch.float32).to(dml)

# 加载模型
model_path = os.path.join(script_dir, 'model.pth')
state_dict = torch.load(model_path, map_location=dml)
input_dim = 10000
hidden_dim = 128    
layer_dim = 1       # Number of LSTM layers
output_dim = 2      # Number of unique classes

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim).to(dml)
model.load_state_dict(state_dict)
model.eval()  # 设置模型为评估模式，这会关闭诸如Dropout这样的训练时特有的层

# 添加一个时间步维度，因为LSTM模型可能期望三维输入 (batch_size, sequence_length, feature_dim)
if input_tensor.dim() == 2:
    input_tensor = input_tensor.unsqueeze(0)

# 预测
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)  # 假设是分类任务，获取最高概率的类别

# 解释结果
is_phishing = "是钓鱼邮件" if predicted_class.item() == 1 else "不是钓鱼邮件"
print(is_phishing)
