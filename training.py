import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import joblib
import re

dml = torch_directml.device()

## convert the categorical label into numerical
def label_encoder(df, column_name):
    # 创建一个字典来保存类别与编码之间的映射
    label_mapping = defaultdict(int)
    for i, label in enumerate(df[column_name].unique()):
        label_mapping[label] = i
    # 使用映射转换列的值
    df[column_name] = df[column_name].map(label_mapping)
    return df, label_mapping

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

class EmailDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 训练LSTM模型
## 定义LSTM模型，包括一个LSTM层、一个全连接层用于分类任务
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def init_hidden(self, batch_size):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(dml)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(dml)
        return (h0, c0)

    def forward(self, x):
        batch_size = x.size(0)
        h0, c0 = self.init_hidden(batch_size)  # Initialize hidden and cell states
        
        # Adding singleton time-step dimension if the input is not already a sequence
        x = x.unsqueeze(1) if x.dim() == 2 else x  # This safely adds a dimension only if needed
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Get the last time step output for classification
        out = self.fc(out[:, -1, :])
        
        return out
'''
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(dml)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(dml)
        
        # Reshape h0 and c0 to match the expected shape when input is not batched
        if batch_size == 1:  # Check if it's an unbatched input
            h0 = h0.squeeze(0)  # Remove the batch dimension
            c0 = c0.squeeze(0)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x.unsqueeze(1), (h0.detach(), c0.detach()))  # Add a singleton time-step dimension if input is unbatched
        
        # Get last time step output
        out = self.fc(out[:, -1, :])
        
        return out
'''

if __name__ == "__main__":

    # 加载数据集
    df = pd.read_csv('/home/wzh/dp/data/Phishing_Email.csv')

    # 数据预处理
    ## Drop duplicate and null values
    df.drop(['Unnamed: 0'],axis=1,inplace=True)
    df.dropna(inplace=True,axis=0)
    df.drop_duplicates(inplace=True)

    df_encoded, mapping = label_encoder(df, 'Email Type')
    df_encoded["Email Text"] =df_encoded["Email Text"].apply(preprocess_text)

    # 文本转换为PyTorch张量
    tf = TfidfVectorizer(stop_words='english', max_features=10000)  # dimension reduction
    feature_x = tf.fit_transform(df_encoded['Email Text']).toarray()
    y_tf = np.array(df_encoded['Email Type'])  # convert the label into numpy array
    X_tr, X_tst, y_tr, y_tst = train_test_split(feature_x, y_tf, test_size=0.2, random_state=0)
    joblib.dump(tf, 'tfidf_vectorizer.pkl')  # 保存transformer

    ## Convert numpy arrays to PyTorch tensors
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32).to(dml)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long).to(dml)
    X_tst_tensor = torch.tensor(X_tst, dtype=torch.float32).to(dml)
    y_tst_tensor = torch.tensor(y_tst, dtype=torch.long).to(dml)
    ## Create Dataset objects and DataLoader
    train_dataset = EmailDataset(X_tr_tensor, y_tr_tensor)
    test_dataset = EmailDataset(X_tst_tensor, y_tst_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    ## 实例化模型并设定损失函数和优化器
    input_dim = X_tr_tensor.shape[1]  # Assuming the second dimension is the feature size
    hidden_dim = 128  # You can adjust this
    layer_dim = 1  # Number of LSTM layers
    output_dim = len(np.unique(y_tf))  # Number of unique classes

    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim).to(dml)
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ## 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(dml))
            loss = criterion(outputs, labels.to(dml))
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 评估模型
    model_save_path = "/home/wzh/dp/phishing-email-detection/model.pth"

    # 使用torch.save()保存模型的state_dict
    torch.save(model.state_dict(), model_save_path)

    print(f"模型已保存至: {model_save_path}")