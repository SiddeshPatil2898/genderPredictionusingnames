#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install torch --upgrade')
get_ipython().system('pip install torchtext --upgrade')
get_ipython().system('pip install torchvision')


# In[8]:





# In[32]:


import torch
import torch.nn as nn
import pandas as pd 
from torchtext.vocab import vocab
from collections import Counter
from sklearn.metrics import f1_score
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from sklearn.metrics import accuracy_score
from sklearn.metrics import  confusion_matrix
male_names=pd.read_csv(r'https://gist.githubusercontent.com/mbejda/7f86ca901fe41bc14a63/raw/38adb475c14a3f44df9999c1541f3a72f472b30d/Indian-Male-Names.csv')
female_names =pd.read_csv(r'https://gist.githubusercontent.com/mbejda/9b93c7545c9dd93060bd/raw/b582593330765df3ccaae6f641f8cddc16f1e879/Indian-Female-Names.csv')
names_df = pd.concat([male_names, female_names])
names_df.to_csv("indian_names.csv", index= False )
names_df = pd.read_csv("indian_names.csv")
names_df.dropna(inplace=True)
names_df=names_df.sample(frac=1)
train_size = int(len(names_df)*.8)
train_data = names_df[:train_size]
test_data = names_df[train_size:]
all_chars = [t for text  in names_df['name'] for t in text if text is not None] 
char_count = Counter(all_chars)
name_char_vocab = vocab(char_count)
class NamesDataset(Dataset):
    
    def __init__(self,names_df,name_char_vocab):
        self.names_df=names_df
        self.name_char_vocab=name_char_vocab
        self.gender_dict = {'m':0, 'f':1}
        self.rev_gender_dict = {v:k for k,v in self.gender_dict.items()}
        
        
    def __len__(self):
        return len(self.names_df)
    
    def __getitem__(self, idx):
        item = self.names_df.iloc[idx, :]
        label = torch.zeros(2)
        label[self.gender_dict[item['gender']]] = 1
        name = self.get_names_tensor(item['name'])
        return name, torch.tensor(self.gender_dict[item['gender']])
    
    def get_names_tensor(self, name ):
        name_ids = self.name_char_vocab.lookup_indices([t for t in  name])
        name_tensor = torch.as_tensor(name_ids, dtype = int)
        return name_tensor
    
    def get_category_from_idx(self, idx ):
        
        return self.rev_gender_dict[idx]
train_ds = NamesDataset(train_data,name_char_vocab)
train_ds.get_names_tensor('pankaj')
class NamesClassifier(nn.Module):
    
    def __init__(self, size):
        super(NamesClassifier, self).__init__()
        self.embedding = nn.Embedding(size,128)
        self.rnn = nn.LSTM(128,256)
        #self.rnn = nn.RNN(128, 256)
        self.linear1 = nn.Linear(256,256)
        self.relu1= nn.ReLU()
        self.linear2 = nn.Linear(256,2)
    
    def forward(self, ip):
        op= self.embedding(ip)
        op, hi = self.rnn(op)
        #op, hi = self.lstm(op)
        output = self.linear1(hi[0])
        output = self.relu1(output)
        output = self.linear2(output)
        return output
def predict(name, model1):
    names_tensor = train_ds.get_names_tensor(name)
    output = model1(names_tensor)
    category_idx = output.topk(1)[1].item()
    category = train_ds.get_category_from_idx(category_idx)
    return category
model = NamesClassifier(len(train_ds.name_char_vocab))
train_data['predicted'] = train_data.name.apply(lambda x: predict(x, model))
confusion_matrix(train_data.predicted, train_data.gender)
criteria = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters())
num_step= len(train_ds)
# num_step=6
step =0
total_loss=0
for  i in range(0, train_size):
    name_ip, label = train_ds[i]
    step=step+1
    optimizer.zero_grad()
    op= model(name_ip)
    loss = criteria(op.squeeze(), label)
    loss.backward()
    optimizer.step()
    total_loss=loss+total_loss
    if step%1000==0:
        print(total_loss)
        total_loss=0
predicted = [predict(n, model) for n in test_data.name]
test_data['predicted'] = test_data.name.apply(lambda x: predict(x, model))
train_data['predicted'] = train_data.name.apply(lambda x: predict(x, model))
confusion_matrix(test_data.predicted, test_data.gender)
accuracy_score( test_data.predicted, test_data.gender )
train_data['predicted'] = train_data.name.apply(lambda x: predict(x, model))
confusion_matrix(train_data.predicted, train_data.gender)
train_data['predicted'] = train_data.name.apply(lambda x: predict(x, model))
test_data[test_data.predicted==test_data.gender].head() 
actual = test_data.gender
confusion_matrix(predicted,actual)
accuracy_score( train_data.predicted, train_data.gender )


# In[27]:


predict('darshana',model)


# In[28]:


predict('rutvij',model)


# In[29]:


predict('kia',model)


# In[ ]:




