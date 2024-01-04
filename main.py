

from model.encoder import Encoder,Forecaster
#from model.forecaster import Forecaster
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from model.discriminator import Discriminator
class Discriminative_Dataset(Dataset):
    def __init__(self, source,target, source_label,target_label):
        self.source = source
        self.target = target
        self.len = len(source)
        self.source_label=source_label
        self.target_label=target_label
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return torch.tensor(self.source[idx],dtype=torch.float32),torch.tensor(self.target[idx],dtype=torch.float32),torch.tensor(self.source_label[idx],dtype=torch.float32),torch.tensor(self.target_label[idx],dtype=torch.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def large_categories_getter(source_idx,target_idx):
    a=pd.read_csv('data/train.csv')
    largeones=a['대분류'].unique()
    smallones=a['소분류'].unique()
    # find row where 대분류 is largeones[0]
    largerows=a.loc[(a['대분류'] == largeones[source_idx])] 
    targets=largerows['소분류'].unique()


    targetrows=largerows.loc[(largerows['소분류'] == targets[target_idx])]
    # sum every row
    targetrows.drop(['ID','대분류','중분류','소분류','브랜드','제품'],axis=1,inplace=True) 
    targetrows=targetrows.sum(axis=0)

    # sum every row
    largerows.drop(['ID','대분류','중분류','소분류','브랜드','제품'],axis=1,inplace=True) 
    sourcerows=largerows.sum(axis=0)
    return sourcerows,targetrows


def small_categories_getter():

    a=pd.read_csv('data/train.csv')
    largeones=a['대분류'].unique()
    smallones=a['소분류'].unique()
    # find row where 대분류 is largeones[0]
    smallrows=a.loc[(a['소분류'] == smallones[0])] 
    # sum every row
    smallrows.drop(['ID','대분류','중분류','소분류','브랜드','제품'],axis=1,inplace=True) 
    totalsmallrows=smallrows.sum(axis=0)
    return totalsmallrows

def create_sequence(df, seq_length,pred_length):
    xs = []
    ys = []
    for i in range(len(df)-seq_length-pred_length):
        x = df[i:(i+seq_length)]
        y = df[(i+seq_length):(i+seq_length+pred_length)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def preprocessing(xs,ys):
    ## train test split
    train_len=int(len(xs)*0.8)
    train_x=xs[:train_len]
    train_y=ys[:train_len]
    test_x=xs[train_len:]
    test_y=ys[train_len:]

    # min max scaler
    x_scaler=MinMaxScaler()
    y_scaler=MinMaxScaler()
    train_x=x_scaler.fit_transform(train_x)
    train_y=y_scaler.fit_transform(train_y)
    test_x=x_scaler.transform(test_x)
    test_y=y_scaler.transform(test_y)

    # train test split
    return train_x,train_y,test_x,test_y
    
def main():
    ## 1. 데이터 불러오기-> 대분류
    source_idx=0  # 대분류 총 5개 , 0,1,2,3,4 선택 가능. 
    target_idx=0 # 각 대분류마다 할당된 소분류-> 조금씩 다름
    source_data,target_data=large_categories_getter(source_idx,target_idx)
    

    ## 2. 데이터 전처리
    seq_length=100 # 얼마의 기간을 가지고 다음 기간을 예측할 것인가
    pred_length=50 # 다음 기간을 얼마나 예측할 것인가 ex) 다음 5일에 대한 예측치를 한번에 제공
    xs,ys=create_sequence(source_data,seq_length,pred_length)
    xt,yt=create_sequence(target_data,seq_length,pred_length)
    
    # 전처리
    train_xs,train_ys,test_xs,test_ys=preprocessing(xs,ys) # train test split, min max scaler    
    train_xt,train_yt,test_xt,test_yt=preprocessing(xt,yt) # train test split, min max scaler

    train_s_label=np.ones(len(train_xs))
    train_t_label=np.zeros(len(train_xt))

    discriminative_dataset=Discriminative_Dataset(train_xs,train_xt,train_s_label,train_t_label)
    discriminative_loader=DataLoader(discriminative_dataset,batch_size=32,shuffle=True)

    # dataset, dataloader
    train_dataset_source=TimeSeriesDataset(train_xs,train_ys)
    test_dataset_source=TimeSeriesDataset(test_xs,test_ys)
    train_loader_source=DataLoader(train_dataset_source,batch_size=32,shuffle=True)
    test_loader_sourcer=DataLoader(test_dataset_source,batch_size=32,shuffle=False)

    


    ## ADDA 1. Source 모델 훈련시키기
    source_model = Forecaster(input_dim=1, hidden_dim=64,output_dim=pred_length) #LSTM Encoder
    source_trainer=pl.Trainer(max_epochs=200)
    source_trainer.fit(source_model,train_loader_source,test_loader_sourcer)



    ## ADDA 2. Discriminator 훈련, Target Encoder 훈련 (번갈아 가면서)
    target_encoder=Encoder(input_dim=1, hidden_dim=64)
    discriminator=Discriminator(source_model.encoder,target_encoder, hidden_dim=64, latent_dim=64)
    discriminator_trainer=pl.Trainer(max_epochs=200)
    discriminator_trainer.fit(discriminator,discriminative_loader)

    ## ADDA 3. Evaluation
    # 1) Source 모델 평가
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_model.eval()
    source_model=source_model.to(device)

    test_xt=torch.tensor(test_xt,dtype=torch.float32).to(device)
    #test_xt=test_xt.unsqueeze(2)
    source_output=source_model.forward(test_xt)
    source_output=source_output.squeeze()
    source_output=source_output.detach().cpu().numpy()
    # evaluate rmse
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmse = sqrt(mean_squared_error(test_yt[:,49], source_output[:,49]))
    print('Test RMSE using source only: %.3f' % rmse)
    
    
    # 2) Target 모델 평가
    target_encoder.eval()
    target_encoder=target_encoder.to(device)
    target_output=target_encoder.forward(test_xt)
    tar_pred=source_model.forecaster(target_output[:,-1,:])
    tar_pred=tar_pred.squeeze()
    tar_pred=tar_pred.detach().cpu().numpy()
    # evaluate rmse
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmse = sqrt(mean_squared_error(test_yt[:,49], tar_pred[:,49]))
    print('Test RMSE using target encoder trained: %.3f' % rmse)







if __name__ == '__main__':
    main()





