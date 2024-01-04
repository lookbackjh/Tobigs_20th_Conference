import torch.nn as nn
import pytorch_lightning as pl
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.latent_dim = latent_dim
        
        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x=x.unsqueeze(2)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
        out_encoder, (hn, cn) = self.encoder(x, (h0, c0))
        return out_encoder


class Forecaster(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim,output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim)
        self.forecaster=nn.Linear(hidden_dim,output_dim)

    def forward(self, x):
        #x=x.unsqueeze(2) # shape: (batch_size, seq_length, input_dim) 32, 30, 1(단변량이니ㄱ까)
        out_encoder=self.encoder.forward(x)
        out=self.forecaster(out_encoder[:,-1,:])
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z=self.forward(x)
        loss = nn.MSELoss()(z, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

