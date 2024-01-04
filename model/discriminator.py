import torch
import torch.nn as nn
# lightening
import pytorch_lightning as pl
class Discriminator(pl.LightningModule):

    def __init__(self, source_encoder,target_encoder,hidden_dim,latent_dim):
        super().__init__()
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.automatic_optimization = False
        self.sig=nn.Sigmoid()

        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, s,t):
        z_source = self.source_encoder(s)
        z_source=z_source[:,-1,:]
        z_target = self.target_encoder(t)
        z_target=z_target[:,-1,:]

        z = torch.cat((z_source, z_target), dim=0)
        return self.discriminator(z)
    
    def training_step(self, batch, batch_idx):

        source,target, source_label,target_y = batch # x,y refers to sequence ang label(0 or 1)
        optimizer_d,optimizer_t=self.optimizers() ## ligthning 내장함수



        optimizer_d.zero_grad()
        z = self.forward(source,target)
        z=z.squeeze()
        y=torch.cat((source_label,target_y),dim=0)
        d_loss = nn.BCEWithLogitsLoss()(z, y)
        self.manual_backward(d_loss)  ## l;ightning 내장함수, 알아서 backprop해줌
        optimizer_d.step()
        y_pred=self.sig(z)
        acc = (y_pred.round() == y).float().mean()
        self.log('train_discriminator_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        ## train encoder
        optimizer_t.zero_grad()
        optimizer_d.zero_grad()
        feat_target=self.target_encoder(target)
        feat_target=feat_target[:,-1,:]
        pred_target=self.discriminator(feat_target)
        label_target=torch.ones_like(pred_target).to(self.device)
        t_loss=nn.BCEWithLogitsLoss()(pred_target,label_target)
        self.manual_backward(t_loss)
        optimizer_t.step()

        
        self.log('train_target_loss', t_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_target_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    
    def configure_optimizers(self):

        opt_d=torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        opt_t=torch.optim.Adam(self.target_encoder.parameters(), lr=1e-3)

        return [opt_d,opt_t],[]
