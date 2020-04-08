import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self,input_dim=128,norm=nn.BatchNorm2d):
        
        super().__init__()
        self.input_dim=input_dim
        
        use_bias=norm==nn.BatchNorm2d
        
        
        self.linear=nn.Sequential(
            nn.Linear(128, 4*4*4*64,bias=True),
            nn.ReLU(True))
        
        self.block_1=nn.Sequential(
            nn.ConvTranspose2d(4*64,2*64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(True),
        )
        
        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(2*64, 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(True),
        )
        
    
        self.deconv_out=nn.ConvTranspose2d(64, 1,kernel_size=4,stride=2,padding=1)
        
        self.tanh = nn.Tanh()
          
        
    
    def forward(self,input):
        x=self.linear(input)
        
        x = x.view(-1, 4*64, 4, 4)
        
        x=self.block_1(x)
        
        x=self.block_2(x)
      
        x=self.deconv_out(x)

        output=self.tanh(x)
        
        return output


    
class Discriminator(nn.Module):
    def __init__(self,norm=nn.BatchNorm2d):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(64, 2*64, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*64, 4*64, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.output = nn.Linear(4*4*4*64, 1)

        
    def forward(self,input):
        
        x=self.main(input)
        x = x.view(-1, 4*4*4*64) 
        output=self.output(x)
        
        return output
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
def save_model(model,optimizer,name,scheduler=None):
    if scheduler==None:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    else:
        checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()}

    torch.save(checkpoint,name)

def load_model(filename,model,optimizer=None,scheduler=None):
    checkpoint=torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("Done loading")
    if  optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
    if  scheduler:
        scheduler.load_state_dict(checkpoint['optimizer'])
        print(scheduler.state_dict()['param_groups'][-1]['lr'],' : Learning rate')
        
def set_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad