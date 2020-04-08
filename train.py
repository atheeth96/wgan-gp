import datetime
import os
from tqdm import tqdm 
import torch
import torchvision
from torch.utils.data import DataLoader


from Models import Generator,Discriminator,weights_init,save_model,load_model,set_requires_grad
from Losses import _gradient_penalty
from Dataset import Dataset,Scale,ToTensor,Normalize,visualize_loader

batch_size=64

## CREATE DATASET ##

transform=torchvision.transforms.Compose([Scale(),ToTensor(),Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset=Dataset(csv_file='fashion-mnist_train.csv'
            , transform=transform)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=Dataset(csv_file='fashion-mnist_test.csv'\
                     ,transform=transform)

test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)


test_img=visualize_loader(test_loader,0)

## DEFINE MODELS ##

modelG=Generator()
print(modelG)

modelD=Discriminator()
print(modelD)

## DEFINE DEVICE AND OPTIMIZERS ##

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained=False
lr=1e-4
beta1=0.5
optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr, betas=(beta1, 0.9))
optimizerG = torch.optim.Adam(modelG.parameters(), lr=lr, betas=(beta1, 0.9))

# INITIALIZE MODELS ##

if pretrained==False:
    modelG.apply(weights_init)
    modelD.apply(weights_init)
    

else:
    load_model('model_2020_04_07/Generator_5590.pth',modelG)
    load_model('model_2020_04_07/Discriminator_5590.pth',modelD) 
    
    
## BEGIN TRAINING ##

print(torch.cuda.get_device_name(torch.cuda.current_device()))
model_start_date=datetime.datetime.now().strftime("%Y_%m_%d")
MODEL_PATH=os.path.join(os.getcwd(),'model_{}'.format(model_start_date))
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
    print('{} dir has been made'.format(MODEL_PATH))
no_steps=train_dataset.__len__()//batch_size
restart_epochs=10
num_epochs=200
best_ssim=0
lambda_gp=10
critic_iter=5
gen_iter=1

modelD = modelD.to(device)
modelG = modelG.to(device)

for epoch in range(7,num_epochs):
    
    print("Learning Rate Generator : {}\nLearning Rate Discriminator : {}"\
          .format(optimizerG.state_dict()['param_groups'][-1]['lr'],\
                  optimizerD.state_dict()['param_groups'][-1]['lr']))
    gen_adv_score=0
    Wasserstein_loss=0
  # loop over the dataset multiple times

    modelD.train()
    modelG.train()
    loop=tqdm(train_loader)
    
    gen_train_steps=0

    for i, sample_batched in (enumerate(loop)):
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # get the inputs;
        img_batch, label_batch= sample_batched['img'],\
        sample_batched['label']

        img_batch, label_batch = img_batch.to(device, dtype = torch.float)\
        ,label_batch.to(device, dtype = torch.float)
        

        if (i+1)%critic_iter==0:
        ## Generator training ##
            
            set_requires_grad(modelD, requires_grad=False)
            modelG.zero_grad()
            
            random_noise=torch.randn((img_batch.size(0),128)).to(device)
            fake_images_batch=modelG(random_noise)

            loss_G=-(modelD(fake_images_batch).mean())
            
            loss_G.backward()
            optimizerG.step()
            
            gen_adv_score=loss_G.detach().item()

            
            if (i+1)%100==0:

                img_train = torchvision.utils.make_grid(fake_images_batch.detach().cpu()\
                                                        ,nrow=batch_size//4,padding=40)
            
                torchvision.utils.save_image(img_train,os.path.join(os.getcwd(),\
                                                                    MODEL_PATH+'/train_iter_{}.png'.\
                                                                    format(epoch*len(train_loader)+i+1)))
                save_model(modelD,optimizerD,os.path.join(MODEL_PATH\
                                                          ,'Discriminator_{}.pth'.format\
                                                          (epoch*len(train_loader)+i+1)),scheduler=None)
                save_model(modelG,optimizerG,os.path.join(MODEL_PATH,\
                                                          'Generator_{}.pth'.format\
                                                          (epoch*len(train_loader)+i+1)),scheduler=None)
        
            
        else:
            

            #Train discriminator
            random_noise_dis=torch.randn((img_batch.size(0),128)).to(device)

            set_requires_grad(modelD, requires_grad=True)
            modelD.zero_grad()
            
            fake_images_batch_dis=modelG(random_noise_dis).detach()
            D_fake_score=modelD(fake_images_batch_dis).mean()

            D_real_score=modelD(img_batch).mean()
            
            gp=_gradient_penalty(img_batch, fake_images_batch_dis,modelD,device)
           
            loss_D=(D_fake_score-D_real_score+lambda_gp*gp)
            
            loss_D.backward()
            optimizerD.step()
            
            Wasserstein_loss=loss_D.detach().item()
            
        loop.set_postfix(Genrated_score=gen_adv_score,Wasserstein_loss=Wasserstein_loss)

save_model(modelG,optimizerG,os.path.join(MODEL_PATH,'Generator_final.pth'),scheduler=None)
print('Finished Training')