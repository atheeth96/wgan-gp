import torch

def _gradient_penalty(real_data, generated_data,model_D,device):
    bs = real_data.size()[0]


    alpha = torch.rand(bs, 1, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated.requires_grad_(True)
    
    interpolated = interpolated.to(device)
    

    
    interpolated_outputs = model_D(interpolated)

    gradients = torch.autograd.grad(outputs=interpolated_outputs, inputs=interpolated,
                           grad_outputs=torch.ones(interpolated_outputs.size()).to(device),
                           create_graph=True, retain_graph=True)[0]


    gradients_norm = gradients.norm(2, dim=1)

    return ((gradients_norm - 1) ** 2).mean()