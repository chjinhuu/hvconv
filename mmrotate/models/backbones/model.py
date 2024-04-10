import torch


model = torch.load("/mnt/csip-108/vision-main/references/classification/HVBlock_ImageNet_pretrain_batchsize256/model_599.pth")
model = model['model_ema'] # ['state_dict_ema']

# model.pop('n_averaged')
model_out = {}

for m in model:
    model_out[m[7:]] = model[m]

torch.save(model_out, "/mnt/csip-108/vision-main/references/classification/HVBlock_ImageNet_pretrain_batchsize256/model.pth")