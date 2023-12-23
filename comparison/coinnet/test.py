import os
import logging
import torch.utils.data
import torch.nn.functional as F
from torch.nn import DataParallel
from core import model_nets, dataset
from core.utils import init_log, progress_bar

f=open("results.txt","w+")
testset = dataset.loader(root='CoinsDataset/', is_train=False, data_len=None)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0, drop_last=False)

# define model
net_ = model_nets.model_r50_net()
ckpt = torch.load('./models/best_model.ckpt')
net_.load_state_dict(ckpt['net_state_dict'])


net_ = net_.cuda()
# net_ = DataParallel(net_)

for i, data in enumerate(testloader):
    with torch.no_grad():
         img, label = data[0].cuda(), data[1].cuda()
         batch_size = img.size(0)
         concat_logits = net_(img)
         concat_logits.div_(torch.norm(concat_logits,2)) 
         #concat_logits = F.softmax(concat_logits)
         max_score, concat_predict = torch.max(concat_logits, 1)
         f.write("%d, %d, %.2f\n"%(concat_predict.data + 1, label + 1, max_score.data))
         #f.write("%d, "%(concat_predict.data + 1))
         progress_bar(i, len(testloader), 'eval test set')

f.close()
print('finishing Testing')
