'''
evaluation
'''
import argparse
import os
import random
import time
from tqdm import tqdm
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python eval.py

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
parser.add_argument('--iters', type=int, default = 500, help='start epoch')

parser.add_argument('--save_root_dir', type=str, default='./results',  help='output folder')
parser.add_argument('--model', type=str, default = 'best_model.pth',  help='model name for training resume')
parser.add_argument('--test_path', type=str, default = '../dataset',  help='model name for training resume')
parser.add_argument('--protocal', type=str, default = 's0',  help='model name for training resume')

parser.add_argument('--dataset', type=str, default = 'dexycb', help='optimizer name for training resume')
parser.add_argument('--model_name', type=str, default = 'handdiff',  help='')
parser.add_argument('--gpu', type=str, default = '3',  help='gpu')

parser.add_argument('--test_step', type=int, default = 5,  help='number of test timesteps')
parser.add_argument('--test_hypo', type=int, default = 5,  help='number of test hypotheses')

opt = parser.parse_args()
print (opt)

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


if opt.dataset == 'dexycb':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_'+opt.protocal +'_' + opt.model_name+'_'+str(opt.iters)+'iters'+'_com')
	from dataloader import loader 
	opt.JOINT_NUM = 21
elif opt.dataset == 'nyu':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+ str(opt.iters)+'iters'+'_com')
	from dataloader import loader 
	opt.JOINT_NUM = 14
	# opt.JOINT_NUM = 23
	# calculate = [0, 2, 
    #          4, 6,
    #          8, 10, 
    #          12, 14,
    #          16, 17, 18,
    #          20, 21, 22]


# 1. Load data                                         
if opt.dataset == 'dexycb' :
	test_data = loader.DexYCBDataset(opt.protocal, 'test', opt.test_path)
elif opt.dataset == 'nyu':
	test_data = loader.nyu_loader(opt.test_path, 'test', joint_num=opt.JOINT_NUM)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers), pin_memory=False)
                                          
print('#Test data:', len(test_data))
print (opt)

# 2. Define model, loss
model = getattr(module, 'HandModel')(joints=opt.JOINT_NUM, iters=opt.iters)

if opt.ngpu > 1:
    model.netR_1 = torch.nn.DataParallel(model.netR_1, range(opt.ngpu))
    model.netR_2 = torch.nn.DataParallel(model.netR_2, range(opt.ngpu))
    model.netR_3 = torch.nn.DataParallel(model.netR_3, range(opt.ngpu))
if opt.model != '':

	model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)), strict=False)
		
model.cuda()
# print(model)

criterion = nn.MSELoss(size_average=True).cuda()

# 3. evaluation
torch.cuda.synchronize()

model.eval()
test_mse = 0.0
test_wld_err = 0.0
test_wld_err_mean = 0.0

timer = 0

saved_points = []
saved_gt = []
saved_fold1 = []
saved_final = []
saved_error = []
saved_length = []

h = opt.test_hypo
step = opt.test_step
for i, data in enumerate(tqdm(test_dataloader, 0)):

	torch.cuda.synchronize()
	with torch.no_grad():
		# 3.2.1 load inputs and targets
		if opt.dataset == "nyu":
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length = data
			volume_length = volume_length.cuda()
		else:
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
			volume_length = 250.

		points, gt_xyz, img = points.cuda(),  gt_xyz.cuda(), img.cuda()
		center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()

		t = time.time()
		estimation = model.sample(points.transpose(1,2), points.transpose(1,2), img, test_data, center, M, cube, cam_para, h, step)
		timer += time.time() - t

	torch.cuda.synchronize()

	outputs_xyz = torch.zeros_like(estimation).to(estimation.device)
	outputs_xyz[:,0] = estimation[:, 0]
	for k in range(1, h):
		outputs_xyz[:,k] = (estimation[:, k] + outputs_xyz[:, k-1] * k)/(k+1)
	diff = torch.pow(outputs_xyz.transpose(2,3)-gt_xyz.unsqueeze(1).repeat(1,h,1,1), 2).view(-1, h, opt.JOINT_NUM,3) # B, H, J, 3
	diff_sum = torch.sum(diff, -1)# B, H, J
	diff_sum_sqrt = torch.sqrt(diff_sum)# B, H, J
	if opt.dataset == 'nyu' and opt.JOINT_NUM != 14:
		diff_sum_sqrt = diff_sum_sqrt[:, :,calculate]
	diff_mean = torch.mean(diff_sum_sqrt,2).view(-1, h, 1)
	diff_mean_wld = torch.mul(diff_mean, volume_length / 2 if opt.dataset == 'dexycb' else volume_length.view(-1, 1, 1)/2)
	
	test_wld_err = test_wld_err + diff_mean_wld.sum(0).cpu()

	outputs_xyz = estimation.mean(1).transpose(1,2)
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
	diff_sum = torch.sum(diff,2)
	diff_sum_sqrt = torch.sqrt(diff_sum)
	if opt.dataset == 'nyu' and opt.JOINT_NUM != 14:
		diff_sum_sqrt = diff_sum_sqrt[:, calculate]
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
	diff_mean_wld = torch.mul(diff_mean,volume_length / 2 if opt.dataset == 'dexycb' else volume_length.view(-1, 1)/2)
	test_wld_err_mean = test_wld_err_mean + diff_mean_wld.sum().item()

# time taken
torch.cuda.synchronize()
# timer = time.time() - timer
timer = timer / len(test_data)
print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

# print mse
print('average estimation error in world coordinate system: ')
print(test_wld_err.squeeze() / len(test_data))
print(test_wld_err_mean / len(test_data))
