#测试SVD分解并缩放是否能降低量化误差：并没有，反而增加了，为什么？我需要进行缩放操作，如何缩放？
#注意：全正数的矩阵svd分解后的正交矩阵有正有负，我的量化细节不对。
#注意：如果实在无法降低，我就试试acc效果怎样(因为误差高不等于acc低)
#注意：我使用的随机生成矩阵，矩阵似乎不经常出现离群点，怎么生成离群点多的矩阵？
import numpy as np
import torch
import os
import sys
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.dirname(SCRIPT_DIR))
import utils.util as util

# 创建一个示例矩阵
base = 2e1
X = torch.rand(3, 4)*base
W = torch.rand(4, 3)*base
X[-1][-1] = 1600
W[-1][-1] = 1600

# 进行 SVD 分解
Ux, Sx, VTx = torch.linalg.svd(X,full_matrices=False)
Uw, Sw, VTw = torch.linalg.svd(W,full_matrices=False)
Sx = torch.diag(Sx)
Sw = torch.diag(Sw)

# 缩放svd
# print("缩放前后的U,S,V是：")
# print(torch.mm(Ux, torch.mm(Sx, VTx)))
# Ux, Sx, VTx = scaleSVD(Ux, Sx, VTx)
# print(torch.mm(Ux, torch.mm(Sx, VTx)))
Ux, Sx, VTx = util.scaleSVD(Ux, Sx, VTx)
Uw, Sw, VTw = util.scaleSVD(Uw, Sw, VTw)

# 分别量化svd
Ux_hat, Sx_hat, VTx_hat = util.pTensor_quant_svd(Ux, Sx, VTx, bit=8)
Uw_hat, Sw_hat, VTw_hat = util.pTensor_quant_svd(Uw, Sw, VTw, bit=8)
# print(Ux_hat.shape, Sx_hat.shape, VTx_hat.shape)#test

print("\nX和W分别是：")
print(X)
print(W)
#查看svd情况
print("\n")
print(Ux)
print(Ux_hat)
print(Sx)
print(Sx_hat)
# print(VTx)
# print(VTx_hat)

X_hat_svd = torch.mm(Ux_hat, torch.mm(Sx_hat, VTx_hat))
W_hat_svd = torch.mm(Uw_hat, torch.mm(Sw_hat, VTw_hat))
loss_X_svd = util.cal_quant_loss(X, X_hat_svd)
print("loss_X_svd: ", loss_X_svd)
loss_W_svd = util.cal_quant_loss(W, W_hat_svd)
print("loss_W_svd: ", loss_W_svd)
loss_svd = util.cal_quant_loss(torch.mm(X,W), torch.mm(X_hat_svd, W_hat_svd))
print("loss_svd: ", loss_svd)

#-----------------------------------------
#以上是svd，以下是ord的
#-----------------------------------------

X_hat = util.pTensor_quant(X, bit=8)
W_hat = util.pTensor_quant(W, bit=8)

print("\n")
print(X)
print(X_hat)
# print(W)
# print(W_hat)
# print(torch.mm(X,W))
# print(torch.mm(X_hat,W_hat))

loss_X_ord = util.cal_quant_loss(X, X_hat)
print("loss_X_ord: ", loss_X_ord)
loss_W_ord = util.cal_quant_loss(W, W_hat)
print("loss_W_ord: ", loss_W_ord)
loss_ord = util.cal_quant_loss(torch.mm(X,W), torch.mm(X_hat,W_hat))
print("loss_ord: ", loss_ord)
