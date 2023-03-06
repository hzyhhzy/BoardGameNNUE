import random

from dataset import trainset
from model import ModelDic


import argparse
import glob
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time

from config import boardH,boardW





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str ,default='test', help='model path')
    args = parser.parse_args()

    modelname=args.model



    file_path = f'../saved_models/{modelname}/model.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path,map_location="cpu")
        model_type=data['model_type']
        model_param=data['model_param']
        model = ModelDic[model_type](*model_param)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, size={model_param}, totalstep={totalstep}")
    else:
        print("Invalid Model Path")
        exit(0)

    model.eval()

    boardstr='' \
    "oooooxx"\
    "oooxxxx"\
    ".ooxxxx"\
    "o.ooxx."\
    "..oooo."\
    ".xo..o."\
    "x.....o"\


    # boardstr=''
    # for i in range(boardH*boardW):
    #     c=random.randint(0,2)
    #     if(c==0):
    #         boardstr+="."
    #     if(c==1):
    #         boardstr+="x"
    #     if(c==2):
    #         boardstr+="o"


    # boardstr='' \
    # "......."\
    # "......."\
    # "......."\
    # "......."\
    # "......."\
    # "......."\
    # "......."

    print(boardstr)


    board=torch.zeros((1,2,boardH,boardW))
    for y in range(boardH):
        for x in range(boardW):
            c=boardstr[(y*boardW+x)]
            if(c=='x'):
                board[0,0,y,x]=1
            if(c=='o'):
                board[0,1,y,x]=1
    v,p=model.forward(board,debugPrint=True)
    v=v.detach().cpu().numpy()
    #v=np.exp(v)
    #v=v/np.sum(v)
    np.set_printoptions(linewidth=256,threshold=1000000)
    print(v)
    if p is not None:
        p=p.detach().cpu().numpy()
        print(np.array(p*32,dtype=np.int32))
    #model.testeval(board,7*15+7)