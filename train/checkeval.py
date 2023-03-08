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


    boardstrs=[
        "x.o..x.ooxoxxoxo.oox..o...oxx.xxxxo.ox.xo.o.o.oo.",
        "oxoxox.x.xoxxoxoxx.xxxo.xxx.xxoxo.xoooox.xxoxoxoo",
        "oxo..xxx.ooooo.xxxo.ooo.ooox.oo.x..xox.oo.oooxo..",
        ".xxxox.ox...o...ox..oo.....oo...o.xx.xoox.ooo...o",
        "o.o.xoxx.xxx.x..xxxx.oxxoo...x.o.o.xx..xx.xo..xxx",
        "xo.xxoxx...xxox.o....ooooxo.xxoo.xxoo.xxx.xxoxx.o",
        "xxx.xxoo.o.x.oooxx...o..o...oxoo..x....o.oxoxoo.x",
        ".x..oxox.oooo...oxx.oxxoo.x..x...xooooxoo..o.xxx.",
        "o.oooxo.x...ox.xoo.o.oo.oo.oxoooxo.o.....xxxoxo.x",
        "ox.xoox.xoxoxoxox.oo.x.x.x...o.ox...xxxxooxxx.xox",
    ]



    for boardstr in boardstrs:
        board=torch.zeros((1,2,boardH,boardW))
        for y in range(boardH):
            for x in range(boardW):
                c=boardstr[(y*boardW+x)]
                if(c=='x'):
                    board[0,0,y,x]=1
                if(c=='o'):
                    board[0,1,y,x]=1
        v,p=model.forward(board)
        v=v.detach().cpu().numpy()
        np.set_printoptions(linewidth=256,threshold=1000000)
        print(v)
        if p is not None:
            p=p.detach().cpu().numpy()
            print(np.array(p*32,dtype=np.int32))
    #model.testeval(board,7*15+7)