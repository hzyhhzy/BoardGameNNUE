
from model import Model_vo3
from config import boardH,boardW

import argparse
import numpy as np
import torch
import os
import time
import shutil

try:
    os.mkdir("../export")
except:
    pass
else:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        default=0, help='which gpu')
    parser.add_argument('--cpu', action='store_true', default=False, help='whether use cpu')
    parser.add_argument('--model', type=str ,default='test', help='model path')
    parser.add_argument('--export', type=str ,default='', help='export path')
    parser.add_argument('--copy', action='store_true', default=False, help='copy a backup for this model, for selfplay training')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    if(args.cpu):
        device=torch.device('cpu')
    modelname=args.model
    exportname=args.export
    if(exportname==''):
        exportname=modelname






    file_path = f'../saved_models/{modelname}/model.pth'
    model_type=None
    if os.path.exists(file_path):
        data = torch.load(file_path, map_location=device)
        model_type = data['model_type']
        model_param = data['model_param']
        if(model_type != "vo3"):
            print(f"Invalid Model Type: {model_type}")
            exit(0)
        model = Model_vo3(*model_param).to(device)

        model.load_state_dict(data['state_dict'])
        totalstep = data['totalstep']
        print(f"loaded model: type={model_type}, param={model.model_param}, totalstep={totalstep}")
    else:
        print(f"Invalid Model Path: {file_path}")
        exit(0)


    #copy model file
    if(args.copy):
        modeldir='../export/'+modelname
        try:
            os.mkdir(modeldir)
        except:
            pass
        else:
            pass
        modelDestPath=modeldir+'/'+str(totalstep)+'.pth'
        shutil.copy(file_path,modelDestPath)


    model.eval()



    time0=time.time()
    feature_c, mlp_c, _, _, _ = model_param
    print(f"Start: feature_c={feature_c}, mlp_c={mlp_c}")
    exportPath='../export/'+exportname+'.txt'
    exportfile=open(exportPath,'w')

# file head
# -------------------------------------------------------------
    print("vo3",file=exportfile)
    print(feature_c,mlp_c,end=' ',file=exportfile)
    print('',file=exportfile)
#export featuremap
#-------------------------------------------------------------
    print("Exporting mapping")
    print("mapping",file=exportfile)

    mapping=model.exportMapping(device=device)
    mapping=mapping.flatten()

    scale_now=1 #这个变量存储int16和float的换算比
    bound=np.abs(mapping).max()#这个变量存储上界，时刻注意int16溢出
    print("Mapping max=",bound)


    map_maxint=32700/25
    w_scale=map_maxint/bound #w_scale表示这一步的倍数

    scale_now*=w_scale
    bound*=w_scale
    maxint=bound #maxint表示导出权重的最大值,此处恰好为bound
    for i in range(mapping.shape[0]):
        print(int(mapping[i]*w_scale),end=' ',file=exportfile)
    print('',file=exportfile)

    scale_now *= 25 #model.py里是取平均，cpp里是相加
    bound *= 25 #model.py里是取平均，cpp里是相加
    print("Bound after mapping = ",bound)
    print("Scale after mapping = ", scale_now)


# export others
    print("Finished mapping, now exporting others")
# -------------------------------------------------------------

    # prelu1
    # prelu1本身对scale和bound无影响，w_scale=1
    prelu1_w = model.prelu1.export_slope()

    maxint = np.abs(prelu1_w * 2 ** 15).max()  # mulhrs右移15位
    if (maxint > 32760):
        print("Error! Maxint=", maxint)
        exit(0)

    # write
    print("prelu1_w", file=exportfile)
    for i in range(feature_c):
        print(int(prelu1_w[i] * 2 ** 15), end=' ', file=exportfile)  # mulhrs右移15位
    print('', file=exportfile)


    #mlp layer 1
    w=model.mlp1.weight.data.cpu().numpy()
    # 从这里开始全是float了
    w=w/scale_now
    b=model.mlp1.bias.data.cpu().numpy()

    print("mlp_w1",file=exportfile)
    for i in range(feature_c):
        for j in range(mlp_c):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_b1",file=exportfile)
    for i in range(mlp_c):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)

    #mlp layer 2
    w=model.mlp2.weight.data.cpu().numpy()
    b=model.mlp2.bias.data.cpu().numpy()

    print("mlp_w2",file=exportfile)
    for i in range(mlp_c):
        for j in range(mlp_c):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlp_b2",file=exportfile)
    for i in range(mlp_c):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)

    #mlpfinal
    w=model.mlp3.weight.data.cpu().numpy()
    b=model.mlp3.bias.data.cpu().numpy()

    print("mlpfinal_w",file=exportfile)
    for i in range(mlp_c):
        for j in range(3):
            print(w[j][i],end=' ',file=exportfile)
    print('',file=exportfile)

    print("mlpfinal_b",file=exportfile)
    for i in range(3):
        print(b[i],end=' ',file=exportfile)
    print('',file=exportfile)


    exportfile.close()



    #copy txt file
    if(args.copy):
        exportCopyDestPath=modeldir+'/'+str(totalstep)+'.txt'
        shutil.copy(exportPath,exportCopyDestPath)


    print("success")






