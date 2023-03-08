
import torch
import torch.nn as nn
import numpy as np
from config import *
input_c=2




class Conv0dLayer(nn.Module):
    def __init__(self,in_c,out_c,groups=1):
        super().__init__()
        self.conv=nn.Conv2d(in_c,out_c,1,1,0,groups=groups)
        self.bn=nn.BatchNorm2d(out_c)
        self.act=nn.Mish()


    def forward(self, x):
        y=self.conv(x)
        y=self.bn(y)
        y=self.act(y)
        return y

class Conv0dResBlock(nn.Module):
    def __init__(self,in_c,mid_c,groups=1):
        super().__init__()
        self.layer1=Conv0dLayer(in_c,mid_c)
        self.layer2=Conv0dLayer(mid_c,in_c)


    def forward(self, x):
        y=self.layer1(x)
        y=self.layer2(y)
        y=y+x
        return y


class PRelu1(nn.Module):
    def __init__(self,c,bias=True,bound=0):
        super().__init__()
        self.c=c
        self.slope = nn.Parameter(torch.ones((c))*0.5,True)
        self.bias = nn.Parameter(torch.zeros((c)),True)
        self.useBias=bias
        self.bound=bound


    def forward(self, x, dim=1):
        xdim=len(x.shape)
        wshape=[1 for i in range(xdim)]
        wshape[dim]=-1

        slope = self.slope.view(wshape)
        if(self.bound>0):
            slope=torch.tanh(slope/self.bound)*self.bound

        y=x
        if(self.useBias):
            y=y+self.bias.view(wshape)

        y=torch.maximum(y,slope*y)
        return y

    def export_slope(self):
        slope = self.slope
        if(self.bound>0):
            slope=torch.tanh(slope/self.bound)*self.bound
        return slope.data.cpu().numpy()




class Model_vov1(nn.Module):
    def __init__(self, c=128, mlpc1=8, mlpc2=64, b=6, midc=512, mapbound=100):
        super().__init__()
        self.model_type = "vov1"
        self.model_param = (c, mlpc1, mlpc2, b, midc,mapbound)
        self.c=c
        self.mlpc1=mlpc1
        self.mlpc2=mlpc2
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        #mapping
        self.loc_embedding=torch.zeros((1,boardH*boardW,boardH,boardW))
        for y in range(boardH):
            for x in range(boardW):
                self.loc_embedding[0,y*boardW+x,y,x]=1
        self.loc_embedding=nn.Parameter(self.loc_embedding,requires_grad=False)

        self.conv1=nn.Conv2d(2+boardH*boardW,midc,3,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc1)
        self.mlp2=nn.Linear(mlpc1,mlpc2)
        self.mlp3=nn.Linear(mlpc2,1)

    def forward(self, x):
        #mapping
        y=torch.cat((x,torch.repeat_interleave(self.loc_embedding,dim=0,repeats=x.shape[0])),dim=1)
        y = self.conv1(y)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.midc,5*5))
        y=torch.einsum("nch,hco->noh",y,self.mappingFinalWeights)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==5*5)

        if (self.mapbound != 0):
            y = self.mapbound * torch.tanh(y / self.mapbound)

        y=torch.mean(y,dim=2)

        #mlp
        y=self.prelu1(y)
        y=self.mlp1(y)
        y=torch.relu(y)
        y=self.mlp2(y)
        y=torch.relu(y)
        v=self.mlp3(y)

        return v, None

    def exportMapping(self,device='cuda'):
        #first, generate [3^9,2,3,3]
        x=torch.zeros((1,2,3*3))
        for i in range(9):
            x1=torch.clone(x)
            x1[:,0,i]=1.0
            x2=torch.clone(x)
            x2[:,1,i]=1.0
            x=torch.cat((x,x1,x2),dim=0)
        assert(x.shape[0]==3**9)
        shapev=x.view((3**9,2,3,3)).to(device)

        assert(boardH==7 and boardW==7)

        map_each_loc=[]
        for loc in range(25):
            locX=loc%5 + 1
            locY=loc//5 + 1
            locIdx=locX+locY*7
            assert(locIdx>=8 and locIdx<=40)
            locv=torch.zeros((1,49,3,3))
            locv[:,locIdx-8,0,0]=1
            locv[:,locIdx-7,0,1]=1
            locv[:,locIdx-6,0,2]=1
            locv[:,locIdx-1,1,0]=1
            locv[:,locIdx,1,1]=1
            locv[:,locIdx+1,1,2]=1
            locv[:,locIdx+6,2,0]=1
            locv[:,locIdx+7,2,1]=1
            locv[:,locIdx+8,2,2]=1

            locv=locv.to(device)
            x=torch.cat((shapev,torch.repeat_interleave(locv,dim=0,repeats=shapev.shape[0])),dim=1)

            with torch.no_grad():
                y = self.conv1(x)
                for block in self.mappingtrunk:
                    y=block(y)
                assert(y.shape[1]==self.midc)
                assert(y.shape[2]==1)
                assert(y.shape[3]==1)
                y=y.view((-1,self.midc))
                y=torch.einsum("nc,co->no",y,self.mappingFinalWeights[loc,:,:])
                assert(y.shape[1]==self.c)
                if (self.mapbound != 0):
                    y = self.mapbound * torch.tanh(y / self.mapbound)

            y=y.detach().cpu().numpy()
            map_each_loc.append(y)

        map=np.stack(map_each_loc)
        assert(map.shape[0]==25)
        assert(map.shape[1]==3**9)
        assert(map.shape[2]==self.c)
        return map

class CNNLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv=nn.Conv2d(in_c,
                      out_c,
                      3,
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      bias=False,
                      padding_mode='zeros')
        self.bn= nn.BatchNorm2d(out_c)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = torch.relu(y)
        return y

class ResnetLayer(nn.Module):
    def __init__(self, inout_c, mid_c):
        super().__init__()
        self.conv_net = nn.Sequential(
            CNNLayer(inout_c, mid_c),
            CNNLayer(mid_c, inout_c)
        )

    def forward(self, x):
        x = self.conv_net(x) + x
        return x

class Outputhead_v1(nn.Module):

    def __init__(self,out_c,head_mid_c):
        super().__init__()
        self.cnn=CNNLayer(out_c, head_mid_c)
        self.valueHeadLinear = nn.Linear(head_mid_c, 3)
        self.policyHeadLinear = nn.Conv2d(head_mid_c, 1, 1)

    def forward(self, h):
        x=self.cnn(h)

        # value head
        value = x.mean((2, 3))
        value = self.valueHeadLinear(value)

        # policy head
        policy = self.policyHeadLinear(x)
        policy = policy.squeeze(1)

        return value, policy



class Model_ResNet(nn.Module):

    def __init__(self,b,f):
        super().__init__()
        self.model_type = "res"
        self.model_param=(b,f)

        self.inputhead=CNNLayer(input_c, f)
        self.trunk=nn.ModuleList()
        for i in range(b):
            self.trunk.append(ResnetLayer(f,f))
        self.outputhead=Outputhead_v1(f,f)

    def forward(self, x):
        if(x.shape[1]!=input_c):#global feature is not none
            x=torch.cat((x,torch.zeros((1,input_c-2,boardH,boardW)).to(x.device)),dim=1)
        h=self.inputhead(x)

        for block in self.trunk:
            h=block(h)

        return self.outputhead(h)

ModelDic = {
    "res": Model_ResNet, #resnet对照组
    "vov1": Model_vov1,
}
