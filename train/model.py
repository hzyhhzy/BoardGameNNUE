
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




class Model_vo1(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "vo1"
        self.model_param = (c, mlpc, b, midc)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc

        #mapping
        self.conv1=nn.Conv2d(2,midc,3,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts


        #mlp
        self.prelu1=PRelu1(c,bias=True,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

    def forward(self, x):
        #mapping
        y = self.conv1(x)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.midc*5*5))
        y=self.conv2(y)

        #mlp
        y=self.prelu1(y)
        y=self.mlp1(y)
        y=torch.relu(y)
        y=self.mlp2(y)
        y=torch.relu(y)
        v=self.mlp3(y)

        return v, None

class Model_vo2(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256):  # 1d卷积通道，policy通道，胜负和通道
        super().__init__()
        self.model_type = "vo2"
        self.model_param = (c, mlpc, b, midc)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc

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
        self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts


        #mlp
        self.prelu1=PRelu1(c,bias=True,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

    def forward(self, x):
        #mapping
        y=torch.cat((x,torch.repeat_interleave(self.loc_embedding,dim=0,repeats=x.shape[0])),dim=1)
        y = self.conv1(y)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.midc*5*5))
        y=self.conv2(y)

        #mlp
        y=self.prelu1(y)
        y=self.mlp1(y)
        y=torch.relu(y)
        y=self.mlp2(y)
        y=torch.relu(y)
        v=self.mlp3(y)

        return v, None



class Model_vo3(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo3"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

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


class Model_vo4(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo4"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        #mapping
        self.conv1=nn.Conv2d(2,midc,3,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        self.conv2=nn.Conv2d(midc,c,1)
        self.prelu0=PRelu1(c*25,bias=True,bound=0.999)

        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

    def forward(self, x):
        #mapping
        y = self.conv1(x)
        for block in self.mappingtrunk:
            y=block(y)
        y=self.conv2(y)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.c*5*5))

        if (self.mapbound != 0):
            y = self.mapbound * torch.tanh(y / self.mapbound)

        y=self.prelu0(y)
        y=y.view((-1,self.c,5*5))
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
        return
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

class Model_vo5(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo5"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))

        self.conv2=nn.Conv2d(c,c,kernel_size=3,groups=c)
        self.prelu0=PRelu1(c*9,bias=False,bound=0.999)
        self.conv3=nn.Conv2d(c,c,kernel_size=3,groups=c)


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

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


        y=y.view((-1,self.c,5,5))
        y=self.conv2(y)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==3)
        assert(y.shape[3]==3)
        y=y.view((-1,self.c*9))
        y=self.prelu0(y)
        y=y.view((-1,self.c,3,3))
        y=self.conv3(y)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==1)
        assert(y.shape[3]==1)
        y=y.view((-1,self.c))

        #y=torch.mean(y,dim=2)

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


class Model_vo6(nn.Module):
    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo6"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound


        self.conv1=nn.Conv2d(2,midc,3,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(6,midc,c))-1))

        self.prelu0=PRelu1(c*25,bias=True,bound=0.999)


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.99)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

    def forward(self, x):
        #mapping
        y = self.conv1(x)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.midc,5*5))

        w=self.mappingFinalWeights
        mappingFinalWeightsUnSym=torch.stack((w[0],w[1],w[2],w[1],w[0],w[1],w[3],w[4],w[3],w[1],w[2],w[4],w[5],w[4],w[2],w[1],w[3],w[4],w[3],w[1],w[0],w[1],w[2],w[1],w[0]),dim=0).contiguous()
        y=torch.einsum("nch,hco->noh",y,mappingFinalWeightsUnSym)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==5*5)

        y=y.reshape((-1,self.c*5*5))

        if (self.mapbound != 0):
            y = self.mapbound * torch.tanh(y / self.mapbound)

        #here the mapping finished. we get 25*c

        y=self.prelu0(y)

        y=y.view((-1,self.c,5*5))

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


class Model_vo7(nn.Module):
    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo7"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        self.initPreconvSymMat() #self.preconvSymMat=tensor(7,7,9,5,5),self.preconvSymEmbedding=tensor(1,6,5,5)
        self.conv1=nn.Conv2d(2*9+6,midc,1,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(6,midc,c))-1))

        self.prelu0=PRelu1(c*25,bias=True,bound=0.999)


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.99)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)
    def initPreconvSymMat(self):
        m=torch.zeros((7,7,9,5,5))

        for y0 in range(-2,3):
            for x0 in range(-2,3):
                for dy0 in range(-1,2):
                    for dx0 in range(-1,2):
                        y=y0
                        x=x0
                        dy=dy0
                        dx=dx0
                        if(y<0):
                            y=-y
                            dy=-dy
                        if(x<0):
                            x=-x
                            dx=-dx
                        if(x<y):
                            tmp=x
                            x=y
                            y=tmp
                            tmp=dx
                            dx=dy
                            dy=tmp
                        m[y0+dy+3,x0+dx+3,dy0*3+dx0+4,y0+2,x0+2]=1


        b=torch.stack((
            torch.tensor((1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1)),
            torch.tensor((0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0)),
            torch.tensor((0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0)),
            torch.tensor((0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0)),
            torch.tensor((0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0)),
            torch.tensor((0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)),
        ),dim=0).reshape((1,6,5,5))

        self.preconvSymMat = nn.Parameter(m,requires_grad=False)
        self.preconvSymEmbedding = nn.Parameter(b,requires_grad=False)

    def forward(self, x):
        #mapping
        y=torch.einsum("nchw,hwlab->nclab",x,self.preconvSymMat) #把25个3x3块的特征（c*3*3）收集到25个点上,之后这些点就不再有联系了，preconvSymMat是为了处理对称
        assert(y.shape[2]==9)
        assert(y.shape[3]==5)
        assert(y.shape[4]==5)

        y=y.view(x.shape[0],x.shape[1]*9,5,5)
        y=torch.cat((y,torch.repeat_interleave(self.preconvSymEmbedding,dim=0,repeats=x.shape[0])),dim=1)

        y = self.conv1(y)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.midc,5*5))

        w=self.mappingFinalWeights
        mappingFinalWeightsUnSym=torch.stack((w[0],w[1],w[2],w[1],w[0],w[1],w[3],w[4],w[3],w[1],w[2],w[4],w[5],w[4],w[2],w[1],w[3],w[4],w[3],w[1],w[0],w[1],w[2],w[1],w[0]),dim=0).contiguous()
        y=torch.einsum("nch,hco->noh",y,mappingFinalWeightsUnSym)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==5*5)

        y=y.reshape((-1,self.c*5*5))

        if (self.mapbound != 0):
            y = self.mapbound * torch.tanh(y / self.mapbound)

        #here the mapping finished. we get a 25*c vector (maybe int8)
        #在这以上全都是映射表，看着花里胡哨，但不用管它怎么运行

        y=self.prelu0(y)

        y=y.view((-1,self.c,5*5))

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

class Model_vo8(nn.Module):
    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo8"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        self.initPreconvSymMat() #self.preconvSymMat=tensor(7,7,9,5,5),self.preconvSymEmbedding=tensor(1,6,5,5)
        self.conv1=nn.Conv2d(2*9+6,midc,1,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(6,midc,c))-1))



        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.99)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)
    def initPreconvSymMat(self):
        m=torch.zeros((7,7,9,5,5))

        for y0 in range(-2,3):
            for x0 in range(-2,3):
                for dy0 in range(-1,2):
                    for dx0 in range(-1,2):
                        y=y0
                        x=x0
                        dy=dy0
                        dx=dx0
                        if(y<0):
                            y=-y
                            dy=-dy
                        if(x<0):
                            x=-x
                            dx=-dx
                        if(x<y):
                            tmp=x
                            x=y
                            y=tmp
                            tmp=dx
                            dx=dy
                            dy=tmp
                        m[y0+dy+3,x0+dx+3,dy0*3+dx0+4,y0+2,x0+2]=1


        b=torch.stack((
            torch.tensor((1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1)),
            torch.tensor((0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0)),
            torch.tensor((0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0)),
            torch.tensor((0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0)),
            torch.tensor((0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0)),
            torch.tensor((0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)),
        ),dim=0).reshape((1,6,5,5))

        self.preconvSymMat = nn.Parameter(m,requires_grad=False)
        self.preconvSymEmbedding = nn.Parameter(b,requires_grad=False)

    def forward(self, x, debugPrint=False):
        #mapping
        y=torch.einsum("nchw,hwlab->nclab",x,self.preconvSymMat) #把25个3x3块的特征（c*3*3）收集到25个点上,之后这些点就不再有联系了，preconvSymMat是为了处理对称
        assert(y.shape[2]==9)
        assert(y.shape[3]==5)
        assert(y.shape[4]==5)

        y=y.view(x.shape[0],x.shape[1]*9,5,5)
        y=torch.cat((y,torch.repeat_interleave(self.preconvSymEmbedding,dim=0,repeats=x.shape[0])),dim=1)

        y = self.conv1(y)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.midc,5*5))

        w=self.mappingFinalWeights
        mappingFinalWeightsUnSym=torch.stack((w[0],w[1],w[2],w[1],w[0],w[1],w[3],w[4],w[3],w[1],w[2],w[4],w[5],w[4],w[2],w[1],w[3],w[4],w[3],w[1],w[0],w[1],w[2],w[1],w[0]),dim=0).contiguous()
        y=torch.einsum("nch,hco->noh",y,mappingFinalWeightsUnSym)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==5*5)


        if (self.mapbound != 0):
            y = self.mapbound * torch.tanh(y / self.mapbound)

        #here the mapping finished. we get a 25*c vector (maybe int8)
        #在这以上全都是映射表，看着花里胡哨，但不用管它怎么运行


        y=torch.mean(y,dim=2)

        #mlp
        y=self.prelu1(y)
        if(debugPrint):
            print((y*25*126/self.mapbound).detach().int())
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
        shapev=x.view((3**9,2*9,1,1)).to(device)

        assert(boardH==7 and boardW==7)

        map_each_loc=[]
        for loc in range(6):
            locv=torch.zeros((1,6,1,1))
            locv[:,loc,:,:]=1

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
        assert(map.shape[0]==6)
        assert(map.shape[1]==3**9)
        assert(map.shape[2]==self.c)
        return map


class Model_vo9t1(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo9t1"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        #mapping

        self.conv1=nn.Conv2d(2,midc,3,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        self.conv2=nn.Conv2d(midc,c,3,padding=0)


        #mlp
        self.prelu1=PRelu1(9*c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(9*c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

    def forward(self, x):
        #mapping
        y = self.conv1(x)
        for block in self.mappingtrunk:
            y=block(y)
        y=self.conv2(y)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==3)
        assert(y.shape[3]==3)
        y=y.view((-1,self.c*3*3))

        #mlp
        y=self.prelu1(y)
        y=self.mlp1(y)
        y=torch.relu(y)
        y=self.mlp2(y)
        y=torch.relu(y)
        v=self.mlp3(y)

        return v, None

class Model_vo10t1(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo10t1"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        #mapping

        self.conv1=nn.Conv2d(2,midc,3,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        self.conv2=nn.Conv2d(midc,c,5,padding=2)
        self.prelu0=PRelu1(25*c,bias=True,bound=0.999)


        #mlp
        self.prelu1=PRelu1(c,bias=True,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,3)

    def forward(self, x):
        #mapping
        y = self.conv1(x)
        for block in self.mappingtrunk:
            y=block(y)
        y=self.conv2(y)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==5)
        assert(y.shape[3]==5)
        y=y.view((-1,self.c*5*5))
        y=self.prelu0(y)

        y=y.view((-1,self.c,5*5))
        y=torch.mean(y,dim=2)

        #mlp
        y=self.prelu1(y)
        y=self.mlp1(y)
        y=torch.relu(y)
        y=self.mlp2(y)
        y=torch.relu(y)
        v=self.mlp3(y)

        return v, None


class Model_vo3t1(nn.Module):
    #no mlp, single value
    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo3t1"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.finalLinear=nn.Linear(c,1)

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
        v=self.finalLinear(y).view(-1)

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


class Model_vo3t2(nn.Module):
    #no mlp
    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo3t2"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.finalLinear=nn.Linear(c,3)

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
        v=self.finalLinear(y)

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

class Model_vo3t3(nn.Module):
    #simple sum
    def __init__(self, c=1, mlpc=114514, b=3, midc=128,mapbound=500):
        super().__init__()
        self.model_type = "vo3t3"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))



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


        return y, None

class Model_vo3t4(nn.Module):
    #simple sum, 4x4 kernel
    def __init__(self, c=1, mlpc=114514, b=3, midc=128,mapbound=500):
        super().__init__()
        self.model_type = "vo3t4"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc
        self.mapbound=mapbound

        #mapping
        self.loc_embedding=torch.zeros((1,boardH*boardW,boardH,boardW))
        for y in range(boardH):
            for x in range(boardW):
                self.loc_embedding[0,y*boardW+x,y,x]=1
        self.loc_embedding=nn.Parameter(self.loc_embedding,requires_grad=False)

        self.conv1=nn.Conv2d(2+boardH*boardW,midc,4,padding=0)
        self.mappingtrunk=nn.ModuleList()
        for i in range(b):
            self.mappingtrunk.append(Conv0dResBlock(in_c=midc,mid_c=midc))
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(16,midc,c))-1))



    def forward(self, x):
        #mapping
        y=torch.cat((x,torch.repeat_interleave(self.loc_embedding,dim=0,repeats=x.shape[0])),dim=1)
        y = self.conv1(y)
        for block in self.mappingtrunk:
            y=block(y)
        assert(y.shape[1]==self.midc)
        assert(y.shape[2]==4)
        assert(y.shape[3]==4)
        y=y.view((-1,self.midc,4*4))
        y=torch.einsum("nch,hco->noh",y,self.mappingFinalWeights)
        assert(y.shape[1]==self.c)
        assert(y.shape[2]==4*4)

        if (self.mapbound != 0):
            y = self.mapbound * torch.tanh(y / self.mapbound)

        y=torch.mean(y,dim=2)


        return y, None

class Model_vo3t5(nn.Module):
    #no mlp
    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=500):
        super().__init__()
        self.model_type = "vo3t5"
        self.model_param = (c, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        finalWeights=param_init_scale*(2*torch.rand(size=(1,midc,c))-1)
        self.mappingFinalWeights=nn.Parameter(torch.repeat_interleave(finalWeights,25,dim=0))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.finalLinear=nn.Linear(c,3)

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
        v=self.finalLinear(y)

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
def fake_quant(x: torch.Tensor, scale=128, zero_point=0, num_bits=8, signed=True):
    """Fake quantization while keep float gradient."""
    x_quant = (x.detach() * scale + zero_point).round().int()
    if num_bits is not None:
        if signed:
            qmin = -(2**(num_bits - 1))
            qmax = 2**(num_bits - 1) - 1
        else:
            qmin = 0
            qmax = 2**num_bits - 1
        x_quant = torch.clamp(x_quant, qmin, qmax)
    x_dequant = (x_quant - zero_point).float() / scale
    x = x - x.detach() + x_dequant  # stop gradient
    return x

class QuantLinear(nn.Linear):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 input_quant_scale=128,
                 input_quant_bits=8,
                 weight_quant_scale=128,
                 weight_quant_bits=8,
                 bias_quant_bits=16):
        super().__init__(in_dim, out_dim, bias=bias)
        self.input_quant_scale = input_quant_scale
        self.input_quant_bits = input_quant_bits
        self.weight_quant_scale = weight_quant_scale
        self.weight_quant_bits = weight_quant_bits
        self.bias_quant_bits = bias_quant_bits

    def forward(self, x):
        x = fake_quant(x, self.input_quant_scale, num_bits=self.input_quant_bits)
        w = fake_quant(self.weight, self.weight_quant_scale, num_bits=self.weight_quant_bits)
        if self.bias is not None:
            b = fake_quant(self.bias,
                           self.weight_quant_scale * self.input_quant_scale,
                           num_bits=self.bias_quant_bits)
            out = nn.functional.linear(x, w, b)
        else:
            out = nn.functional.linear(x, w)
        return out


class Model_vo3_int8(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256):
        super().__init__()
        self.model_type = "vo3_int8"
        self.model_param = (c, mlpc, b, midc)
        self.c=c
        self.mlpc=mlpc
        self.b=b
        self.midc=midc

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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=QuantLinear(c,mlpc)
        self.mlp2=QuantLinear(mlpc,mlpc)
        self.mlp3=QuantLinear(mlpc,3)

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

        y=torch.clamp(y, min=-1, max=127 / 128)
        y=fake_quant(y, scale=128)  # int8
        y=torch.sum(y,dim=2)        # sum, without division

        #mlp
        y=self.prelu1(y)
        y=fake_quant(y, scale=32768)  # int16
        y=torch.clamp(y, min=-1, max=127 / 128)
        y=self.mlp1(y)
        y=torch.clamp(y, min=0, max=127 / 128)
        y=self.mlp2(y)
        y=torch.clamp(y, min=0, max=127 / 128)
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


class Model_vo3a(nn.Module):

    def __init__(self, c=32, mlpc=16, b=5, midc=256,mapbound=100):
        super().__init__()
        self.model_type = "vov1"
        self.model_param = (c, mlpc, mlpc, b, midc,mapbound)
        self.c=c
        self.mlpc1=mlpc
        self.mlpc2=mlpc
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,mlpc)
        self.mlp3=nn.Linear(mlpc,1)

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

class Model_vo3b(nn.Module):
    #vo3a+fixed_layer2_mlp_size
    def __init__(self, c=128, mlpc=8, b=6, midc=512,mapbound=100):
        super().__init__()
        self.model_type = "vov1"
        self.model_param = (c, mlpc, 64, b, midc,mapbound)
        self.c=c
        self.mlpc1=mlpc
        self.mlpc2=64
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
        #self.conv2=nn.Linear(midc*25,c,bias=False) #can be separated to 25 parts
        param_init_scale=midc**-0.5
        self.mappingFinalWeights=nn.Parameter(param_init_scale*(2*torch.rand(size=(25,midc,c))-1))


        #mlp
        self.prelu1=PRelu1(c,bias=False,bound=0.999)
        self.mlp1=nn.Linear(c,mlpc)
        self.mlp2=nn.Linear(mlpc,64)
        self.mlp3=nn.Linear(64,1)

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
    "vo1": Model_vo1,
    "vo2": Model_vo2,
    "vo3": Model_vo3,
    "vo4": Model_vo4,
    "vo5": Model_vo5,
    "vo6": Model_vo6,
    "vo7": Model_vo7,
    "vo8": Model_vo8,
    "vo9t1": Model_vo9t1,
    "vo10t1": Model_vo10t1,
    "vo3t1": Model_vo3t1,
    "vo3t2": Model_vo3t2,
    "vo3t3": Model_vo3t3,
    "vo3_int8": Model_vo3_int8,
    "vo3t4": Model_vo3t4,
    "vo3t5": Model_vo3t5,
    "vo3a": Model_vo3a,
    "vo3b": Model_vo3b,
    "vov1": Model_vov1,
}
