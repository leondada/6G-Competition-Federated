import torch
from torch import nn
import torch.nn.functional as F


class block(nn.Module):
    def __init__(self,innn,hid,scale=1.5):
        super(block, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(innn,hid),
            nn.LayerNorm(hid),
                nn.ReLU(),
                nn.Linear(hid,int(hid*scale)),
            nn.LayerNorm(int(hid*scale)),
                nn.ReLU(),
                nn.Linear(int(hid*scale),innn),
            nn.LayerNorm(innn),
                )
        
    def forward(self,x):
        out = F.relu(x+self.mlp(x))
        return out

class Resblock(nn.Module):
    def __init__(self,innn,dim,hid):
        super(Resblock, self).__init__()
        
        self.mlpa = nn.Sequential(
                nn.Linear(innn,dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                )
        self.mlpb = block(dim,hid)
        self.mlpc = block(dim,hid)
        self.mlpd = block(dim,hid)
        self.mlpe = block(dim,hid)
        self.midres = nn.Sequential(
            nn.Linear(dim,dim//2),
            nn.ReLU(),
            nn.Linear(dim//2,dim)
        )
        self.mid = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,1)
        )
        
    def forward(self,x):
        x = self.mlpa(x)
        x = self.mlpb(x)
        x = self.mlpc(x)
        x = self.mlpd(x)
        x = self.mlpe(x)
        out = self.mid(F.relu(x+self.midres(x)))
        return out

class Dora(nn.Module):
    def __init__(self,test=False,innn = 10,dim=96,hid=96):
        super(Dora, self).__init__()
        self.test=test

        self.mlp0 = Resblock(innn,dim,hid)
        self.mlp1 = Resblock(innn,dim,hid)
        self.mlp2 = Resblock(innn,dim,hid)
        self.mlp3 = Resblock(innn,dim,hid)
 
        
    def forward(self, finalpos):
        
        o0 = self.mlp0(finalpos[:,0,:])
        o1 = self.mlp1(finalpos[:,1,:])
        o2 = self.mlp2(finalpos[:,2,:])
        o3 = self.mlp3(finalpos[:,3,:])
        
        pathloss = torch.cat([o0,o1,o2,o3],dim=1)
        return pathloss

class DoraNet(nn.Module):
    def __init__(self,test=False):
        super(DoraNet, self).__init__()
        self.test=test

        self.DoraNet1 = Dora()
        self.DoraNet2 = Dora()
 
        
    def forward(self, finalpos):
        
        o0 = self.DoraNet1(finalpos)
        o1 = self.DoraNet2(finalpos)
        return (o0+o1)/2


            
def main():    
    b = 10
    doraNet = DoraNet()
    pos=torch.zeros((b,2))
    pathloss=torch.zeros(b,4)

    p_pathloss = doraNet(pos)
    print(torch.mean(torch.abs(p_pathloss-pathloss)))

        
if __name__ == '__main__':
    main()