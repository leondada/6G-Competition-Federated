import os
import numpy as np
from torch.utils.data import Dataset
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
import random
random.seed(42)

class DoraSet(Dataset):
    # def __init__(self,data):
    #     self.data = data
    def __init__(self,dataset_path,set='train',clientId=1):
        self.set=set
        folder=dataset_path
        if set=='test':
            self.data = np.reshape(np.load(folder + 'test.npy'),(-1,6))
        else:
            self.data=np.reshape(np.load(folder+f'user_{clientId:02d}.npy'),(-1,6))
        
        self.center = np.array([[100,100.,50],[100,300.,50],[300,100.,50],[300,300.,50]])
        self.data[:,:2]+=35 
        pos = self.data[:,:2]
        finalpos = []
        for i in range(4):
            a0 = np.sqrt((pos[:,0]-self.center[i][0])**2+(pos[:,1]-self.center[i][1])**2)/400
            a1 = np.log10(a0)/np.log10(400)
            a2 = np.sqrt((pos[:,0]-self.center[i][0])**2+(pos[:,1]-self.center[i][1])**2+(0-self.center[i][2])**2)/(400*1.415)
            a3 = np.log10(a2)/np.log10(400*1.415)
            a4 = (pos[:,0]-self.center[i][0])/a0
            a5 = (pos[:,1]-self.center[i][1])/a0
            a6 = 100/a2
            a7 = a0/a2
            
            a8 = (self.center[i][0]-pos[:,0])/400
            a9 = (self.center[i][1]-pos[:,1])/400
            a10 = np.zeros(pos.shape[0])+self.center[i][0]/400
            a11 = np.zeros(pos.shape[0])+self.center[i][1]/400

            
            c = []
            c.append(a0)
            c.append(a1)
            
            c.append(a2)
            c.append(a3)
            
            c.append(a4)
            c.append(a5)
            c.append(a6)
            c.append(a7)
            
            c.append(a8)
            c.append(a9)
            c.append(a10)
            c.append(a11)
            # c.append(a12)
            # c.append(a13)
            # c.append(a14)

            c = [pa[:,None] for pa in c]
            c = np.concatenate(c,axis=1)[:,[0,1,2,3,4,5,6,7]]#[0，4，5]
            temp =np.concatenate([pos,c],axis=1)
            for j in range(2):
                temp[:,j] = temp[:,j]/400

            finalpos.append(temp)
        
        self.augdata = np.concatenate(finalpos, axis=1)

    def __getitem__(self, idx):
        

            
        return self.augdata[idx].reshape(4,10),self.data[idx,2:]

    def __len__(self):
        return len(self.augdata)
    

class DoraSetComb(Dataset):
    def __init__(self,datasets):
        self.dataLen=[]
        self.datasets=datasets
        for i in datasets:
            self.dataLen.append(len(i))

    def __getitem__(self, idx):
        for i in range(len(self.dataLen)):
            if idx<np.sum(self.dataLen[:i+1]):
                if i==0:
                    idx2=idx
                else:
                    idx2=idx-np.sum(self.dataLen[:i])
                break
        return self.datasets[i][idx2]

    def __len__(self):
        return np.sum(self.dataLen)

if __name__=='__main__':
    dataset=DoraSet("data/train/",set="train",clientId=1)
    pos,pathloss=dataset[0]
    print(f'pos:',pos)
    print(f'pathloss:',pathloss)

