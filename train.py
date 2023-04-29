import copy
import sys
from model import Dora,DoraNet
from util import *
from dataset import DoraSet, DoraSetComb
import os
from tqdm import tqdm
import math
import random
random.seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
sys.path.append("../..")

'''
相对于示例代码做了如下更新:
客户端相关:
1:修改了数据加载文件,除了坐标外加入了额外8个特征,并且数据增强与信号类型有关。
2:修改了模型结构,一方面使用了一种残差全连接网络,另一面方针对不同的信号使用了不同的子网络,并且使用layernorm。
3:batchsize由500改为256,localepoch改为10,学习率使用cos衰减.

服务器相关:
4:【line183】修改了聚合算法,一方面,客户端统计本地每个信号的非0数量,服务器据此进行加权平均,另一方面,使用FedAdam进行全局模型参数更新。
5:【line491】最终选用验证集(训练集的1%)上性能最好和次好的模型进行集成。

客户端和服务器通信相关:
6:【line433】加入了warmup策略,即前40个轮次只一个客户端进行训练(0号客户端),不进行多客户端模型聚合。
7:【line478】每轮先选5个区域,再在每个区域中分别随机选1个客户端。并且warmup客户端所在区域每次都会选中。
8:【line340】在中间400个轮次中,在偶数轮时,仍使用上一轮次的客户端,并且客户端不使用服务器模型,而是使用本地模型继续训练,训练后上传模型进行聚合。\
    在奇数轮中,重新选择客户端,并且客户端接收服务器模型进行训练。
9:通信量压缩:
【line107、244】上行与下行策略一致:采用STC压缩方法,上传单个值,以及梯度的方向掩码。并对掩码进行压缩传输。
上下行涉及的对掩码的编码策略主要是:1.将位置编码成01向量后,存储为整数(每32位存为一个整数); 2.当目标值数量过少,直接传序号。
'''
epochs = 500  # total epochs
local_epochs = 10 # local epochs of each user at an iteration
saveLossInterval = 1  # intervals to save loss
saveModelInterval = 10  # intervals to save model
batchSize = 256  # batchsize for training and evaluation
num_users = 90   # total users
num_activate_users = 5
cudaIdx = "cuda:0"  # GPU card index
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
num_workers = 0  # workers for dataloader
evaluation = False  # evaluation only if True

def lrcos(step=0,lr=0.01,lr_min=0.0001,T_max=500):
    return 0.5*(1 + math.cos(math.pi * (step - 1) / T_max)) *(lr - lr_min) + lr_min


def STCcompress(tensor):
    '''参数压缩函数:
    来源于文章Sattler, Felix, et al. "Robust and communication-efficient federated learning from non-iid data.".
    只保留均值(一个标量),和符号掩码。
    '''
    k = np.ceil(tensor.numel()*0.99).astype(int)
    top_k_element, top_k_index = torch.kthvalue(-tensor.abs().flatten(), k)
    tensor_masked = (tensor.abs() > -top_k_element) * tensor
    magnitude = (1/k) * tensor_masked.abs().sum()
    return magnitude,tensor_masked.sign()

class Link(object):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.size = np.zeros((1,), dtype=np.float64)
    
    def pass_link(self, pay_load):
        '''通信数据的格式为：
        (value,(sign1,mask1),(sign2,mask2))
        其中value、sign1和sign2为单个标量值,mask为(0,list)
        '''
        for k, v in pay_load.items():
            if type(v)==tuple: 
                self.size = self.size +1+len(v[1][1][1])+len(v[2][1][1])+2
            else: 
                if len(v)==0: #空值,暂时按 1 计算
                    self.size += 1
                else: #未经压缩的参数
                    self.size = self.size + np.sum(v.numel())
        return pay_load


class FedAvgServer: # used as a center
    def __init__(self, global_parameters, down_link):
        self.global_parameters = global_parameters
        self.down_link = down_link
        # FedAdam的超参:
        self.eta = 1
        self.tau = 0.1
        self.m_t = None
        self.v_t = None
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.last_state = [None for i in range(90)]
        self.aera_grad = [None for i in range(90)]
        self.aera_avg = [None for i in range(9)]
        self.global_bias = {'mlp0.mlpa.0.bias': torch.ones([1])}
        # lastglobalmodel为上一轮的模型参数,用于下行通信时的参数压缩
        self.lastglobalmodel = None

        
    def download(self, user_idx):
        local_parameters = []
        if self.lastglobalmodel==None: #第一轮:所有参数全发
            for i in range(len(user_idx)):
                local_parameters.append(self.down_link.pass_link(copy.deepcopy(self.global_parameters)))
        else:# 计算当前全局模型 与 服务器保留的上一轮全局模型 相比的变化量，并继续压缩
            tempmodel = copy.deepcopy(self.lastglobalmodel)
            # 压缩策略与上传时一致。
            for j, (k, v) in enumerate(self.lastglobalmodel.items()):
                temp = self.global_parameters[k]-v
                if temp.numel()==1:
                    tempmodel[k] = temp
                else:
                    tempmodel[k] = [0,0,0]
                    value,mask = STCcompress(temp) # 原始掩码mask是一个只包含0,1、-1的矩阵
                    # 判断传输-1,0,1中的哪两个比较划算:
                    l1,l2,l3 = len(torch.where(mask.flatten()==-1)[0]),len(torch.where(mask.flatten()==0)[0]),len(torch.where(mask.flatten()==1)[0])
                    maxone = np.argsort(np.array([l1,l2,l3]))[-1]
                    a,b = np.array([[0,1],[-1,1],[-1,0]])[maxone]
                    
                    tempmodel[k][0]=value
                    for kk,now in enumerate([a,b]):
                        # 判断用32位整数压缩mask划算还是直接传序号划算
                        if len(torch.where(mask.flatten()==now)[0])*1./mask.numel()<(1/33):
                            # 直接传序号
                            masknow = (0,torch.where(mask.flatten()==now)[0].cpu())
                        else:
                            #mask编码成01向量后,存储为整数(每32位存为一个整数)
                            masknow = np.zeros(mask.flatten().shape,dtype=int)
                            masknow[torch.where(mask.flatten()==now)[0].cpu()] = 1
                            padded_length = ((masknow.shape[0] - 1) // 32 + 1) * 32
                            masknow = np.concatenate([masknow, np.zeros([padded_length - masknow.shape[0]], dtype=int)])
                            masknow = (1,np.packbits(masknow,axis=None).view(np.uint32))  
                        tempmodel[k][kk+1] = (now,masknow)
                    tempmodel[k] = tuple(tempmodel[k]) 
            for i in range(len(user_idx)):
                self.down_link.pass_link(tempmodel)
                local_parameters.append(copy.deepcopy(tempmodel))
        return local_parameters


    def download_bias(self, user_idx):
        local_parameters = []
        for i in range(len(user_idx)):
            local_parameters.append(self.down_link.pass_link(copy.deepcopy(self.global_bias)))
        return local_parameters

    def upload_bias(self, local_parameters,size_num): 
        weight2 = size_num/torch.sum(size_num,0)
        for i, (k, v) in enumerate(local_parameters[0].items()):
            tmp_v = torch.zeros_like(v)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][k]*weight2[j][int(k[3])]
            self.global_bias[k] = tmp_v
        
    def upload(self, local_parameters,user_idx,size_num):         
        # 更新上一轮全局模型参数
        self.lastglobalmodel = copy.deepcopy(self.global_parameters)
        # 服务器对压缩后的客户端梯度进行重建
        for j in range(len(local_parameters)):
            for i, (k, v) in enumerate(local_parameters[j].items()):
                if type(v)==tuple:
                    if len(v)!=3: print('error')
                    a = self.global_parameters[k].numel()
                    temp = self.global_parameters[k]*0+v[0]
                    base = 0-(v[1][0]+v[2][0])
                    if base not in [0,-1,1]:
                        print('error:' ,base)
                    target= base*torch.ones([a])
                    for m in v[1:]:
                        if m[1][0]==0:
                            target[m[1][1]]=m[0]
                        else:
                            mask = torch.tensor(np.unpackbits(m[1][1].view(np.uint8),axis=None)[:a])
                            target[torch.where(mask>0)]=m[0]
                    local_parameters[j][k] = temp*(target.view(self.global_parameters[k].shape))
                        
                else:
                    local_parameters[j][k] = local_parameters[j][k].cpu()

        # 进行梯度的加权平均
        new_global_parameters = copy.deepcopy(self.global_parameters)
        weight2 = size_num/torch.sum(size_num,0)
        for i, (k, v) in enumerate(new_global_parameters.items()):
            tmp_v = torch.zeros_like(v)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][k]*weight2[j][int(k[3])]
            new_global_parameters[k] = tmp_v
            
        # 参数更新策略:FedAdam 
        delta_t= [
            -1*new_global_parameters[x]   for x, y in zip(new_global_parameters, self.global_parameters)
        ]
        # m_t
        if not self.m_t:
            self.m_t = [torch.zeros_like(x) for x in delta_t]
        self.m_t = [
            self.beta_1*x + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]
        # v_t
        if not self.v_t:
            self.v_t = [torch.zeros_like(x)+self.tau*self.tau for x in delta_t]#+self.tau*self.tau
        self.v_t = [
            # x - (1.0 - self.beta_2) * (y*y) * torch.sign(x - (y*y))
            self.beta_2 * x + (1 - self.beta_2)*(y*y)
            for x, y in zip(self.v_t, delta_t)
        ]
        tempp = [self.global_parameters[i] for i in self.global_parameters]   
        new_weights = [
            x - self.eta * y / (torch.sqrt(z) + self.tau)
            for x, y, z in zip(tempp, self.m_t, self.v_t)
        ] 
        for k,v in zip(self.global_parameters,new_weights):
            self.global_parameters[k] = v


class Client: # as a user
    def __init__(self, dataset):
        self.dataset = dataset
            
    def train(self, model, learningRate, global_model,epoch,size_num): # training locally

        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)#weight_decay=0.001

        for local_epoch in range(1, local_epochs + 1):

            for i, (pos, pathloss) in enumerate(self.dataset):
                pos = pos.float().to(device)
                if local_epoch==1:
                    size_num+=torch.sum(pathloss!=0,0)
                pathloss = pathloss.float().to(device)
                optimizer.zero_grad()
                p_pathloss = model(pos)
                loss =  torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0]))
                # loss += 0.1*(regloss(fea[0])+regloss(fea[1])+regloss(fea[2])+regloss(fea[3]))
                loss.backward()
                optimizer.step()
        '''
        对梯度进行压缩:只上传梯度变化的大致方向,value只保留单个值,但是所有位置的梯度的方向都保留。
        '''  
        state = {}
        if epoch%2==0 or epoch>440:
            for i, (k, v) in enumerate(global_model.state_dict().items()):
                temp = model.state_dict()[k]-v.cuda()
                if temp.numel()==1:
                    state[k] = temp
                else:
                    state[k] = [0,0,0]
                    value,mask = STCcompress(temp) # 原始掩码mask是一个只包含0,1、-1的矩阵
                    # 判断传输-1,0,1中的哪两个比较划算:
                    l1,l2,l3 = len(torch.where(mask.flatten()<0)[0]),len(torch.where(mask.flatten()==0)[0]),len(torch.where(mask.flatten()>0)[0])
                    maxone = np.argsort(np.array([l1,l2,l3]))[-1]
                    a,b = np.array([[0,1],[-1,1],[-1,0]])[maxone]
                    
                    state[k][0]=value.cpu()
                    for kk,now in enumerate([a,b]):
                        # 判断用32位整数压缩mask划算还是直接传序号划算
                        if len(torch.where(mask.flatten()==now)[0])*1./mask.numel()<(1/33):
                            masknow = (0,torch.where(mask.flatten()==now)[0].cpu())
                        else:
                            masknow = np.zeros(mask.flatten().shape,dtype=int)
                            masknow[torch.where(mask.flatten()==now)[0].cpu()] = 1
                            padded_length = ((masknow.shape[0] - 1) // 32 + 1) * 32
                            masknow = np.concatenate([masknow, np.zeros([padded_length - masknow.shape[0]], dtype=int)])
                            masknow = (1,np.packbits(masknow,axis=None).view(np.uint32))  
                        state[k][kk+1] = (now,masknow)
                    state[k] = tuple(state[k])
        else:
            # 440轮之前的奇数轮,上传任意参数
            for i, (k, v) in enumerate(model.state_dict().items()):
                if k=='mlp1.mid.0.bias':
                    state[k] = torch.ones([1])
        
        return state
                
class Client_self: # as a user
    def __init__(self, data_loader, user_idx,model):
        self.data_loader = data_loader
        self.user_idx = user_idx
        self.model = model
        
    def train(self, learningRate): # training locally
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate,)#weight_decay=0.001
        for _ in range(1, local_epochs+1):
            for i, (pos, pathloss) in enumerate(self.data_loader):
                pos = pos.float().to(device)
                pathloss = pathloss.float().to(device)
                optimizer.zero_grad()
                p_pathloss = self.model(pos)
                loss =  torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0]))
                # loss += 0.1*(regloss(fea[0])+regloss(fea[1])+regloss(fea[2])+regloss(fea[3]))
                loss.backward()
                optimizer.step()
                
def activateClient(train_dataloaders, user_idx, server, bias=False):
    if bias:
        local_parameters= server.download_bias(user_idx)
    else:
        local_parameters = server.download(user_idx)
    clients = []
    for i in range(len(user_idx)):
        clients.append(Client(train_dataloaders[user_idx[i]]))
    return clients, local_parameters


# 客户端收到压缩参数后进行重建
def recon(new_parameters,last_global, haslast):
    '''
    new_parameters: 客户端收到的来自服务器的参数梯度
    last_global: 客户端保留的上一轮全局模型参数
    haslast: 判断是不是第一轮,刚开始的时候,服务器传过来的是模型参数,而不是梯度
    return 重建后的客户端的参数
    '''
    if haslast==None:
        return new_parameters 
    for j in range(len(new_parameters)):
        for _, (k, v) in enumerate(new_parameters[j].items()):
            if type(v)==tuple:
                if len(v)!=3: print('error')#判断压缩类型
                a = last_global[k].numel()
                temp = last_global[k]*0+v[0]
                base = 0-(v[1][0]+v[2][0])
                target= base*torch.ones([a])
                for m in v[1:]:
                    if m[1][0]==0:
                        target[m[1][1]]=m[0]
                    else:
                        mask = torch.tensor(np.unpackbits(m[1][1].view(np.uint8),axis=None)[:a])
                        target[torch.where(mask>0)]=m[0]
                new_parameters[j][k] = temp*(target.view(last_global[k].shape))+last_global[k]
            else:
                if len(v)==0:
                    new_parameters[j][k] = last_global[k]
                else:
                    new_parameters[j][k] = new_parameters[j][k]+last_global[k]
    return new_parameters             
    
def train20(train_dataloaders, user_idx, server, global_model,last_gloabl_model, up_link, learningRate,epoch,clientmodels,last_useridx,final=450):
    # size_num 记录客户端数据量
    size_num = torch.zeros([5,4])
    if epoch>final:
        clients, local_parameters = activateClient(train_dataloaders, user_idx, server)
        local_parameters = recon(local_parameters,last_gloabl_model.state_dict(),server.lastglobalmodel)#server.lastglobalmodel一个辅助标记,图方便,没有新建变量存
    elif epoch%2==1:
        clients, local_parameters = activateClient(train_dataloaders, user_idx, server)
        local_parameters = recon(local_parameters,last_gloabl_model.state_dict(),server.lastglobalmodel)
    else: # 第偶数轮,客户端不使用服务器模型,直接用自己的模型继续训练.
        user_idx = [i for i in last_useridx]
        clients, bias = activateClient(train_dataloaders, last_useridx, server,bias=True)
        local_parameters = [clientmodels[i].state_dict() for i in last_useridx]
    
    states = []
    for i in range(len(user_idx)):
        model = Dora().to(device)
        model.load_state_dict(local_parameters[i])
        model.train()
        state = clients[i].train(model, learningRate, global_model,epoch,size_num[i])#user_idx[i]//10
        states.append(state)
        
        up_link.pass_link(state)
        clientmodels[user_idx[i]].load_state_dict(model.to('cpu').state_dict())
        
    if epoch%2==0 or epoch>final:  # 第偶数轮以及最后60轮上传压缩后的参数
        server.upload(states,user_idx,size_num)
    else: # 其他轮次上传任意参数
        server.upload_bias(states,size_num)
    # 更新客户端保留的上一轮全局模型参数
    last_gloabl_model.load_state_dict(global_model.state_dict())
    
    global_model.load_state_dict(server.global_parameters)

def valid(data_loader, model, epoch):
    with torch.no_grad():
        model.eval()
        losses = Recoder()
        scores = Recoder()
        for i, (pos, pathloss) in enumerate(data_loader):
            pos = pos.float().to(device)
            pathloss = pathloss.float().to(device)
            p_pathloss = model(pos)
            loss = torch.mean(torch.abs(p_pathloss[pathloss != 0] - pathloss[pathloss != 0])) ## unit in dB
            tmp1 = (torch.abs(10 ** (0.1 * p_pathloss[pathloss != 0]) - 10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
            tmp2 = (torch.abs(10 ** (0.1 * pathloss[pathloss != 0])) ** 2)
            score = torch.sum(tmp1) / torch.sum(tmp2)
            if score>1:
                score=torch.tensor([1])
            losses.update(loss.item(), len(pos))
            scores.update(score.item(), len(pos))
        print(f"Global Epoch: {epoch}----loss:{losses.avg():.4f}")
    model.train()
    return -10 * np.log10(scores.avg())      
            
def train_main(train_dataset_path):
    seedn = 19960107
    seed_everything(seedn)
    train_dataloaders = []
    valloaders = []
    train_datasets = []
    valid_datasets = []
    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')
    if not os.path.exists(f'results/'):
        os.makedirs(f'results/')
    for i in tqdm(range(1, num_users + 1)):
        all_dataset = DoraSet(train_dataset_path, set='train', clientId=i)
        train_size = int(0.99 * len(all_dataset))
        valid_size = len(all_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(all_dataset, [train_size, valid_size])
        train_datasets.append(train_dataset)##
        valid_datasets.append(valid_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batchSize, shuffle=True, num_workers=num_workers)##
        valid_data = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, num_workers=num_workers)
        train_dataloaders.append(train_loader)
        valloaders.append(valid_data)
 
    valid_data_comb = DoraSetComb(valid_datasets)
    valid_loader = torch.utils.data.DataLoader(valid_data_comb, 1, shuffle=False, num_workers=num_workers)
    model = Dora()
    global_parameters = model.state_dict()
    up_link = Link("uplink")
    down_link = Link("downlink")
    server = FedAvgServer(global_parameters, down_link)

    clientmodels = [copy.deepcopy(model) for _ in range(90)]
    pathloss_scores = []
    ul_commCost_scores = []
    dl_commCost_scores = []
    max_pathloss_score = 0

    # 前40轮进行warmup,只对一个客户端进行训练,不进行多模型聚合
    warmup = 40
    model = model.to(device)
    model.train()
    warmupclient = Client_self(train_dataloaders[0],0,model)
    testloader = torch.utils.data.DataLoader(DoraSetComb(valid_datasets[:1]), 1, shuffle=False, num_workers=num_workers)
    maxscore = 0
    for epoch in tqdm(range(1, warmup+1)):  ## start training
        # 模型下发,为了符合规则,下行通信开销计算了5次
        if epoch==1: #初始轮下发完整模型参数
            for _ in range(5):
                down_link.pass_link(global_parameters)
        else: #其他39轮,下发任意参数即可
            for _ in range(5):
                down_link.pass_link({'zeropara':torch.ones([1])})
        lrn = 0.0003#lrcos(epoch-1,lr = 0.00035, lr_min = 0.0003, T_max = warmup+1)
        warmupclient.train(lrn)
        test_model = copy.deepcopy(warmupclient.model).to(device)
        pathloss_score = valid(testloader, test_model, epoch)
        if pathloss_score>maxscore:
            maxscore=pathloss_score
        print(pathloss_score,maxscore)
        
        if epoch==warmup+1: #最后一轮需要上传完整参数
            # 模型上传,只有固定的一个上传参数,其他4个客户端本阶段上传任意参数(空参数)
            up_link.pass_link(global_parameters)
        else:  #其他轮次上传任意参数
            for _ in range(5):
                up_link.pass_link({'zeropara':torch.ones([1])})          
    model.load_state_dict(warmupclient.model.state_dict())
    server.global_parameters = model.to('cpu').state_dict()
    
    last_useridx = None
    modelem = DoraNet().to(device)
    fstmodel = Dora().to(device)
    secmodel = Dora().to(device)
    last_gloabl_model = copy.deepcopy(model.cpu())
    fst,sec = 0,-1
    for epoch in tqdm(range(warmup+1, epochs + 1)):  ## start training
        
        if epoch<=440:
            lrn = lrcos((epoch-warmup-1)//2,lr = 0.00035, lr_min = 0.00003, T_max = 260)
        else:
            lrn = lrcos(epoch-440+200,lr = 0.00035, lr_min = 0.00003, T_max = 260)

        # warmup的域始终参与
        g_idx = [0]+np.random.choice(a=list(range(1,9)), size=4, replace=False, p=None).tolist()
        user_idx = [np.random.choice(a=list(range(lll*10,lll*10+10)), size=1, replace=False, p=None).tolist()[0] for lll in g_idx]

        train20(train_dataloaders, user_idx, server, model, last_gloabl_model,up_link, lrn,epoch,clientmodels,last_useridx,440)     
        
        last_useridx = user_idx
        test_model = copy.deepcopy(model).to(device)

        pathloss_score = valid(valid_loader, test_model, epoch)
        pathloss_scores.append(pathloss_score)
        ul_commCost_scores.append(up_link.size)
        dl_commCost_scores.append(down_link.size)

        if pathloss_score>max_pathloss_score:
            max_pathloss_score = pathloss_score    
        print(pathloss_score,max_pathloss_score,up_link.size[0]/1e6,down_link.size[0]/1e6)
        # 服务器端保留最好和第二好的模型,最后集成进modelem中,modelem是最终使用的模型。
        if pathloss_score>fst:
            sec = fst
            fst = pathloss_score
            secmodel.load_state_dict(fstmodel.state_dict())
            fstmodel.load_state_dict(test_model.state_dict())
            modelem.DoraNet1.load_state_dict(fstmodel.state_dict())
            modelem.DoraNet2.load_state_dict(secmodel.state_dict())
        elif pathloss_score>sec and pathloss_score<fst:
            sec = pathloss_score
            secmodel.load_state_dict(test_model.state_dict())
            modelem.DoraNet2.load_state_dict(secmodel.state_dict())
        
        if epoch==epochs: 
            # 训练结束后,保存服务器的集成模型(最好和次好的模型的集成)'final_model19960107model.pth'
            print('em:',valid(valid_loader, modelem, epoch))
            print(up_link.size[0]/1e6,down_link.size[0]/1e6)
            checkPoint(epoch, epochs, modelem, pathloss_scores, ul_commCost_scores, dl_commCost_scores, saveModelInterval, saveLossInterval,'')

if __name__ == '__main__':
    train_main("data/train/")











