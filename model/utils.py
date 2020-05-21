import numpy as np
import torch
import datetime
import math
from tqdm import tqdm
import math
from torch.nn import Module,Parameter
from torch import  nn
import torch.nn.functional as F



class MKM_DATA():
    def __init__(self,data):
        self.data_paddings,self.data_operation_paddings, self.data_masks, self.data_targets = np.array(data[0]),np.array(data[1]), np.array(data[2]), np.array(data[3])

class GNN(Module):
    def __init__(self,hidden_size,step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size*2
        self.gate_size = 3*hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size,self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size,self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size,self.hidden_size,bias=True)

    def GNN_cell(self,A,hidden):
        input_in = torch.matmul(A[:,:,:A.shape[1]],self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:,:,A.shape[1]:2*A.shape[1]],self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in,input_out],2)
        g_i = F.linear(inputs,self.w_ih,self.b_ih) # batch_size * xx * gate_size
        g_h = F.linear(hidden,self.w_hh,self.b_hh)
        i_r,i_i,i_n = g_i.chunk(3,2) # tensors,chunks,dim
        h_r,h_i,h_n = g_h.chunk(3,2)
        resetgate = torch.sigmoid(i_r+h_r)
        inputgate = torch.sigmoid(i_i+h_i)
        newgate = torch.tanh(i_n + resetgate*h_n)
        hy = newgate + inputgate*(hidden-newgate)
        return hy

    def forward(self,A,hidden):
        for i in range(self.step):
            hidden = self.GNN_cell(A,hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def generate_batch_slices(len_data,shuffle=True,batch_size=128): #padding,masks,targets
    n_batch = math.ceil(len_data / batch_size)
    shuffle_args = np.arange(n_batch*batch_size)
    if shuffle:
        np.random.shuffle(shuffle_args)
    slices = np.split(shuffle_args,n_batch)
    slices = [i[i<len_data] for i in slices]
    return slices
def get_slice(slice_index,data_paddings,data_masks,data_targets):
    inputs,masks,targets = data_paddings[slice_index],data_masks[slice_index],data_targets[slice_index]
    items,n_node,A,alias_input = [],[],[],[]
    for u_input in inputs:
        n_node.append(len(np.unique(u_input))) #the length of unique items
    max_n_node = np.max(n_node) #the longest unique item length
    for u_input,u_mask in zip(inputs,masks):
        node = np.unique(u_input) #the unique items of inputs
        items.append(node.tolist()+(max_n_node-len(node))*[0]) #items list
        u_A = np.zeros((max_n_node,max_n_node))
        for i in range(len(u_input)-1):
            if u_input[i+1] == 0:
                break
            u = np.where(node == u_input[i])[0][0] #np.where return a tuple,so need use [0][0] to show the value
            v = np.where(node == u_input[i+1])[0][0]
            u_A[u][v] +=1
        u_sum_in = np.sum(u_A,0) # in degree
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A,u_sum_in)
        u_sum_out = np.sum(u_A,1) #out degree
        u_sum_out[np.where(u_sum_out ==0)] = 1
        u_A_out = np.divide(u_A.T,u_sum_out)
        u_A = np.concatenate([u_A_in,u_A_out]).T
        A.append(u_A)
        alias_input.append([np.where(node == i)[0][0] for i in u_input] )
    return alias_input,A,items,masks,targets

def get_mkm_slice(slice_index,data_paddings,data_operation_paddings,data_masks,data_targets):
    alias_input, A, items, masks, targets = get_slice(slice_index,data_paddings,data_masks,data_targets)
    operation_inputs = data_operation_paddings[slice_index]
    return alias_input,A,items,operation_inputs,masks,targets


def forward_mkm_model(model,slice_index,data,itemindexTensor):
    alias_inputs,A,items,operation_inputs,masks,targets = get_mkm_slice(slice_index, data.data_paddings,data.data_operation_paddings, data.data_masks, data.data_targets)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    operation_inputs = trans_to_cuda(torch.Tensor(operation_inputs).long())
    masks = trans_to_cuda(torch.Tensor(masks).long())

    entity_hidden,relation_hidden = model.forward(items, A,operation_inputs)

    get = lambda i: entity_hidden[i][alias_inputs[i]]
    seq_hiddens = torch.stack(
        [get(i) for i in torch.arange(len(alias_inputs)).long()])  # batch_size*L-length*hidden_size # todo
    seq_hiddens = torch.cat([seq_hiddens,relation_hidden],dim=2)

    state = model.predict(seq_hiddens, masks, itemindexTensor)
    return targets, state, masks


def train_predict_mkm(model,train_data,test_data,item_ids,itemid2index):
    itemindexTensor = torch.Tensor(item_ids).long()
    total_loss = 0.0
    slices = generate_batch_slices(len(train_data.data_paddings), shuffle=True, batch_size=model.batch_size)
    index = 0
    model.train()
    for slice_index, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, masks = forward_mkm_model(model, slice_index, train_data, itemindexTensor)
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        index += 1
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % 100 == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()),datetime.datetime.now())
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = generate_batch_slices(len(test_data.data_paddings), shuffle=False, batch_size=model.batch_size)
    for slice_index in slices:
        targets, scores, masks = forward_mkm_model(model, slice_index, test_data, itemindexTensor)
        sub_scores = scores.topk(20)[1]  # tensor has the top_k functions
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = [itemid2index[tar] for tar in targets]
        for score, target, mask in zip(sub_scores, targets, masks):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr