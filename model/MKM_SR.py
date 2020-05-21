import torch
from torch import  nn
import torch.nn.functional as F
import math
from torch.nn import Module,Parameter
from model.utils import trans_to_cuda,GNN



class MKM_SR(Module):
    def __init__(self,opt,n_entity,n_relation,n_item):
        super(MKM_SR, self).__init__()
        self.hidden_size = opt.hidden_size
        self.l2 = opt.l2
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.batch_size = opt.batch_size
        self.kg_loss_rate = trans_to_cuda(torch.Tensor([opt.kg_loss_rate]).float())

        self.entity_embedding = nn.Embedding(self.n_entity, self.hidden_size)
        self.relation_embedding = nn.Embedding(self.n_relation, self.hidden_size)
        self.norm_vector = nn.Embedding(self.n_relation, self.hidden_size)

        self.gnn_entity = GNN(self.hidden_size, step=opt.step)
        self.gru_relation = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)
        self.linear_one = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.linear_two = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.linear_three = nn.Linear(self.hidden_size * 2, 1, bias=True)
        self.linear_transform = nn.Linear(self.hidden_size * 4, self.hidden_size, bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size*2)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(self, seq_hiddens, masks, itemindexTensor):
        ht = seq_hiddens[
            torch.arange(masks.shape[0]).long(), torch.sum(masks, 1) - 1]  # the last one #batch_size*hidden_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size*1*hidden_size
        q2 = self.linear_two(seq_hiddens)  # batch_size*seq_length*hidden_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size * seq_len *1
        a = torch.sum(alpha * seq_hiddens * masks.view(masks.shape[0], -1, 1).float(), 1) # a.shape batch_size *hidden_size
        a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.entity_embedding.weight[itemindexTensor]  # n_items *latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A,relation_inputs):
        entity_hidden = self.entity_embedding(inputs)  # batch,L,hidden_size
        entity_hidden = self.gnn_entity(A, entity_hidden)  # batch,hidden_size
        relation_inputs = self.relation_embedding(relation_inputs)
        relation_output,relation_hidden = self.gru_relation(relation_inputs,None)
        return entity_hidden,relation_output