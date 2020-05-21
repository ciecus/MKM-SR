import sys,os,pickle,argparse
import pandas as pd
import datetime
import sys, os, pickle, argparse
from random import shuffle
import numpy as np
import random
import math
import time
from tqdm import tqdm

def sigmoid(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))

def load_data_from_pickle(f_dir):
    tr_data = pickle.load(open(f_dir+'train.fpmc','rb'))
    tr_data = list(zip(tr_data[0],tr_data[1],tr_data[2]))
    te_data = pickle.load(open(f_dir+'test.fpmc','rb'))
    te_data = list(zip(te_data[0],te_data[1],te_data[2]))
    user_set = set(pickle.load(open(f_dir+'user_index.fpmc','rb')))
    item_set = set(pickle.load(open(f_dir+'item_index.fpmc','rb')))
    return tr_data,te_data,user_set,item_set

class FPMC():
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        self.user_set = set()
        self.item_set = set()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    @staticmethod
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def init_model(self, std=0.1):
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def compute_x(self, u, i, b_tm1):
        acc_val = 0.0
        for l in b_tm1:
            acc_val += np.dot(self.VIL[i], self.VLI[l])
        return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val / len(b_tm1)))

    def compute_x_batch(self, u, b_tm1):
        former = self.VUI_m_VIU[u]
        latter = np.mean(self.VIL_m_VLI[:, b_tm1], axis=1).T
        return (former + latter)

    def evaluation(self, data_list):
        topk = 20
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        print('self._VUI_m_VIU')
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        print('self.VIL_m_VLI')
        correct_count = 0
        rr_list = []
        for (u, i, b_tm1) in tqdm(data_list):
            scores = self.compute_x_batch(u, b_tm1)
            rank = len(np.where(scores > scores[i])[0]) + 1
            if rank<=topk:
                rr = 1.0 / rank
                correct_count += 1
            else:
                rr = 0
            rr_list.append(rr)

        try:
            acc = correct_count / len(rr_list)
            mrr = (sum(rr_list) / len(rr_list))
            return (acc, mrr)
        except:
            return (0.0, 0.0)

    def learn_epoch(self, tr_data, neg_batch_size):
        print('len(tr_data):',len(tr_data))
        for iter_idx in range(len(tr_data)):
            if iter_idx == 1:
                start_time = time.time()
            (u, i, b_tm1) = random.choice(tr_data)
            exclu_set = self.item_set - set([i])
            j_list = random.sample(exclu_set, neg_batch_size)
            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)
                VUI_update = self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                VIUi_update = self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                VIUj_update = self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])
                self.VUI[u] += VUI_update
                self.VIU[i] += VIUi_update
                self.VIU[j] += VIUj_update

                eta = np.mean(self.VLI[b_tm1], axis=0)
                VILi_update = self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                VILj_update = self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                VLI_update = self.learn_rate * (
                        (delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

                self.VIL[i] += VILi_update
                self.VIL[j] += VILj_update
                self.VLI[b_tm1] += VLI_update
            if iter_idx == 1:
                print('a index instance:',time.time()-start_time)



    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=15, neg_batch_size=10,
                       eval_per_epoch=False):
        print('learn sbpr_fpmc:',datetime.datetime.now())
        for epoch in range(n_epoch):
            print('start epoch:%d,%s'%(epoch,datetime.datetime.now()))
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)
            print('learn epoch end,',datetime.datetime.now())
            if eval_per_epoch == True:
                print('evaluate epoch:',datetime.datetime.now())
                #acc_in, mrr_in = self.evaluation(tr_data)
                if te_data != None:
                    acc_out, mrr_out = self.evaluation(te_data)
                    acc_out = acc_out * 100
                    mrr_out = mrr_out * 100
                    print(' test sample:%.8f\t%.8f' % (acc_out, mrr_out))
                else:
                    print('no test sample')
            else:
                print('epoch %d done' % epoch, datetime.datetime.now())

        if eval_per_epoch == False:
            #acc_in, mrr_in = self.evaluation(tr_data)
            if te_data != None:
                acc_out, mrr_out = self.evaluation(te_data)
                print(' test sample:%.4f\t%.4f' % (acc_out, mrr_out))
                #print('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
            else:
                #print('In sample:%.4f\t%.4f' % (acc_in, mrr_in))
                print('no test sample')

        if te_data != None:
            return (acc_out, mrr_out)
        else:
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='demo')
    parser.add_argument('-e', '--n_epoch', help='# of epoch', type=int, default=15)
    parser.add_argument('--n_neg', help='# of neg samples', type=int, default=10)
    parser.add_argument('-n', '--n_factor', help='dimension of factorization', type=int, default=32)
    parser.add_argument('-l', '--learn_rate', help='learning rate', type=float, default=0.01)
    parser.add_argument('-r', '--regular', help='regularization', type=float, default=0.001)
    parser.add_argument('--remove_new_items', action='store_true',
                        help='whether remove new items:if add argument,it will be true')
    args = parser.parse_args(['--remove_new_items'])
    f_dir = './data/%s/'%(args.input_dir)
    if args.remove_new_items:
        f_dir+='no_new_item/'
    else:
        f_dir+='with_new_item/'
    print('file dir is:',f_dir)
    tr_data, te_data, user_set, item_set = load_data_from_pickle(f_dir)
    print('load data have done:', datetime.datetime.now())
    fpmc = FPMC(n_user=max(user_set) + 1, n_item=max(item_set) + 1, n_factor=args.n_factor,
                learn_rate=args.learn_rate, regular=args.regular)
    fpmc.user_set = user_set
    fpmc.item_set = item_set
    fpmc.init_model()
    print('start to learn,',datetime.datetime.now())
    acc, mrr = fpmc.learnSBPR_FPMC(tr_data, te_data, n_epoch=args.n_epoch,
                                   neg_batch_size=args.n_neg, eval_per_epoch=True)

    print("Accuracy:%.2f MRR:%.2f" % (acc, mrr))


