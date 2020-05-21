from model.MKM_SR import MKM_SR
from model.utils import trans_to_cuda,MKM_DATA,train_predict_mkm
import pandas as pd
import time
import pickle
from datetime import datetime
import argparse


def get_n_entity_relation_item(dataset,opt):
    entity2id = pd.read_csv(dataset+'entity2id',sep='\t')
    item2id =entity2id[entity2id['type']=='item']
    if opt.mode =='MKM_SR':
        relation2id = pd.read_csv(dataset+'relation_operation2id',sep='\t')
    n_entity = len(entity2id)+1
    n_relation = len(relation2id)+1
    n_item = len(item2id)+1
    return n_entity,n_relation,n_item



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='demo', help='dataset name:jdata2018/kkbox/demo')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
    parser.add_argument('--kg_loss_rate',type=float,default=0.001,help=' the rate of kg_loss')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--l2', type=float, default=1e-3, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    parser.add_argument('--remove_new_items',action='store_true',help='whether remove new items:if add argument,it will be true') #if add argument,it will be true,no new item
    parser.add_argument('--mode',default='MKM_SR',help='which mode:MKM_SR')
    opt = parser.parse_args(['--remove_new_items'])
    print('the mode is :',opt.mode)
    print("the dataset is :",opt.dataset)
    print('remove new item',opt.remove_new_items)


    if opt.dataset == 'kkbox':
        dataset = './data/kkbox/'
    elif opt.dataset == 'jdata2018':
        dataset = './data/jdata2018/'
    elif opt.dataset == 'demo':
        dataset = './data/demo/'

    if opt.remove_new_items:
        dataset += 'no_new_item/'
    else:
        dataset += 'with_new_item/'

    if opt.mode == 'MKM_SR':
        print('the mode is :', opt.mode)
        data_path = dataset + 'MKM_SR/'
        kg2id = pd.read_csv(dataset + 'kg2id', sep='\t')
        item_ids = kg2id['head'].unique()
        print('len(kg items):', len(item_ids))
        itemid2index = {}
        for index, item_id in enumerate(item_ids):
            itemid2index[item_id] = index
        train_processed = MKM_DATA(pickle.load(open(data_path + 'train_processed.pkl', 'rb')))
        test_processed = MKM_DATA(pickle.load(open(data_path + 'test_processed.pkl', 'rb')))
        n_entity, n_relation, n_item = get_n_entity_relation_item(dataset,opt)
        print('n_entity:{}, n_relation:{}, n_item:{}'.format(n_entity, n_relation, n_item))
        model = trans_to_cuda(MKM_SR(opt, n_entity, n_relation, n_item))
        start = time.time()
        print('start time: ', datetime.now())
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0
        for epoch in range(opt.epoch):
            epoch_start_time = time.time()
            print('-------------------------------------------------------')
            print('epoch: ' + str(epoch))
            hit, mrr = train_predict_mkm(model, train_processed, test_processed, item_ids, itemid2index)
            print('the epoch \tRecall@20:\t%.4f\tMMR@20:\t%.4f\t' % (hit, mrr))
            flag = 0
            if hit >= best_result[0]:
                best_result[0] = hit
                best_epoch[0] = epoch
                flag = 1
            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch
                flag = 1
            print('Best Result:')
            print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
                best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
            bad_counter += 1 - flag
            print('the single epoch time is :%d s' % (time.time() - epoch_start_time))
            if bad_counter >= opt.patience:
                break
        print('-------------------------------------------------------')
        print("now time: {} ,running time:{}".format(datetime.now(), time.time() - start))