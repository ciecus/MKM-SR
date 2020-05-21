# MKM-SR
Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation


# Paper data and code
This is the code for the SIGIR2020 Paper:Incorporating User Micro-behaviors and Item Knowledge into Multi-task Learning for Session-based Recommendation. We have implemented our methods in pytorch.

Here are two datasets we used in our paper. After download the datasets, you can put them in the folder: data/
- KKbox: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
- JData: https://jdata.jd.com/html/detail.html?id=8
There is a small dataset demo included in the folder data/,which can be used to test the correctness of the code.


We have also inclueded some baseline codes in this paper.


# Usage
You need to run the file data/data_prepare.py first to preprocess the data.

For example: cd data; python data_prepare.py --dataset=demo

```
usage: prepare.py [--dataset Demo][--remove_new_item]
optional arguments:
--dataset DATASET_PATH dataset name: Demo/Jdata/KKbox
```

Then you can run the file ```main.py``` to train the model.
```
usage: main.py 
optional arguments:
  --dataset             dataset name
  --batchSize           input batch size
  --hiddenSize          hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --remove_new_items    whether keep new item
  --mode                model mode,there we only keep MKM_SR,if you need other mode, you can contact with us, or use the ablation version.
  

```
