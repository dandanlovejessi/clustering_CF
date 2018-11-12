# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:35:23 2018

@author: Administrator
用kmeans对movielens中的item聚类
"""

import numpy as np
import math


class Kmeans:
#构造函数——初始化一些值
    def __init__(self,filepath,n_clusters,user_num,item_num):
        self.filepath=filepath
        self.n_clusters=n_clusters
        self.user_num=user_num
        self.item_num=item_num
        self.dataset=self.loadData()
        self.cc=self.randCent()
        self.cc_new={}
        self.c={}
        self.c_new={}
        
#下载数据集，将数据集处理为item——user的形式 ，1682*943
    def loadData(self):
        dataset=np.zeros((self.item_num,self.user_num))
        with open(self.filepath,'r') as f:
            for line in f.readlines():
                curline=line.strip().split("\t")
                dataset[int(curline[1])-1,int(curline[0])-1]=int(curline[2])
        return dataset
#选取评分最多的7个电影生成聚类中心  
    def randCent(self):
        cc_item_dic={}
        cc={}
        for i in range(self.item_num):
            cc_item_dic[i]=len(np.nonzero(self.dataset[i,:])[0])
        cc_item_dic=sorted(cc_item_dic.items(),key=lambda one:one[1],reverse=True)
        for item in cc_item_dic[:self.n_clusters]:
            cc[item[0]]=self.dataset[item[0],:]
        return cc
#定义距离函数         
    def distance(self,item_id,cc_id):
        sum_num=0.0
        sum_den_1=0.0
        sum_den_2=0.0
        for i in range(len(self.dataset[item_id])):
            sum_num+=self.dataset[item_id][i]*self.cc[cc_id][i]
            sum_den_1+=self.dataset[item_id][i]**2
            sum_den_2+=self.cc[cc_id][i]**2
        return sum_num/(np.sqrt(sum_den_1)*np.sqrt(sum_den_2))
#利用7个聚类中心给item聚类
    def kmeans(self):       
        for item_id in range(self.item_num):
            max_cos =float('-inf')
            clusterindex=-1
            for cc_id in self.cc.keys():
                cos=self.distance(item_id,cc_id)
                if cos>max_cos:
                    max_cos=cos
                    clusterindex=cc_id
            self.c.setdefault(clusterindex,[]).append(item_id)
            self.c_new.setdefault(clusterindex,[]).append(self.dataset[item_id])
        return self.c
#主函数
if __name__=='__main__':
    k_means=Kmeans('F:/Machine/dataset/ml-100k/u.data',n_clusters=7,user_num=943,item_num=1682)
    print(k_means.kmeans())
        
        