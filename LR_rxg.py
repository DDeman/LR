#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/3/25'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

class LR_rxg():
    def __init__(self,l1=False,l2=False,lambda_l1=1,lambda_l2=1):
        self.l1 = l1
        self.l2 = l2
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
    def l1_fun(self,x,lambd):
        x = x.tolist()[0]
        for i in range(len(x)):
            if i > 0:
                x[i] = 1
            elif i < 0:
                x[i] = -1
        return x * lambd
    def l2_fun(self,x,lambd):
        return lambd* abs(x) / 2
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def fit(self,x,y,max_steps, alpha):
        x_mat = np.mat(x)      #(m,n)
        y_mat = np.mat(y).T    #(m,1)
        m = x_mat.shape[0]
        n = x_mat.shape[1]
        weight = np.zeros((1, n))
        for i in range(max_steps):
            if not self.l1 and self.l2:
                grad = (self.sigmoid(x_mat * weight.T) - y_mat).T * x_mat / m
                weight = weight - alpha * grad
            elif self.l1:
                l1_ = self.l1_fun(weight,self.lambda_l1)
                grad = (self.sigmoid(x_mat * weight.T) - y_mat).T * x_mat / m + l1_
                weight = weight - alpha * grad
            elif self.l2:
                l2_ = self.l2_fun(weight,self.lambda_l2)
                grad = (self.sigmoid(x_mat * weight.T) - y_mat).T * x_mat / m + l2_
                weight = weight - alpha * grad
            # loss_train = self.LR_loss(x_mat,y_mat, weight)
            # print(loss_train.mean())
            # self.loss_train.append(loss_train.mean())
        self.weight = weight
        return self.weight
    def predict(self,test,threshold):
        test_mat = np.mat(test)
        y_pred = self.sigmoid(test_mat * self.weight.T)
        y_p = []
        for i in y_pred:
            if i > threshold:
                y_p.append(1)
            else:
                y_p.append(0)
        return y_p
    def LR_loss(self,x,y, theta):
        print(np.log(self.sigmoid(x * theta.T)))
        loss =  np.multiply(np.log(self.sigmoid(x * theta.T)),y) + np.multiply(np.log(1 - self.sigmoid(x * theta.T)),(1-y))
        return loss



if __name__ == '__main__':
    # np.set_printoptions(suppress=True)
    data = load_breast_cancer()
    x = data.data  ##有30个特征
    y = data.target
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

    # fig = plt.figure()
    # # print(print(loss_train))
    # # print(loss_train)
    # # plt.plot([range(50)],loss_train)
    # # plt.show()acc = accuracy_score(y_test,y_pred)
    lr = LR_rxg(l2=True,lambda_l1=1)
    w = lr.fit(x_train, y_train, 50, 0.01)


    # for i in w:
    #     print(i)

    '''
    #画图
    steps = np.arange(10, 5000, 50)
    acc_list = []
    time_list = []
    for step in steps:
        lr = LR_rxg()
        lr.fit(x_train, y_train, step, 0.01)
        y_pred = lr.predict(x_test, 0.5)
        acc = accuracy_score(y_test,y_pred)
        acc_list.append(acc)
    print(acc_list)
    fig = plt.figure()
    plt.plot(steps,acc_list,label='step --- acc')
    plt.show()
    '''


