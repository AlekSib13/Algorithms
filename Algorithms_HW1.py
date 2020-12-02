#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd,matplotlib as mtl


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


X = np.array([[ 1,  1],
              [ 1,  1],
              [ 1,  2],
              [ 1,  5],
              [ 1,  3],
              [ 1,  0],
              [ 1,  5],
              [ 1, 10],
              [ 1,  1],
              [ 1,  2]])


# In[4]:


y = [45, 55, 50, 55, 60, 35, 75, 80, 50, 60]


# In[5]:


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err


# <h3>Домашнее задание</h3>

# 1. Подберите скорость обучения (eta) и количество итераций

# In[6]:


input_=int(input('Введите количество итераций: '))


# In[7]:


def change_parameter(param1):
    if param1<=10000 and param1>5000:
        eta=0.5
        return eta,param1
    elif param1<=5000 and param1>1000:
        eta=1e-1
        return eta,param1
    elif param1<=1000:
        eta=1e-2
        return eta,param1
    else:
        print('Не верно указано число итераций')


# In[8]:


def draw(*args):
    title_dict={'family':'Calibri','fontsize':16,'fontweight':'bold'}
    x_args,y_args,n_iter=args
    plt.figure(figsize=(10,8))
    plt.plot(x_args,y_args,color='red')
    plt.title('MSE dynamics',fontdict=title_dict)
    plt.xlabel('number of iters')
    plt.ylabel('MSE')
    if n_iter>1000:
        plt.yscale(value='log')
    plt.show()


# In[9]:


n = X.shape[0]
eta,n_iter=change_parameter(input_)

W = np.array([1, 0.5])
print(f'Number of objects = {n}        \nLearning rate = {eta}        \nInitial weights = {W} \n')
number_of_iters=[]
number_of_MSE=[]
for i in range(n_iter):
    y_pred = np.dot(X, W)
    err = calc_mse(y, y_pred)
    for k in range(W.shape[0]):
        W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y))
        MSE=round(err, 2)

        
    if i % 10 == 0:
        eta /= 1.1
    
    if i%10==0:
        number_of_iters.append(i)
        number_of_MSE.append(MSE)
    if n_iter<=500:
        if i%10==0:
            print(f'Iteration #{i}: W_new = {W}, MSE = {MSE},eta={eta}')
    elif n_iter>500 and n_iter<=1000:
        if i%20==0:
            print(f'Iteration #{i}: W_new = {W}, MSE = {MSE},eta={eta}')
    else:
        if i%50==0:
            print(f'Iteration #{i}: W_new = {W}, MSE = {MSE},eta={eta}')
            
draw(number_of_iters,number_of_MSE,n_iter)


# In[9]:


#Увеличение количества итераций, при сохранении скорости обучения или ее постепенном уменьшении сокращает ошибку постепенно.
#Если изначально берем бОльшую скорость обучения, например на 1e-1 или 0.5, то есть риск, что мы перескочим точку эксремума(точку минимума), что
#и прозошло. Далее по мере уменьшения скорости обучения, мы приблежаемся к точке экстремума(точка минимума). В решении данной задачи
#мы рассмотрели пример, когда взяли бОльшую скорость обучения, а потом резко ее уменьшили, на что увидели характерное поведение
#ошибки: сперва она сильна выросла, потом также сильно рухнула и далее держалась на том уже уровне, поскольку скорость обучения
#стала весьма мала


# -----------------------

# 2*. В этом коде мы избавляемся от итераций по весам, но тут есть ошибка, исправьте ее

# In[10]:


n = X.shape[0]

eta = 1e-2 
n_iter = 100

W = np.array([1, 0.5])
print(f'Number of objects = {n}        \nLearning rate = {eta}        \nInitial weights = {W} \n')

for i in range(n_iter):
    y_pred = np.dot(X, W)
    err = calc_mse(y, y_pred)
    W -= eta * (1/n * 2 * X.T @(y_pred - y))
    if i % 10 == 0:
        eta /= 1.1
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)},eta={eta}')
        
#Убрал внутренний цикл for, вместо W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y)) записал: W -= eta * (1/n * 2 * X.T @(y_pred - y))


# ---------------------

# 3*. Вместо того, чтобы задавать количество итераций, задайте другое условие останова алгоритма - когда веса перестают изменяться меньше определенного порога $\epsilon$.

# In[11]:


n = X.shape[0]
epsilon = 1e-8
eta = 1e-2 
#n_iter = 100
i=0

W_list=[]
W = np.array([1, 0.5])
print(f'Number of objects = {n}        \nLearning rate = {eta}        \nInitial weights = {W} \n')

while True:
    list(map(lambda x:W_list.append(x),W))
    y_pred = np.dot(X, W)
    err = calc_mse(y, y_pred)
    W -= eta * (1/n * 2 * X.T @(y_pred - y))
    i+=1
    if i % 10 == 0:
        eta /= 1.1
        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)},eta={eta}')
    try:
        if np.abs(W_list[-1]-W_list[-3])<=epsilon and np.abs(W_list[-2]-W_list[-4])<=epsilon:
            break
    except:
        continue
print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)},eta={eta}')


# In[111]:


#Понадобилось 1732 итерации, чтобы дойти до значений весов, разность с предыдущими значениями которых <=эпсилон

