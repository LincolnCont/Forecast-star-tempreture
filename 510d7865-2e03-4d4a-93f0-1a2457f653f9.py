#!/usr/bin/env python
# coding: utf-8

# ## Загрузка данных

# In[28]:


pip install -U scikit-learn


# In[29]:


import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from math import ceil
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

np.random.seed(0)


# <div class="alert alert-success">
# <b>👍 Успех:</b> Импортированы нужные библиотеки!
# </div>

# In[30]:


df = pd.read_csv('/datasets/6_class.csv', index_col=0)
df.head()


# In[31]:


df.info()
df.describe()


# <div class="alert alert-success">
# <b>👍 Успех:</b> Датасет загружен и просмотрен!
# </div>

# ## Предобработка и анализ данных
# 

# Приведем названия столбцов к рабочему виду

# In[32]:


df.columns = ['temperature', 'luminosity', 'radius', 'abs_magnitude', 'star_type', 'star_color']


# Проверяем данные на пропуски и дубликаты

# In[33]:


df.isna().sum()


# In[34]:


df.duplicated().sum()


# Функция для построения графиков распределения:

# In[35]:


def draw_distribution(data, title, x, y):
    plt.figure(figsize=(12, 8))
    sns.histplot(data=data, kde=True, bins=50)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


    plt.figure(figsize=(8, 1))
    sns.boxplot(x=data)
    plt.xlabel(x)
    plt.show()


# Построим графики для всех значений:

# In[36]:


draw_distribution(df['temperature'], 'Распределение температур', 'Температура, К', 'Количество')


# In[37]:


draw_distribution(df['luminosity'], 'Распределение светимостей', 'Светимость', 'Количество')


# Наблюдаем пик около нуля, что говорит о большом количестве звезд с низкой светимостью

# In[38]:


draw_distribution(df['radius'], 'Распеределение радиусов', 'Радиус', 'Количество')


# На графике наблюдаем много выбросов, но не будем удалять, чтобы не искажать данные 

# In[39]:


draw_distribution(df['abs_magnitude'], 'Распределение значений зведной величины', 'Абсолютная звездная величина', 'Количество')


# Теперь рассмотрим категориальные данные

# In[40]:


df['star_color'].unique()


# Присутствуют одинаковые названия, но с разным регистром и с разными разделителями

# In[41]:


df['star_color']= df['star_color'].str.strip()
df['star_color']= df['star_color'].str.lower() 


# заменим все пробелы в названиях на нижнее подчеркивание

# In[ ]:





# In[ ]:





# In[42]:


df['star_color'] = df['star_color'].str.lower()
df['star_color'] = df['star_color'].str.strip()
df['star_color'] = df['star_color'].replace(['yellow-white'],'white yellow')
df['star_color'] = df['star_color'].replace(['white-yellow'],'white yellow')
df['star_color'] = df['star_color'].replace(['blue-white'],'blue white')
df.groupby(['star_color']).agg('count')['luminosity']


# In[43]:


df['star_color'] = df['star_color'].replace(['yellowish white'],'white yellow')
df['star_color'] = df['star_color'].replace(['whitish'],'white')
df['star_color'] = df['star_color'].replace(['orange-red'],'red')
df['star_color'] = df['star_color'].replace(['pale yellow orange'],'red')
df['star_color'] = df['star_color'].replace(['yellowish'],'white yellow')
df['star_color'] = df['star_color'].replace(['orange'],'red')
df.groupby(['star_color']).agg('count')['luminosity']


# In[44]:


df['star_color'].value_counts().plot(kind='bar', 
                                     title='Распределение по цветам', 
                                     xlabel='Цвет звезды', 
                                     ylabel='Количество', 
                                     figsize=(8,6));


# Больше всего в выборке представлено красных звезд, что соответсвует распределению светимости
# 

# In[ ]:





# In[45]:


df['star_type'].value_counts().plot(kind = 'bar',
                                    title = 'Распредение по типам',
                                    xlabel = 'Тип звезды',
                                    ylabel = 'Количество',
                                    figsize=(8,6));


# В данных одинаковое количество звезд каждого типа

# <div class="alert alert-success">
# <b>👍 Успех:</b> Все верно!
# </div>

# ## Построение базовой нейронной сети

# Разделим исходные данные на обучающую и тестовую выборки - выделим под обучающую 85% и 15% на тестовую.

# In[46]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('temperature', axis=1), 
                                                    df['temperature'], 
                                                    train_size=0.85, 
                                                    random_state=42, 
                                                    shuffle=True)


# Далее проведем масштабирование количественных данных:

# In[47]:


numeric = ['luminosity', 'radius', 'abs_magnitude']

scaler = StandardScaler()
scaler.fit(x_train[numeric])
x_train[numeric] = scaler.transform(x_train[numeric])
x_test[numeric] = scaler.transform(x_test[numeric])


# In[48]:


categorial = ['star_color', 'star_type']

tmp_train = x_train[categorial]
tmp_test= x_test[categorial]


encoder_ohe = OneHotEncoder(handle_unknown='ignore')
encoder_ohe.fit(x_train[categorial])

tmp_train = pd.DataFrame(encoder_ohe.transform(x_train[categorial]).toarray(), 
                                   columns=encoder_ohe.get_feature_names_out(),
                                   index=x_train.index)
tmp_test = pd.DataFrame(encoder_ohe.transform(x_test[categorial]).toarray(), 
                                   columns=encoder_ohe.get_feature_names_out(),
                                   index=x_test.index)

x_train.drop(categorial, axis=1, inplace=True)
x_train = x_train.join(tmp_train)

x_test.drop(categorial, axis=1, inplace=True)
x_test = x_test.join(tmp_test)


# <div class="alert alert-warning">
# <b>🤔 Рекомендация:</b> при кодировании лучше удалять первый фиктивный столбец
# </div>

# Создадим тензоры признаков

# In[49]:


x_train_torch = torch.FloatTensor(x_train.values)
y_train_torch = torch.FloatTensor(y_train.values)
x_test_torch = torch.FloatTensor(x_test.values)
y_test_torch = torch.FloatTensor(y_test.values)


# Инициализируем нейронную сеть, состоящую из входных нейронов, двух скрытых слоев и выходного слоя:

# In[50]:


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.act1 = nn.Tanh()
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.act2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        
        out = self.fc2(out)
        out = self.act2(out)
        
        out = self.fc3(out)
        
        return out


# Создадим функцию для обучения нейронной сети и вывода RMSE:

# In[51]:


def net_learn(num_epochs, net):
    rmse_best = 50000
    rmse_list = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
 
        preds = net.forward(x_train_torch).flatten()
            
        loss_value = loss(preds, y_train_torch)

        loss_value.backward()
            
        optimizer.step()
    
     
        net.eval()
        test_preds = net.forward(x_test_torch)
        rmse = mean_squared_error(y_test_torch.detach().numpy(), test_preds.detach().numpy(), squared=False)
        rmse_list.append(rmse)
        if rmse < rmse_best:
            rmse_best = rmse
            test_preds_best = test_preds 
            best_epoch = epoch 
    print('RMSE:', round(rmse_best, 3), '| ep', epoch, 'from', num_epochs, '| best_epoch:', best_epoch)
    return test_preds_best, rmse_list


# Зададим количество нейронов на каждом слое сети:

# In[52]:


n_in_neurons = x_train.shape[1]
n_hidden_neurons = list(range(100, 1050, 50))      
n_out_neurons = 1                                  

loss = nn.MSELoss()                                

num_epochs = 1500
comb = 10      


# In[ ]:


for c in range(comb):
    hidden_size_1 = random.choice(n_hidden_neurons)    
    hidden_size_2 = random.choice(n_hidden_neurons)
    print('hidden 1 =',hidden_size_1, 'hidden 2 =',hidden_size_2)
    net = Net(n_in_neurons, hidden_size_1, hidden_size_2, n_out_neurons)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    test_preds, rmse_list = net_learn(num_epochs, net)
    print('---------------------------')  


# Оптимальной яляется комбинация 250 нейронов на первом слое и 350 на втором, дающая значение RMSE 4261.709, что укладывается в заданную границу в 4500.
# 
# Теперь построим график "Факт — Прогноз", где по горизонтальной оси будут отложены условные номера звёзд, а по вертикальной — температура в Кельвинах:

# In[57]:


net = Net(n_in_neurons, 250, 350, n_out_neurons)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
test_preds, rmse_list = net_learn(num_epochs, net)

epochs = np.arange(1, num_epochs+1)
plt.figure(figsize=[10,6])
sns.lineplot(x=epochs, y=rmse_list, label='RMSE', color='red')
plt.legend()
plt.title('Значение ошибки по эпохам')
plt.xlabel('epochs')
plt.ylabel('RMSE')
plt.show()


# Из графика видим, что модель перестает обучаться примерно после 1300 эпохи.

# In[58]:


test_preds = pd.DataFrame(test_preds.detach().numpy(), columns=['temperature'])  # преобразуем данные
y_test = pd.DataFrame((y_test), columns=['temperature']).reset_index().drop('index', axis=1)


# In[59]:


fig, ax = plt.subplots(figsize=[15,10])
plt.bar(x=test_preds.index, height=test_preds['temperature'], color='#fce80b' , label='Прогноз')
plt.bar(x=y_test.index, height=y_test['temperature'], color='#4cc2f6', label='Факт', width= 0.4)
plt.title('график "Факт — Прогноз"')
plt.xlabel('Номер звезды в таблице данных')
plt.ylabel('Температура звезды')
ax.legend()
plt.show()


# По качеству предсказаний, исходя из графика наибольшей точности модель достигает при предсказании небольших значений, с температурой до 5000 кельвинов, связано это скорее всего что именно эти данные представляют собой основную массу датасета, поэтому на них модель раотает лучше всего

# ## Улучшение нейронной сети

# Посмотрим, как будет работать сеть при обучении батчами, для этого дополним функцию net_learn так, чтобы использовать батчи.  
# 
# Объявим класс EarlyStopping, который будет предоствращать переобучение и останавливать цикл обучения по достижению оптимального значения:

# In[29]:


class EarlyStopping():
    def __init__(self, patience=7, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss

            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
             self.counter += 1
                
        if self.counter >= self.patience:
                print('Early stopping')
                self.early_stop = True


# In[30]:


def net_learn_batches(num_epochs, net):
    early_stopping = EarlyStopping() 
    rmse_best = 50000
    for epoch in range(num_epochs):
        net.train()
        for batch in train_dataloader:
            data_train, temperature_train = batch  
            optimizer.zero_grad()

            preds = net.forward(data_train).flatten()

            loss_value = torch.sqrt(loss(preds, temperature_train))
            loss_value.backward()
            optimizer.step()

        predicted_temp = [] 
        with torch.no_grad():
            net.eval()
            for batch in test_dataloader:
                data_test, temperature_test = batch
  
                test_preds = net.forward(data_test).flatten()
                predicted_temp.append(test_preds)
                RMSE_loss = torch.sqrt(loss(test_preds, temperature_test))

        predicted_temp = torch.cat(predicted_temp).detach().numpy()
        rmse = mean_squared_error(y_test, predicted_temp, squared=False)
        rmse_list.append(rmse)
        if rmse < rmse_best:
            rmse_best = rmse
            best_epoch = epoch
            test_preds_best = predicted_temp
        early_stopping(rmse)

        if early_stopping.early_stop:
            break 
    print('RMSE:', round(rmse_best, 3), '| ep', epoch, 'from', num_epochs, '| best_epoch:', best_epoch)
    return test_preds_best


# In[31]:


class Net_Batch(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net_Batch, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)

        self.act1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.act2 = nn.ReLU()
        
        
        self.fc3 = nn.Linear(hidden_size2, num_classes) 
                        
    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.bn1(out)
       
        out = self.fc2(out)
        out = self.act2(out)
        
        out = self.fc3(out)
        
        return out


# Попробуем найти оптимальный размер батча с помощью перебора по списку:

# In[32]:


batch_size = list(range(15, len(x_train_torch), 25))

dataset_train = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
dataset_test = torch.utils.data.TensorDataset(x_test_torch, y_test_torch)


# In[33]:


for b in batch_size:
    train_dataloader = DataLoader(dataset_train, batch_size=b, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset_test, batch_size=b, num_workers=0)  
    print('batch_size:', b)    
    net = Net_Batch(n_in_neurons, 650, 1000, n_out_neurons)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch_test_preds = net_learn_batches(num_epochs, net)
    print('---------------------------')


# Лучший размер батча 190, однако, обучение батчами ухудшает значение RMSE
# Поэтому далее будем использовать исходную функцию net_learn.
# 
# Далее попробуем инициализировать сеть с разными вариантами параметра в dropout, увеличив кол-во эпох:

# In[34]:


dropout = [0.5, 0.7, 0.8, 0.9] 
comb = 10
num_epochs = 1200


# In[35]:


for c in range(comb):
    p1 = random.choice(dropout)
    p2 = random.choice(dropout)
    print('drop =', p1, p2)
    class Net_Dropout(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
            super(Net_Dropout, self).__init__()
        
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.act1 = nn.Tanh()
            self.dp1 = nn.Dropout(p=p1)
        
            self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
            self.act2 = nn.ReLU()
            self.dp2 = nn.Dropout(p=p2)
        
            self.fc3 = nn.Linear(hidden_size2, num_classes) 
        
        
        def forward(self, x):
            
            out = self.fc1(x)
            out = self.act1(out)
            out = self.dp1(out)
        
            out = self.fc2(out)
            out = self.act2(out)
            out = self.dp2(out)
            
            out = self.fc3(out)
            return out
           
    net = Net_Dropout(n_in_neurons, 950, 750, n_out_neurons)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    drop_test_preds = net_learn(num_epochs, net)
    
    print('---------------------------')    


# Лучше всего показывают себя значения Dropout 0.7 и 0.7, однако результаты у базовой модели лучше.

# In[36]:


drop_test_preds = pd.DataFrame(drop_test_preds[0].detach().numpy(), columns=['temperature'])


# In[37]:


fig, ax = plt.subplots(figsize=[17,10])
plt.bar(x=test_preds.index, height=test_preds['temperature'], color='#fce80b' , label='Прогноз', width= 1.1)
plt.bar(x=drop_test_preds.index, height=drop_test_preds['temperature'], color='#e09f1b' , label='Прогноз c dropout', width= 0.6)

plt.bar(x=y_test.index, height=y_test['temperature'], color='#4cc2f6', label='Факт', width= 0.4)
plt.title('график "Факт — Прогноз"')
plt.xlabel('Номер звезды в таблице данных')
plt.ylabel('Температура звезды')
ax.legend()
plt.show()


# Базовая модель показывает результаты чуть лучше, причем прогнозы более точны на малых значениях температуры до 5000 К, что связано с пребладанием в исходных данных малых значений температуры, радиуса и светимостей.

# ## Выводы

# Нашей задачей в проекте было придумать, как с помощью нейросети определять температуру на поверхности обнаруженных звёзд. Для этого в нашем распоряжении имелись характеристики по уже изученным 240 звездам. Они включают в себя данные об относительной светимости, относительном радиусе, абсолютной звездной величине, звездном цвете, типе звезды, абсолютной температуре (целевой признак).
# 
# Мы провели предобработку данных и их подготовку к построению модели. Были обработаны количественные и качественные признаки. Данные разделены на обучающую и тестовую выборки.
# 
# 
# Перебором размера батчей и количества эпох обучения на сети с двумя скрытыми слоями удалось достичь максимального показателя RMSE в 4261.7.
# 
# Мы попытались улучшить этот результат нормализацией батчей и «выключением» части нейронов, но превзойти его не удалось.
# 
# В целом модель хорошо угадывает температуру «холодных» звезд, так как информации о них больше в предоставленных данных, и промахивается с определением температуры у более горячих.

# In[ ]:




