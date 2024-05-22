import numpy as np 
import os 
# entries = os.listdir('my_directory/')

txt =np.load('/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/labels.npy').tolist()
feature = np.load('/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/init_features.npy').tolist()
# init_features= np.load('/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/init_features.npy')
# check = 'Yukiko_Okudo'
print(len(txt))
# print(feature[0])
# print(check in txt)
# count = 0 
# path = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/lfw'
# for folder_name in os.listdir(path):
#     count+=1
# print(count)