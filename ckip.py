import os
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ws = WS("../data", disable_cuda=False)
pos =POS("../data")
# fpath = './data/allData.txt'

# with open(fpath,'r',encoding="UTF-8") as f:
#     data = f.read()

word ="傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。"
# print(data[:10])
# tokenize = ws(data[:100])

tokenize = ws(word,sentence_segmentation = True)
pos_list =pos(tokenize)
for w,p in zip(tokenize,pos_list):
    print(f'{w}',end='\u3000')

del ws 