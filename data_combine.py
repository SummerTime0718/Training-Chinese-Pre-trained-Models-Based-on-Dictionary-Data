import os
from tokenizer import Tokenizer
path = ["./data/allData2025_synthesize.txt","./data/allData2025.txt"]
out = "./data/allData2025_synthesize2.txt"
tokenizer_model = "tokenizer.model"
enc = Tokenizer(tokenizer_model)
data = []
max = 0
index = 0
for doc in path:
    with open(doc,mode = 'r',encoding ='utf-8') as f:
        for line in f.readlines():
            data.append(line)
            # if max <= len(line):
            #     max = len(line)
            #     index = data.index(line)

with open(out,mode = 'w',encoding = 'utf-8') as f:
    f.write(''.join(data))
        