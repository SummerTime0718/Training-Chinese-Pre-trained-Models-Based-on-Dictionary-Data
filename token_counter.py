import os
import sentencepiece as spm
import sys
path = ["./data/allData2025.txt","./data/allData2025_extend.txt","./data/allData2025_synthesize.txt"]
tok_path = "./tokenizer.model"

def count_tok(sentence):
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    tokens = sp.encode(sentence, out_type=int)
    return len(tokens)


for l in range(len(path)):
    with open(path[l],mode='r',encoding='utf-8') as f:
        data = f.readlines()
        print("%s中有%d筆資料"%(path[l],len(data)))
        word= []
        tokens = 0
        byte = 0
        for s in data:
            s =  s.replace("\n","")
            # tokens += count_tok(s)
            byte += sys.getsizeof(s)
        # print("%s中共有%dtokens"%(path[0],tokens))
        print("%s中共有%dbytes"%(path[0],byte))

        