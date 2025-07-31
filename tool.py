import os 
path = ["data/allData2025_synthesize.txt"]
with open(path[0],'r',encoding ='utf-8') as f:
    buf = set()
    # for line in f.readlines():
    #     for word in line:
    #         buf.add(word)
    for word in f.read():
        buf.add(word)
print("資料總字元數:%d"%(len(buf)))
print("資料:%s"%(list(buf)[10]))
out = "dict_set.txt"
with open(out,"w",encoding= 'utf-8') as f:
    for word in list(buf):
        f.write(word+"\n")
## 轉換成Unicode
# for i in range(len(buf)):
#     buf[i] = buf[i].encode()
# print(buf[:10])


class task:
    ## to translate training data to unicode
    def check_unicode(data_path):
        d_path  = data_path
        with open(d_path,mode = 'r',encoding ='utf-8') as f:
            buf = {}
            for line in f.readlines():
                buf.add(line.encode())
        return buf
    ## to check the word model generate is from training data
    def check(buf,word):
        if word in buf:
            return True
        else:
            return False
            