"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
import argparse
import string

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
prompt_args = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
prompt_args.add_argument("--os","--output_size",help="output長度",default='s',choices=["s", "m", "l"])
prompt_args.add_argument("--f","--file",help="是否使用file當作是prompt來源",default="f" ,choices=['t','f'])
args = prompt_args.parse_args()
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#do file i/o
# 處理prompt檔案
match(args.f):
    case 't':
        ts = open('prompt.txt','r',encoding='utf-8')
        count = len(open("prompt.txt",'r',encoding='utf-8').readlines()) #計算Prompt檔案行數
        part_count = count//3 #一個part有多少prompt part_count = 50
        result = open('eval_result.txt','w+',encoding='utf-8')
        prompt_buf = [[0]*part_count for i in  range(3)]
        prompt_buf2 = [part_count]

        for i in range(3):
            for j in range(part_count):
                t = ts.readline()
                prompt_buf[i][j] = t.replace("\n","")
match(args.os):
            case 's':
                max_new_tokens = 50
            case 'm':
                max_new_tokens = 100
            case 'l':
                max_new_tokens = 200

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
""" models info
0. 20250326-project A model ,use origin dictionary data and train 6000 steps
1. 20250402-project A model ,use origin dictionary data and train 12000 steps
2. 20250507-project B model,use origin dictionary&extend data and train 6000 steps
3. 20250706-project B model,use origin dictionary&extend data and train 12000 steps
4. 20250514-project B model ,use origin dictionary&extend data and train 24000 steps
5. 20250607-project C model ,use origin dictionary&extend data&1000 lines synthesize data and train 6000 steps
6. 20250611-project C model ,use origin dictionary&extend data&1000 lines synthesize data and train 12000 steps
7. 20250617-project C model ,use origin dictionary&extend data&1000 lines synthesize data and use custom tokenizer training 12000 steps
8. 20250626-project D model ,use origin dictionary&extend data&1000 lines synthesize data training 6000 steps ,then append origin data behind training data above training 6000 steps
"""
models = ['out/zh_data-20250326/ckpt.pt',
          'out/zh_data-20250402/ckpt.pt',
          'out/zh_data-20250507/ckpt.pt',
          'out/zh_data-20250706/ckpt.pt',
          'out/zh_data-20250514/ckpt.pt',
          'out/zh_data-20250607/ckpt.pt',
          'out/zh_data-20250611/ckpt.pt',
          'out/zh_data-20250617/ckpt.pt',
          'out/zh_data-20250626/ckpt.pt']
tokenizer_model = ['','./data/tokenizer/tok13304.model']
checkpoint = models[2]
print("Model name:%s"%(checkpoint))
# start = input("Your prompt:")
# max_new_tokens = 50
num_samples = 1 # number of samples to draw
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = tokenizer_model[0] # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = True # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config files
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = enc.encode(start, bos=True, eos=False)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

input_len = {0:"短Prompt",1:"中Prompt",2:"長Prompt"}

# run generation
match(args.f):
    case 't':
        for i in range(count): #count =150
            start = prompt_buf[i//part_count][i%part_count] # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
            device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
            ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

            # init from a model saved in a specific directory
            checkpoint_dict = torch.load(checkpoint, map_location=device)
            gptconf = ModelArgs(**checkpoint_dict['model_args'])
            model = Transformer(gptconf)
            state_dict = checkpoint_dict['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict, strict=False)

            model.eval()
            model.to(device)
            if compile:
                model = torch.compile(model) # requires PyTorch 2.0 (optional)

            # load the tokenizer
            vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
            vocab_size = gptconf.vocab_size
            if tokenizer:
                # a specific tokenizer is provided, use it
                tokenizer_model = tokenizer
            else:
                # let's try to find the tokenizer model automatically. bit gross here...
                query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
                tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
            enc = Tokenizer(tokenizer_model=tokenizer_model)
            start_ids = enc.encode(start, bos=True, eos=False)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            with torch.no_grad():
                with ctx:
                    for k in range(num_samples):
                        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                        res = enc.decode(y[0].tolist())
                        l = len(start)
                        if ((i+1)%part_count) == 0:
                            num = str(part_count)
                        elif ((i+1)%part_count) == 1:
                            print(input_len[(i+1)//part_count])
                            num  = "### %s\n"%(input_len[(i+1)//part_count]) + str((i+1)%part_count)
                        else:
                            num = str((i+1)%part_count)
                        res_buf = num + ". " + start + "<font color=red>" + res[l:] + "</font>\n" \
                        + "- [ ] 有通順語句、詞彙\n"\
                        + "- [ ] 正確銜接標點符號\n"\
                        + "- [ ] 標點符號成對\n"\
                        + "- [ ] 沒有亂碼\n"\
                        # + "- [ ] 情節不合理\n" \
                        # + "- [ ] 出現不合理的單字\n"\
                        # + "- [ ] 文意與沒有prompt連貫\n"
                        result.write(res_buf)
        ts.close()
        result.close()
    case 'f':
        end = False
        while(end == False):
            start = input("Your prompt:")
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
            device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
            ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

            # init from a model saved in a specific directory
            checkpoint_dict = torch.load(checkpoint, map_location=device)
            gptconf = ModelArgs(**checkpoint_dict['model_args'])
            model = Transformer(gptconf)
            state_dict = checkpoint_dict['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict, strict=False)

            model.eval()
            model.to(device)
            if compile:
                print("Compiling the model...")
                model = torch.compile(model) # requires PyTorch 2.0 (optional)

            # load the tokenizer
            vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
            vocab_size = gptconf.vocab_size
            if tokenizer:
                # a specific tokenizer is provided, use it
                tokenizer_model = tokenizer
            else:
                # let's try to find the tokenizer model automatically. bit gross here...
                query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
                tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
            enc = Tokenizer(tokenizer_model=tokenizer_model)
            start_ids = enc.encode(start, bos=True, eos=False)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            with torch.no_grad():
                    with ctx:
                        for k in range(num_samples):
                            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                            res = enc.decode(y[0].tolist())
                            l = len(start)
                            print("Your Prompt:",start)
                            print("Results:")
                            print(res[l:])
                            print('---------------')
            if(input("Countinue?")=='y'):
                end = False
            else:
                end =True
        
