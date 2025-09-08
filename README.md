# 使用中文辭典資料訓練預訓練模型
## 參考資料
訓練程式來自(https://github.com/karpathy/llama2.c)
訓練方法請參考[原作者的README](README-llama2.md)
將[教育部四本辭典](https://language.moe.gov.tw/001/Upload/Files/site_content/M0001/respub/index.html)經程式轉換成可被訓練得文字檔訓練。
## 訓練步驟
### Pretokenize
1. 將[辭典資料預處理](https://huggingface.co/datasets/NingJing0718/Traditional_Chinese_Dictionary_Preprocess/tree/main)完成後，將tinystrory.py的pretokenize函式中shard_filename的值改成預處理後的資料路徑。
```python
def pretokenize(vocab_size):
    # iterate the shards and tokenize all of them one by one
    shard_filename = "./data/allData2025_synthesize.txt" ###Your pretokenize data path
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        dir_name = './data/tokenizer/tok' + 'str(vocab_size)'
        bin_dir = dir_name
        os.makedirs(bin_dir, exist_ok=True)
    txt_shard(shard_filename,vocab_size=vocab_size)
    # process all the shards in a process pool
    # fun = partial(txt_shard, vocab_size=vocab_size)
    # # Parallel excuting
    # with ProcessPoolExecutor() as executor: 
    #     executor.map(fun, enumerate(shard_filename))
    print("Done.")
```
	接著在CMD中使用以下指令執行。
```
python tinystory.py pretokenize
```
2. 更改tinystrory.py class PretokDataset中第一個shard_filenames的值改成你剛剛pretokenzie後檔案(.bin格式)的路徑。
```python
class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""
    
    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        
    def __iter__(self):
        DATA_CACHE_DIR = ""
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            shard_filenames = './data/allData2025_synthesize.bin'#m1125503
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            shard_filenames = './data/tokenizer/tok13304.bin' #m1125503
        print("Using file_name:",shard_filenames)#m1125503
        # train/test split. let's use only shard 0 for test split, rest train
        assert len(shard_filenames)>0, f"No bin files found in {shard_filenames}"
        while True:
            # rng.shuffle(shard_filenames) #m1125503
            shard = shard_filenames
            # open the dataset for reading but keep it on disk with memmap
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            # num_batches -= 1  # drop the last partial batch
            # assert num_batches > 0, "this shard is way too small? investigate."
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y
```
調整train.py中的參數。

| 參數名稱                        | 參數解釋                                                                                                                                                                                                                  |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| out_dir                     | 輸出路徑                                                                                                                                                                                                                  |
| max_iters                   | 訓練步數                                                                                                                                                                                                                  |
| eval_interval               | 多少訓練步數更新一次參數並輸出                                                                                                                                                                                                       |
| log_interval                | 多少訓練步數顯示eval_iters時的loss                                                                                                                                                                                              |
| eval_iters                  | 多少訓練步數更新一次參數                                                                                                                                                                                                          |
| eval_only                   | 在第一次評估後結束訓練                                                                                                                                                                                                           |
| always_save_checkpoint      | 在每次eval_iters後儲存check point                                                                                                                                                                                           |
| init_from                   | 'scratch\|resume'，scratch為從頭訓練，resume為從out_dir裡的checkpoint繼續訓練                                                                                                                                                        |
| batch_size                  | 可同時處理的數據數量，影響訓練穩定性以及訓練精度，太小的batch size會導致模型在參數的更新上有很大的波動                                                                                                                                                              |
| max_seq_len                 | 模型一次可以處理的最大token數，影響模型一次可以處理的prompt token數                                                                                                                                                                            |
| vocab_source                | 預設為llama2  <br>使用llama2的vocab或者自定vocab                                                                                                                                                                                |
| vocab_size                  | 預設為llama2，如果使用自訓練的tokenizer則填入tokenizer訓練時使用的vocab size                                                                                                                                                               |
| dim                         | 影響表達能力、模型性能及計算成本<br><br>1. 表達能力:embedding的dim，太小會限制了模型能夠學到的語義和語法信息，進而影響模型表達能力;太大也有可能導致過度擬合，若訓練數據太少的話可能會導致模型學到無意義的詞彙進而影響泛化能力<br>2. 較小的dim可以減少訓練成本<br>3. 較小的dim在處理簡單的任務上可能表現良好，但面對複雜的自然語言理解或生成任務時，這種限制的表達能力可能會降低模型的整體性能 |
| n_layers                    | decoder層數，影響表達能力、模型性能及計算成本  <br>如果 decoder 層數較少，模型的表達能力會受限，因為層數少的模型無法學習到深層次的語義和語法結構。這會使模型在處理複雜的語言任務時表現較差;層數多的 decoder 可以捕捉到更長的依賴關係，對長文本的理解和生成效果更好，這是許多大型語言模型成功的關鍵之一，但是這樣會使得模型有過度擬合的風險，須配合正則化技術（如 dropout）或其他方式來避免過度擬合 |
| n_heads                     | Query head的數量                                                                                                                                                                                                         |
| n_kv_heads                  | Value &Key head的數量                                                                                                                                                                                                    |
| multiple_of                 | 隱藏層數量，可以設置為8,16,32，會影響模型的表達能力且可以學習到更多特徵，但隱藏層過大也可能導致過度擬合                                                                                                                                                               |
| dropout                     | 將神經元輸出設置為0的機率，設置適當的drop out值可以防止過度擬合                                                                                                                                                                                  |
| gradient_accumulation_steps | 當GPU運算能力有限時，可以在多個小batch上累積梯度再一次進行權重更新，ex:當想用batch size=32訓練但GPU只能用batch size=8，可以設置gradient_accumulation_steps = 4，模型會在4個batch後累積梯度再進行權重更新，相當於使用batch size=32，相對的也會提高訓練時長                                             |
| learning_rate               | 學習率最大值                                                                                                                                                                                                                |
| max_iters                   | 最多訓練多少steps                                                                                                                                                                                                           |
| weight_decay                | 權重衰減速度                                                                                                                                                                                                                |
| warmup_iters                | 需要花多少steps數warm up，wram up可以避免剛開始訓練時梯度變化過大導致模型不穩定。如果設置過短，可能無法充分穩定模型的初期訓練；如果過長，則會延遲學習率的有效提升，進而影響模型的訓練效率。通常，warm up 階段會占整個訓練過程的1%-5%左右。                                                                                 |
| device                      | 使用GPU或者CPU訓練                                                                                                                                                                                                          |
| dtype                       | 計算的資料型態                                                                                                                                                                                                               |
| compile                     | 是否使用use PyTorch 2.0去加速模型                                                                                                                                                                                              |
調整完訓練參數後執行train.py
```
python train.py
```
## 訓練結果
### 模型
模型檔案在我的[Hugging Face](https://huggingface.co/NingJing0718/Llama2_zh)內
### 各模型Loss
| 模型名稱         | loss   | 訓練時間(單位：小時) |
| ------------ | ------ | ----------- |
| modelA-6000  | 0.8821 | 10          |
| modelA-12000 | 0.8512 | 20          |
| modelB-6000  | 0.6084 | 14          |
| modelB-12000 | 0.5875 | 30          |
| modelC-6000  | 0.6906 | 18          |
| modelC-12000 | 0.6589 | 35          |
### 評估
從各訓練好的資料中挑選來源於各辭典的訓練資料超過400tokens的句子各十句(重編國語辭典修定本二十句)，並裁切前50/100/200tokens作為評估測試資料，執行sample.py將結果輸出並手動評估訓練結果。
```
python sample.py --f=t --os=s
```

