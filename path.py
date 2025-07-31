import os
DATA_CACHE_DIR = ""
bin_dir = os.path.join(DATA_CACHE_DIR, "./data/allData.bin")
print(bin_dir)
print(os.path.isfile(bin_dir))