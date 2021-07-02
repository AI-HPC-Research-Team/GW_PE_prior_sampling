import pandas as pd
from tqdm import tqdm
import sys
import wget
import tempfile

loc = sys.argv[1]
folder = pd.read_json(loc)
loc_len = len(folder)
print('len:', loc_len)
for i in tqdm(range(loc_len)):
    url = folder.strain[i]['url']
    print('downloading from:', url)
    file_name = wget.download(url)

url = folder.strain[0]['url']
print(url.split('/')[-1])
