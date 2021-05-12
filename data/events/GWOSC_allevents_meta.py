import urllib.request
import json
from tqdm import tqdm

def GWOSC_allevents():
    
    # 这部分代码会从网络上自动下载 JSON 数据
    f = urllib.request.urlopen('https://www.gw-openscience.org/eventapi/json/allevents/')
    allevents = json.loads(f.read())['events']
    return allevents

with open("./GWOSC_allevents.json",'w',encoding='utf-8') as json_file:
    json.dump(GWOSC_allevents(), json_file, ensure_ascii=False)

#check
allevents = {} #存放读取的数据
with open("./GWOSC_allevents.json",'r',encoding='utf-8') as json_file:
        allevents = json.load(json_file)

GWOSC_allevents_meta = {}
for event_version in tqdm(allevents.keys()):
    # 这部分代码会从网络上自动下载 JSON 数据
    f = urllib.request.urlopen(allevents[event_version]['jsonurl'])
    events_meta = json.loads(f.read())['events'][event_version]
    GWOSC_allevents_meta[event_version] = events_meta

with open("./GWOSC_allevents_meta.json",'w',encoding='utf-8') as json_file:
    json.dump(GWOSC_allevents_meta, json_file, ensure_ascii=False)

#check
GWOSC_allevents_meta = {} #存放读取的数据
with open("./GWOSC_allevents_meta.json",'r',encoding='utf-8') as json_file:
        allevents = json.load(json_file)
