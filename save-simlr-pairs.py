from benchmark_ri import ROOT
import json
import requests
from PIL import Image
import io
import os

API_ROOT = "https://usd.tmv.io/lm/image_list/"
USD = 'USD/'
EUD = 'EUD/'


def getbenchmark_map(file):
    bm_json = json.load(open(file, 'rb'))
    mapping = {}
    for item in bm_json.keys():
        id = item.encode('ascii')
        bmis = bm_json[id].encode('ascii')
        mapping[id] = bmis
    return mapping


if __name__ == '__main__':
    bm_file = ROOT + "benchmark.json"
    bm_mapping = getbenchmark_map(bm_file)
    for b in bm_mapping:
        usid = b.split('|')[0].strip()
        euid = b.split('|')[1].strip()
        similar_ids = bm_mapping[b].split(',')

        bpath = os.path.join(ROOT, "similar-samples", b)
        if not os.path.exists(bpath):
            os.mkdir(bpath)
        else: continue
        os.mkdir(os.path.join(bpath, 'us'))
        os.mkdir(os.path.join(bpath, 'eu'))
        os.mkdir(os.path.join(bpath, 'sim'))

        usurls = requests.get(API_ROOT + USD + usid).content.split(',')
        euurls = requests.get(API_ROOT + EUD + euid).content.split(',')
        simurls = []
        for sim in similar_ids:
            if sim[0] == 'D':
                simu = requests.get(API_ROOT + USD + sim).content.split(',')
                for u in simu:
                    simurls.append(u)
            else:
                simu = requests.get(API_ROOT + EUD + sim).content.split(',')
                for u in simu:
                    simurls.append(u)

        print "For mapping {}".format(b)

        for idx, url in enumerate(usurls):
            i = requests.get(url, stream=True)
            if i.status_code == 200:
                i = Image.open(io.BytesIO(i.content))
                i.convert('RGB').save(os.path.join(bpath, 'us', '{}.jpg'.format(idx)))

        for idx, url in enumerate(euurls):
            i = requests.get(url, stream=True)
            if i.status_code == 200:
                i = Image.open(io.BytesIO(i.content))
                i.convert('RGB').save(os.path.join(bpath, 'eu', '{}.jpg'.format(idx)))

        for idx, url in enumerate(simurls):

            i = requests.get(url, stream=True)
            if i.status_code == 200:
                i = Image.open(io.BytesIO(i.content))
                i.convert('RGB').save(os.path.join(bpath, 'sim', '{}.jpg'.format(idx)))