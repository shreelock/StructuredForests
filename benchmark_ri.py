import json
import os
from rankimprovement import getfileobjs, get_file_ids, query_api
import pickle

ROOT = "/Users/sshn/shreelock/tmv/9/findPairs/"
IMGS_ROOT = os.path.join(ROOT, "selected-pairs")


def getbenchmark_map(file):
    bm_json = json.load(open(file, 'rb'))
    mapping = {}
    for item in bm_json.keys():
        id = item.encode('ascii')
        bmis = bm_json[id].encode('ascii')
        mapping[id] = bmis
    return mapping


def process_results(req, results, db_rank, target):
    found = False
    if req.status_code == 200:
        resp = json.loads(req.content)
        design_results = resp['tmv']['design_results']
        for item in design_results:
            id = item['tm_id']
            cumu_rank = item['rank']
            score = item['similarity']
            dataset = item['dataset'].encode('ascii')
            db_rank[dataset] += 1
            if id in target:
                results[dataset] = (db_rank[dataset], cumu_rank, "{0:.5f}".format(score))
                if len(results) == 2:
                    found = True
                    break
    return found, results, db_rank


if __name__ == '__main__':
    bm_file = ROOT + "benchmark.json"
    bm_mapping = getbenchmark_map(bm_file)
    b_ri = os.path.join(ROOT, 'benchmark_ri.txt')
    b_ri_pkl = os.path.join(ROOT, 'benchmark_ri.pkl')
    full_results = {}
    for key in sorted(bm_mapping.keys()):
        print "Processing {}".format(key)
        us = key.split("|")[0]
        eu = key.split("|")[1]
        op = "output"
        targets = bm_mapping[key] + ',' + us + ',' + eu

        parent = os.path.join(IMGS_ROOT, key)
        input_path = os.path.join(parent, eu)
        otput_path = os.path.join(parent, op)

        ip_filenames = filter(lambda name: name[-3:] in "jpg|png", os.listdir(input_path))
        infileobjs = getfileobjs(input_path, ip_filenames)

        op_filenames = filter(lambda name: name[-3:] in "jpg|png", os.listdir(otput_path))
        opfileobjs = getfileobjs(otput_path, op_filenames)

        infileids = get_file_ids(file_list=infileobjs)
        inresults = query_api(image_ids=infileids, target=targets)

        opfileids = get_file_ids(file_list=opfileobjs)
        opresults = query_api(image_ids=opfileids + infileids, target=targets)

        # unioning the list elements to the set.
        keyset = set()
        keyset |= set(inresults.keys())
        keyset |= set(opresults.keys())
        keyset |= set(targets.split(','))

        final_results = {}
        for k in keyset:
            irank = inresults[k][0] if k in inresults else 500
            orank = opresults[k][0] if k in opresults else 500
            final_results[k] = (irank, orank)

        full_results[key] = final_results

        fi = open(b_ri, 'a')
        fi.write("{}\n".format(key))
        for k in final_results:
            print k, final_results[k][0], final_results[k][1]
            fi.write("{}\t{}\t{}\n".format(k, final_results[k][0], final_results[k][1]))
    with open(b_ri_pkl, 'wb') as pklobj:
        pickle.dump(full_results, pklobj)

