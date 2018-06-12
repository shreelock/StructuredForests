from StructuredForests import *
import requests
import json
import time
from secret import *
RESULTS_PP = 40
NEXTP_RES = 500
rand = N.random.RandomState(1)
options = {
    "rgbd": 0,
    "shrink": 2,
    "n_orient": 4,
    "grd_smooth_rad": 0,
    "grd_norm_rad": 4,
    "reg_smooth_rad": 2,
    "ss_smooth_rad": 8,
    "p_size": 32,
    "g_size": 16,
    "n_cell": 5,

    "n_pos": 10000,
    "n_neg": 10000,
    "fraction": 0.25,
    "n_tree": 8,
    "n_class": 2,
    "min_count": 1,
    "min_child": 8,
    "max_depth": 64,
    "split": "gini",
    "discretize": lambda lbls, n_class:
    discretize(lbls, n_class, n_sample=256, rand=rand),

    "stride": 2,
    "sharpen": 2,
    "n_tree_eval": 4,
    "nms": True,
}
true_design_ids = {
    "1.jpg" : 'D0555382|000841671_0004',
    "2.jpg" : 'D0587478|000841671_0003',
    "3.jpg" : 'D0520772|000283940_0009',
    "4.jpg" : 'D0526804|000283940_0011',
    "5.jpg" : 'D0508615|000283940_0012',
    "6.jpg" : 'D0508790|000283940_0013',
    "7.jpg" : 'D0508789|000283940_0014',
    "8.jpg" : 'D0520773|000283940_0017',
    "9.jpg" : 'D0551873|000374038_0001',
    "10.jpg": 'D0541074|000374038_0002',
    "11.jpg": 'D0541546|000374038_0004',
    "12.jpg": 'D0533368|000374038_0005',
    "13.jpg": 'D0544247|000374038_0006',
    "14.jpg": 'D0540061|000374038_0008',
    "15.jpg": 'D0540063|000374038_0009',
    "16.jpg": 'D0549482|000374038_0015',
    "17.jpg": 'D0541072|000374038_0018',
    "18.jpg": 'D0551464|000457643_0011',
    "19.jpg": 'D0596414|000841671_0040'
}


def process_results(req, results):
    found = False
    if req.status_code == 200:
        resp = json.loads(req.content)
        design_results = resp['tmv']['design_results']
        for item in design_results:
            id = item['tm_id']
            rank = item['rank']
            score = item['similarity']
            dataset = item['dataset']
            if id in target:
                results[dataset] = (rank, score)
                if len(results) == 2:
                    found = True
                    break
    return found, results


def fetch_results(results, image_ids=None, search_id=None, pg_num=None):
    if image_ids:
        values = {'user': USER, 'token': TOKEN, 'results_per_page': RESULTS_PP, 'image_boxes': json.dumps(image_ids)}

        apitime = time.time()
        r = requests.post(API_SEARCH_URL, data=values)
        #print "query first time - {0:.2f}s ".format(time.time() - apitime)

        search_id = json.loads(r.content)['tmv']['search_id']
        found, vals = process_results(r, results)
    else:
        values = {'user': USER, 'token': TOKEN, 'search_id': search_id, 'page': pg_num, 'results_per_page': NEXTP_RES}

        apitime = time.time()
        r = requests.post(API_SEARCH_URL, data=values)
        #print "query subsq time - {0:.2f}s ".format(time.time() - apitime)

        found, vals = process_results(r, results)

    return found, search_id, vals


def query_api(image_ids):
    results = {}
    found, search_id, results = fetch_results(results, image_ids=image_ids)

    if not found:
        pg = 1
        while not found and pg < 6:
            found, _, results = fetch_results(results, search_id=search_id, pg_num=pg)
            pg += 1

    results_map[file_name] = results
    if found : print results
    else : print results, "search incomplete"
    return results


def get_file_ids(file_list):
    file_ids = []
    for f in file_list:
        r = requests.post(url=API_SEGMENT_URL, files=f, data={'user': USER, 'token': TOKEN})
        resp = json.loads(r.content)
        file_ids.append({"image_id": resp['image_id']})
    return file_ids


if __name__ == '__main__':
    input_root = "toy"
    output_root = "edges"

    model = StructuredForests(options, rand=rand)
    model.train(bsds500_train(input_root))

    image_dir = os.path.join(input_root, "BSDS500", "data", "images", "test")
    file_names = filter(lambda name: name[-3:] == "jpg" or name[-3:] == "png", os.listdir(image_dir))

    results_map = {}
    for file_name in file_names:
        print "processing {}".format(file_name)
        target = true_design_ids[file_name]

        ipath = os.path.join(image_dir, file_name)
        i_file = {'file': open(ipath, 'rb')}

        opath = os.path.join(output_root, file_name[:-4] + "-proc.png")
        o_file = {'file': open(opath, 'rb')}

        file_ids = get_file_ids(file_list=[i_file, o_file])

        model_start_time = time.time()
        test_single_image(model, ipath, opath)
        #print "fwd pass - {0:.2f}s ".format(time.time() - model_start_time)

        #print "processing photo"
        photo_result = query_api(image_ids=file_ids[0])

        #print "processing sktch"
        sktch_result = query_api(image_ids=file_ids[1])

        #print "processing tgthr"
        tgthr_result = query_api(image_ids=file_ids)

        with open('results.txt', 'a') as f:
            f.write("Processing {}\n".format(file_name))
            f.write("photo : {}\n".format(photo_result))
            f.write("sktch : {}\n".format(sktch_result))
            f.write("tgthr : {}\n".format(tgthr_result))



