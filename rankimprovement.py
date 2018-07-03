from StructuredForests import *
import requests
import json
import time
from secret import *
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

RESULTS_PP = 100000
TOT_PAGES = 10
rand = np.random.RandomState(1)
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


def process_results(req, results, db_rank, target):
    found = False
    if req.status_code == 200:
        resp = json.loads(req.content)
        design_results = resp['tmv']['design_results']
        print "parsing through results"
        for item in design_results:
            id = item['tm_id']
            cumu_rank = item['rank']
            score = item['similarity']
            dataset = item['dataset'].encode('ascii')
            db_rank[dataset] += 1
            if id in target:
                results[id] = (db_rank[dataset], cumu_rank, "{0:.5f}".format(score))
                if len(results) == len(target.split(',')):
                    found = True
                    break
    return found, results, db_rank


def fetch_results(results, db_count, image_ids=None, search_id=None, pg_num=None, targetids=None):
    if image_ids:
        values = {'user': USER, 'token': TOKEN, 'results_per_page': RESULTS_PP, 'image_boxes': json.dumps(image_ids)}

        apitime = time.time()
        r = requests.post(API_SEARCH_URL, data=values)
        print "query first time - {0:.2f}s ".format(time.time() - apitime)

        search_id = json.loads(r.content)['tmv']['search_id']
        found, vals, db_count = process_results(r, results, db_count, targetids)
    else:
        values = {'user': USER, 'token': TOKEN, 'search_id': search_id, 'page': pg_num, 'results_per_page': RESULTS_PP}

        apitime = time.time()
        r = requests.post(API_SEARCH_URL, data=values)
        print "query subsq time - {0:.2f}s ".format(time.time() - apitime)

        found, vals, db_count = process_results(r, results, db_count, targetids)

    return found, search_id, vals, db_count


def query_api(image_ids, target):
    print "querying from api"
    results = {}
    found, search_id, results, db_count = fetch_results(results, db_count={'USD': 0, 'EUD': 0}, image_ids=image_ids, targetids = target)

    if not found:
        pg = 1
        while not found and pg < TOT_PAGES:
            found, _, results, db_count = fetch_results(results, db_count=db_count, search_id=search_id, pg_num=pg, targetids = target)
            pg += 1
    if found:
        print results
    else:
        print results, "not found in {} results".format(TOT_PAGES * RESULTS_PP)
    return results


def get_file_ids(file_list):
    file_ids = []
    for f in file_list:
        r = requests.post(url=API_SEGMENT_URL, files=f, data={'user': USER, 'token': TOKEN})
        resp = json.loads(r.content)
        file_ids.append({"image_id": resp['image_id']})
    return file_ids


def fill_gaps(items):
    for item in items:
        if 'USD' not in item:
            item['USD'] = (RESULTS_PP * TOT_PAGES, RESULTS_PP * TOT_PAGES, "0.00000")

        if 'EUD' not in item:
            item['EUD'] = (RESULTS_PP * TOT_PAGES, RESULTS_PP * TOT_PAGES, "0.00000")
    return items


def get_object_polygon(file_path):
    f = {'file': open(file_path, 'rb')}
    r = requests.post(url=API_SEGMENT_URL, files=f, data={'user': USER, 'token': TOKEN})
    resp = json.loads(r.content)
    segments = resp['segments']
    poly = []
    ppoly = []
    for s in segments:
        if s['segment_type'] == 'default_poly':
            poly = s['polygon']
            break
    for pt in poly:
        ppoly.append((pt[0], pt[1]))

    return ppoly


def cutfrompoly(poly, img=None, img_path=None):
    if img is None:
        img = cv2.imread(img_path)
    mask = Image.new('L', (img.shape[1], img.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
    mask = np.array(mask)

    idxs = mask == 0
    if len(img.shape) == 3:
        img[idxs] = [255, 255, 255]
    else:
        img[idxs] = 255
    return img


def plot_results(results_pickle):
    with open(results_pickle, 'rb') as file:
        full_res = pickle.load(file)

    print full_res
    rng = np.add(range(len(full_res)), 1)
    images_count = len(full_res)
    metric_count = len(full_res.values()[0])

    us_list = np.zeros((metric_count, images_count), np.uint8)
    eu_list = np.zeros((metric_count, images_count), np.uint8)
    names = [None] * images_count

    for i, (fname, fres) in enumerate(full_res.items()):
        names[i] = fname
        for idx, val in enumerate(fres):
            usval = fres[idx]['USD'][0]
            euval = fres[idx]['EUD'][0]

            us_list[idx][i] = 150 if usval == 500 else usval
            eu_list[idx][i] = 150 if euval == 500 else euval

    for i, usli in enumerate(us_list):
        plt.plot(rng, usli, color=np.random.rand(3, ), marker='o', linestyle='solid', linewidth=1, markersize=3)

    plt.xticks(np.arange(min(rng), max(rng) + 1, 1.0))
    plt.legend(["orig photo",
                "sketc_res",
                "im_im_sk_cut_res"
                ])
    plt.show()
    with open("table-file-us.txt", "wb") as tf:
        r, c = us_list.shape
        for j in range(c):
            tf.write("{}\t".format(names[j]))
            for i in range(r):
                tf.write("{}\t".format(us_list[i][j]))
            tf.write("\n")

    with open("table-file-eu.txt", "wb") as tf:
        r, c = eu_list.shape
        for j in range(c):
            tf.write("{}\t".format(names[j]))
            for i in range(r):
                tf.write("{}\t".format(eu_list[i][j]))
            tf.write("\n")
    pass


def getfileobjs(parent, filenames):
    objs = []
    for fname in filenames:
        objs.append({'file': open(os.path.join(parent, fname), 'rb')})
    return objs


if __name__ == '__main__':
    op_file_name = sys.argv[1]
    op_pickle_file = sys.argv[1] + ".pkl"
    train_images_root = "toy"
    INPUT_ROOT = "/Users/sshn/shreelock/tmv/9/findPairs/selected-pairs"  # os.path.join(input_root, "BSDS500", "data", "images", "test")

    results_pickle_obj = {}
    op_pkl_path = os.path.join(INPUT_ROOT, op_pickle_file)

    if 3 > 2:  # Just a Switch.
        model = StructuredForests(options, rand=rand)
        model.train(bsds500_train(train_images_root))

        for designpair in sorted(os.listdir(INPUT_ROOT)):
            if designpair[0] == '.' or "txt" in designpair : continue
            # input folders
            usid, euid = designpair.split("|")
            us_folder = os.path.join(INPUT_ROOT, designpair, usid)  # ground truths
            eu_folder = os.path.join(INPUT_ROOT, designpair, euid)  # sketches

            op_folder = os.path.join(INPUT_ROOT, designpair, "output")
            if not os.path.exists(op_folder):
                os.makedirs(op_folder)
                print "{} doing".format(designpair)
            else:
                print "{} already done".format(designpair)
                continue

            file_names = filter(lambda name: name[-3:] in "jpg|png", os.listdir(eu_folder))
            for file_name in sorted(file_names):
                print "processing {}".format(file_name)

                ipath = os.path.join(eu_folder, file_name)
                img_skc_cut = os.path.join(op_folder, file_name[:-4] + "-proc.png")

                poly = get_object_polygon(ipath)

                # Image, Cut Sketch
                model_start_time = time.time()
                edge = test_single_image(model, img_path=ipath)
                print "fwd pass - {0:.2f}s ".format(time.time() - model_start_time)
                cutedge = cutfrompoly(poly, img=edge)
                cv2.imwrite(img_skc_cut, cutedge)
                # written all the sketches

            skt_filenames = filter(lambda name: name[-3:] in "jpg|png", os.listdir(op_folder))
            des_filenames = filter(lambda name: name[-3:] in "jpg|png", os.listdir(eu_folder))

            sktfileobjs = getfileobjs(op_folder, skt_filenames)
            desfileobjs = getfileobjs(eu_folder, des_filenames)

            skfileids = get_file_ids(file_list=sktfileobjs)
            dsfileids = get_file_ids(file_list=desfileobjs)

            print "processing photo"
            photo_result = query_api(image_ids=dsfileids, target=designpair)

            print "processing sktchs"
            sketch_result = query_api(image_ids=skfileids, target=designpair)

            print "processing tgthr"
            tgethr_result = query_api(image_ids=dsfileids + skfileids, target=designpair)

            results = [photo_result, sketch_result, tgethr_result]
            results = fill_gaps(results)

            op_file_path = os.path.join(INPUT_ROOT, op_file_name)

            results_pickle_obj[designpair] = results

            with open(op_file_path, 'a') as f:
                f.write("Processing {}\n".format(designpair))
                for i, r in enumerate(results):
                    f.write("case {} : {}\n".format(i, r))
    #
    # if not os.path.isfile(op_pkl_path):
    #     with open(op_pkl_path, 'wb') as fileobj:
    #         pickle.dump(results_pickle_obj, fileobj)

    # plot_results(op_pkl_path)
