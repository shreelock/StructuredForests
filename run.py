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

RESULTS_PP = 50
TOT_PAGES = 10
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
    "1.jpg": 'D0555382|000841671_0004',
    "2.jpg": 'D0587478|000841671_0003',
    "3.jpg": 'D0520772|000283940_0009',
    "4.jpg": 'D0526804|000283940_0011',
    "5.jpg": 'D0508615|000283940_0012',
    "6.jpg": 'D0508790|000283940_0013',
    "7.jpg": 'D0508789|000283940_0014',
    "8.jpg": 'D0520773|000283940_0017',
    "9.jpg": 'D0551873|000374038_0001',
    "10.jpg": 'D0541074|000374038_0002',
    "11.jpg": 'D0541546|000374038_0004',
    "12.jpg": 'D0533368|000374038_0005',
    "13.jpg": 'D0544247|000374038_0006',
    "14.jpg": 'D0540061|000374038_0008',
    "15.jpg": 'D0540063|000374038_0009',
    "16.jpg": 'D0549482|000374038_0015',
    "17.jpg": 'D0541072|000374038_0018',
    "18.jpg": 'D0547087|000457643_0011',
    "19.jpg": 'D0596414|000841671_0040'
}


def process_results(req, results, db_rank):
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


def fetch_results(results, db_count, image_ids=None, search_id=None, pg_num=None):
    if image_ids:
        values = {'user': USER, 'token': TOKEN, 'results_per_page': RESULTS_PP, 'image_boxes': json.dumps(image_ids)}

        apitime = time.time()
        r = requests.post(API_SEARCH_URL, data=values)
        print "query first time - {0:.2f}s ".format(time.time() - apitime)

        search_id = json.loads(r.content)['tmv']['search_id']
        found, vals, db_count = process_results(r, results, db_count)
    else:
        values = {'user': USER, 'token': TOKEN, 'search_id': search_id, 'page': pg_num, 'results_per_page': RESULTS_PP}

        apitime = time.time()
        r = requests.post(API_SEARCH_URL, data=values)
        print "query subsq time - {0:.2f}s ".format(time.time() - apitime)

        found, vals, db_count = process_results(r, results, db_count)

    return found, search_id, vals, db_count


def query_api(image_ids):
    results = {}
    found, search_id, results, db_count = fetch_results(results, db_count={'USD': 0, 'EUD': 0}, image_ids=image_ids)

    if not found:
        pg = 1
        while not found and pg < TOT_PAGES:
            found, _, results, db_count = fetch_results(results, db_count=db_count, search_id=search_id, pg_num=pg)
            pg += 1

    results_map[file_name] = results
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

    for i, (fname, fres) in enumerate(full_res.items()):
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
                "cut_im_sketc_res",
                "im_sketc_cut_res",
                "all_together_res",
                "im_im_sk_res",
                "im_cut_im_sk_res",
                "im_im_sk_cut_res"
                ])
    plt.show()
    pass


if __name__ == '__main__':
    op_file_name = sys.argv[1]
    op_pickle_file = sys.argv[1] + ".pkl"
    input_root = "toy"
    output_root = "edges"

    image_dir = os.path.join(input_root, "BSDS500", "data", "images", "test")
    file_names = filter(lambda name: name[-3:] == "jpg" or name[-3:] == "png", os.listdir(image_dir))

    results_map = {}
    results_pickle_obj = {}
    op_pkl_path = os.path.join(output_root, op_pickle_file)

    if 0 > 2:
        model = StructuredForests(options, rand=rand)
        model.train(bsds500_train(input_root))

        for file_name in sorted(file_names):
            print "processing {}".format(file_name)

            ipath = os.path.join(image_dir, file_name)
            img_skc = os.path.join(output_root, file_name[:-4] + "-proc.png")
            cut_img_skc = os.path.join(output_root, file_name[:-4] + "-proc1.png")
            img_skc_cut = os.path.join(output_root, file_name[:-4] + "-proc2.png")

            poly = get_object_polygon(ipath)
            cut_img = cutfrompoly(poly, img_path=ipath)

            # Original: Image, Sketch.
            model_start_time = time.time()
            test_single_image(model, img_path=ipath, opath=img_skc)
            print "fwd pass - {0:.2f}s ".format(time.time() - model_start_time)

            # Cut Image, Sketch
            model_start_time = time.time()
            test_single_image(model, img=cut_img, opath=cut_img_skc)
            print "fwd pass - {0:.2f}s ".format(time.time() - model_start_time)

            # Image, Cut Sketch
            model_start_time = time.time()
            edge = test_single_image(model, img_path=ipath)
            print "fwd pass - {0:.2f}s ".format(time.time() - model_start_time)
            cutedge = cutfrompoly(poly, img=edge)
            cv2.imwrite(img_skc_cut, cutedge)

            target = true_design_ids[file_name]
            i_file = {'file': open(ipath, 'rb')}
            img_skc_file = {'file': open(img_skc, 'rb')}
            cut_img_skc_file = {'file': open(cut_img_skc, 'rb')}
            img_skc_cut_file = {'file': open(img_skc_cut, 'rb')}

            file_ids = get_file_ids(file_list=[i_file, img_skc_file, cut_img_skc_file, img_skc_cut_file])

            print "processing photo"
            photo_result = query_api(image_ids=file_ids[0])

            print "processing sktchs"
            sktch_result = query_api(image_ids=file_ids[1])
            cut_img_skc_result = query_api(image_ids=file_ids[2])
            img_skc_cut_result = query_api(image_ids=file_ids[3])

            print "processing tgthr"
            img_img_skc_tg = query_api(image_ids=[file_ids[0], file_ids[1]])
            img_cut_img_skc_tg = query_api(image_ids=[file_ids[0], file_ids[2]])
            img_img_skc_cut_tg = query_api(image_ids=[file_ids[0], file_ids[3]])

            results = [photo_result, sktch_result, cut_img_skc_result, img_skc_cut_result, img_img_skc_tg,
                       img_cut_img_skc_tg, img_img_skc_cut_tg]
            results = fill_gaps(results)

            op_file_path = os.path.join(output_root, op_file_name)

            results_pickle_obj[file_name] = results

            with open(op_file_path, 'a') as f:
                f.write("Processing {}\n".format(file_name))
                for i, r in enumerate(results):
                    f.write("case {} : {}\n".format(i, r))

    if not os.path.isfile(op_pkl_path):
        with open(op_pkl_path, 'wb') as fileobj:
            pickle.dump(results_pickle_obj, fileobj)

    plot_results(op_pkl_path)
