import pickle
from benchmark_ri import ROOT
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt


def plothist(histlist, title):
    matplotlib.rcParams['xtick.labelsize'] = 5
    poslis = histlist[np.where(histlist > 0)]
    bins = [0, 20, 50, 100, 500, 1000, 2000]
    y, x, _ = plt.hist(poslis, bins=bins, rwidth=0.8)
    for i in range(len(y)): plt.text(x[i]*1.5, y[i] + .25, str(y[i]), color='blue')
    plt.xticks(bins, rotation=90)
    plt.title("{} - improvement".format(title))
    plt.tight_layout()
    plt.show()

    neglis = histlist[np.where(histlist < 0)]
    bins = [-2000, -1000, -500, -100, -50, -20, 0]
    y, x, _ = plt.hist(neglis, bins=bins, rwidth=0.8)
    for i in range(len(y)): plt.text(x[i]*1.5, y[i] + .25, str(y[i]), color='blue')
    plt.xticks(bins, rotation=90)
    plt.title("{} - decrement".format(title))
    plt.show()


if __name__ == '__main__':
    b_ri_pkl = os.path.join(ROOT, 'benchmark_ri.pkl')
    res = pickle.load(open(b_ri_pkl, 'rb'))

    exact_us = []
    exact_eu = []

    simlr_us = []
    simlr_eu = []

    exact_us_nf_g = []
    exact_us_nf_b = []
    exact_us_nf_nr = []

    exact_eu_nf_g = []
    exact_eu_nf_b = []
    exact_eu_nf_nr = []

    simlr_eu_nf_g = []
    simlr_eu_nf_b = []
    simlr_eu_nf_nr = []

    simlr_us_nf_g = []
    simlr_us_nf_b = []
    simlr_us_nf_nr = []

    for orig_pair in res.keys():
        usid = orig_pair.split("|")[0].strip()
        euid = orig_pair.split("|")[1].strip()
        for id in res[orig_pair].keys():
            irank, orank = res[orig_pair][id]

            if irank == 1000000 or orank == 1000000:
                nf_res_pair = (orig_pair, id, irank, orank)
                if id == usid:
                    if irank > orank:
                        exact_us_nf_g.append(nf_res_pair)
                    elif orank > irank:
                        exact_us_nf_b.append(nf_res_pair)
                    else:
                        exact_us_nf_nr.append(nf_res_pair)
                elif id == euid:
                    if irank > orank:
                        exact_eu_nf_g.append(nf_res_pair)
                    elif orank > irank:
                        exact_eu_nf_b.append(nf_res_pair)
                    else:
                        exact_eu_nf_nr.append(nf_res_pair)
                else:
                    if id[0] == 'D':
                        if irank > orank:
                            simlr_us_nf_g.append(nf_res_pair)
                        elif orank > irank:
                            simlr_us_nf_b.append(nf_res_pair)
                        else:
                            simlr_us_nf_nr.append(nf_res_pair)
                    else:
                        if irank > orank:
                            simlr_eu_nf_g.append(nf_res_pair)
                        elif orank > irank:
                            simlr_eu_nf_b.append(nf_res_pair)
                        else:
                            simlr_eu_nf_nr.append(nf_res_pair)
            else:  # none of the items "not found"
                rank_diff = irank - orank
                if id == usid:
                    exact_us.append(rank_diff)
                elif id == euid:
                    exact_eu.append(rank_diff)
                else:
                    if id[0] == 'D':
                        simlr_us.append(rank_diff)
                    else:
                        simlr_eu.append(rank_diff)

    # processing now.
    # processing exacts
    exact_us = np.array(exact_us)
    exact_us_nf_g = np.array(exact_us_nf_g)
    exact_us_nf_b = np.array(exact_us_nf_b)

    total_us_improvement = np.sum(exact_us > 0) + len(exact_us_nf_g)
    total_us_same = np.sum(exact_us == 0) + len(exact_us_nf_nr)
    total_us_decerment = np.sum(exact_us < 0) + len(exact_us_nf_b)

    print "EXACT US : tot : {}, tot-impr : {}, tot-same : {}, tot-decr : {}".format(
        total_us_improvement + total_us_same +
        total_us_decerment, total_us_improvement, total_us_same,
        total_us_decerment)

    exact_us_rankdiff_mean = np.mean(exact_us)
    exact_eu_rankdiff_mean = np.mean(exact_eu)

    # processing similars
    simlr_us = np.array(simlr_us)
    simlr_us_nf_g = np.array(simlr_us_nf_g)
    simlr_us_nf_b = np.array(simlr_us_nf_b)

    total_us_improvement_sm = np.sum(simlr_us > 0) + len(simlr_us_nf_g)
    total_us_same_sm = np.sum(simlr_us == 0) + len(simlr_us_nf_nr)
    total_us_decerment_sm = np.sum(simlr_us < 0) + len(simlr_us_nf_b)

    print "SIMLR US : tot : {}, tot-impr : {}, tot-same : {}, tot-decr : {}".format(
        total_us_improvement_sm + total_us_same_sm + total_us_decerment_sm, total_us_improvement_sm, total_us_same_sm,
        total_us_decerment_sm)

    simlr_eu = np.array(simlr_eu)
    simlr_eu_nf_g = np.array(simlr_eu_nf_g)
    simlr_eu_nf_b = np.array(simlr_eu_nf_b)

    total_eu_improvement_sm = np.sum(simlr_eu > 0) + len(simlr_eu_nf_g)
    total_eu_same_sm = np.sum(simlr_eu == 0) + len(simlr_eu_nf_nr)
    total_eu_decerment_sm = np.sum(simlr_eu < 0) + len(simlr_eu_nf_b)

    print "SIMLR EU : tot : {}, tot-impr : {}, tot-same : {}, tot-decr : {}".format(
        total_eu_improvement_sm + total_eu_same_sm + total_eu_decerment_sm, total_eu_improvement_sm, total_eu_same_sm,
        total_eu_decerment_sm)

    plothist(exact_us, "us - exact")
    plothist(simlr_us, "us - simlr")
    plothist(simlr_eu, "eu - simlr")

    simlr_us_rankdiff_mean = np.mean(simlr_us)
    simlr_eu_rankdiff_mean = np.mean(simlr_eu)

    print "us-exact-rankdiff-mean : {}\n" \
          "eu-exact-rankdiff-mean : {}\n" \
          "us-simlr-rankdiff-mean : {}\n" \
          "eu-simlr-rankdiff-mean : {}\n".format(exact_us_rankdiff_mean, exact_eu_rankdiff_mean, simlr_us_rankdiff_mean,
                                                 simlr_eu_rankdiff_mean)

    exact_us = np.array(exact_us)
    exact_eu = np.array(exact_eu)
    simlr_us = np.array(simlr_us)
    simlr_eu = np.array(simlr_eu)

    print "avg-exact-us-rank-impr : {}".format(np.mean(exact_us[np.where(exact_us > 0)]))
    print "avg-exact-eu-rank-impr : {}".format(np.mean(exact_eu[np.where(exact_eu > 0)]))
    print "avg-simlr-us-rank-impr : {}".format(np.mean(simlr_us[np.where(simlr_us > 0)]))
    print "avg-simlr-eu-rank-impr : {}".format(np.mean(simlr_eu[np.where(simlr_eu > 0)]))

    print "avg-exact-us-rank-decr : {}".format(np.mean(exact_us[np.where(exact_us < 0)]))
    print "avg-exact-eu-rank-decr : {}".format(np.mean(exact_eu[np.where(exact_eu < 0)]))
    print "avg-simlr-us-rank-decr : {}".format(np.mean(simlr_us[np.where(simlr_us < 0)]))
    print "avg-simlr-eu-rank-decr : {}".format(np.mean(simlr_eu[np.where(simlr_eu < 0)]))
