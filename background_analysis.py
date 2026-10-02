#!/usr/bin/env python
# coding: utf-8

import os
import re
import subprocess
import traceback
import pylab
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sklearn.cluster

from plotmtv import get_curves
from useful_routines import save_pickled_file, get_pickled_file


def background_analysis(directory, display=False):
    os.system(f"touch {directory}")
    dbc = subprocess.getoutput(f'find {directory} -iname "dozor_background.mtv"').split(
        "\n"
    )

    print(f"found {len(dbc)} background plots")

    Xs = []
    Ys = []
    Is = []
    for db in dbc:
        cs = get_curves(db)
        xs = np.array(cs[0]["x"])
        Xs.append(xs)
        ys = np.array(cs[0]["y"])
        Ys.append(ys)
        i = interp1d(xs, ys, bounds_error=False, fill_value="extrapolate")
        Is.append(i)

    limit_start = max([item[0] for item in Xs])
    limit_end = min([item[-1] for item in Xs])

    evaluation_points = np.linspace(limit_start, limit_end, 273)

    Ysn = []
    for i in Is:
        ys = i(evaluation_points)
        # eps_norm = np.logical_and(evaluation_points>=0.11, evaluation_points<0.15)
        # norm = ys[eps_norm].mean()
        norm = quad(i, evaluation_points[0], evaluation_points[-1])[0]
        ysn = ys / norm
        Ysn.append(ysn)

    if display:
        pylab.figure()
        for ysn in Ysn:
            pylab.plot(evaluation_points, ysn, "-")
        pylab.show()

    return evaluation_points, Ysn


def cluster(
    Ysn=None,
    evaluation_points=None,
    n_clusters=3,
    display=False,
    save=True,
    directory="./",
):
    if Ysn is None:
        evaluation_points, Ysn = background_analysis(directory)

    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(Ysn)

    ## Cristal direct cluster
    if n_clusters == 3:
        cristal_direct_nl, mitegen_nl, nylon = (
            kmeans.cluster_centers_[
                :, np.logical_and(evaluation_points > 0.062, evaluation_points < 0.082)
            ]
            .sum(axis=1)
            .argsort()
        )
        nylon_cd, mitegen_cd, cristal_direct = (
            kmeans.cluster_centers_[
                :, np.logical_and(evaluation_points > 0.024, evaluation_points < 0.047)
            ]
            .sum(axis=1)
            .argsort()
        )
        cristal_direct_mt, nylon_mt, mitegen = (
            kmeans.cluster_centers_[:, evaluation_points > 0.085].sum(axis=1).argsort()
        )

        print(
            f"labels\ncristal direct {cristal_direct} ({cristal_direct_mt}, {cristal_direct_nl})\nmt {mitegen} ({mitegen_nl}, {mitegen_cd})\nnylon {nylon} ({nylon_cd}, {nylon_mt})"
        )

        labels = {
            cristal_direct: "cristal direct",
            mitegen: "mitegen loop",
            nylon: "nylon loop",
        }

    elif n_clusters == 2:
        cristal_direct_nl, nylon = (
            kmeans.cluster_centers_[
                :,
                np.logical_and(
                    evaluation_points > 1.0 / 4.02**2,
                    evaluation_points < 1.0 / 3.5**2,
                ),  # (0.062 .082)
            ]
            .sum(axis=1)
            .argsort()
        )
        nylon_cd, cristal_direct = (
            kmeans.cluster_centers_[
                :,
                np.logical_and(
                    evaluation_points > 1.0 / 6.45**2,
                    evaluation_points < 1.0 / 4.61**2,
                ),
            ]
            .sum(axis=1)
            .argsort()
        )

        print(
            f"labels\ncristal direct {cristal_direct} ({cristal_direct_nl})\nnylon {nylon} ({nylon_cd})"
        )

        labels = {
            cristal_direct: "cristal direct",
            nylon: "nylon loop",
        }

    if display:
        pylab.figure()
        for k, c in enumerate(kmeans.cluster_centers_):
            pylab.plot(evaluation_points, c, label=labels[k])
        pylab.legend()
        pylab.savefig(os.path.join(directory, "background_support_type.png"))
        pylab.show()

    results = {
        "predictor": kmeans,
        "evaluation_points": evaluation_points,
        "labels": labels,
    }

    if save:
        save_pickled_file(
            os.path.join(directory, "support_type_predictor.pickle"), results
        )

    return results


def _analyze_background(  
    dozor_background_mtv_file,
    predictor=None,
    evaluation_points=None,
    labels=None,
    default="/usr/local/experimental_methods/support_type_predictor.pickle",
    verbose=False,
):
    if predictor is None:
        stp = get_pickled_file(default)
        predictor = stp["predictor"]
        evaluation_points = stp["evaluation_points"]
        labels = stp["labels"]

    os.system(f"touch {os.path.dirname(dozor_background_mtv_file)}")
    cs = get_curves(dozor_background_mtv_file)
    xs = np.array(cs[0]["x"])
    ys = np.array(cs[0]["y"])
    i = interp1d(xs, ys, bounds_error=False, fill_value="extrapolate")

    y = i(evaluation_points)
    
    try:
        raw_result = predictor.predict(y.reshape(1, -1))
        support_type = labels[raw_result[0]]
    except:
        print("could not determine the loop type")
        support_type = "unknown"

    if support_type == "mitegen loop":
        if y.max() >= 0.95 * y[0]:
            support_type = "cristal direct"

    if support_type == "cristal direct":
        if evaluation_points[y.argmax()] < 1.0 / 7.0**2:
            support_type = "mitegen loop"

    if verbose:
        print("raw result", raw_result)
        print(f"support type seems to be {support_type}")

    background_maximum = max(ys)
    background_lowres_integral = quad(i, 0.007, 0.02)[0]

    result = {
        "support type": support_type,
        "background maximum": background_maximum,
        "background lowres integral": background_lowres_integral,
    }

    return result

def analyze_background(
    dozor_background_mtv_file,
    predictor=None,
    evaluation_points=None,
    labels=None,
    default="/usr/local/experimental_methods/support_type_predictor.pickle",
    verbose=False,
    force=False,
):
    result_pickle_file = dozor_background_mtv_file.replace(".mtv", "_analysis.pickle")
    result_log_file = dozor_background_mtv_file.replace(".mtv", "_analysis.log")
    
    if not force and os.path.isfile(result_pickle_file):
        result = get_pickled_file(result_pickle_file)
    else:
        result = _analyze_background(
            dozor_background_mtv_file,
            predictor=predictor,
            evaluation_points=evaluation_points,
            labels=labels,
            default=default,
            verbose=verbose,
        )
        
        f = open(result_log_file, "w")
        for key in ["support type", "background maximum", "background lowres integral"]:
            f.write(f'{key}: {result[key]}\n')
        f.close()
        
        save_pickled_file(result_pickle_file, result)
    
    return result


def analyze_progression(directory, stp=None):
    search_t = re.compile('.*transmission_(\d*).*')
    search_r = re.compile('.*run_(\d*).*')
    #search_e = re.compile('.*exposure_(\d*).*')
    #search_d = re.compile('.
    lin = f'find {directory} -iname "dozor_background.mtv"'
    bf = subprocess.getoutput(lin)
    bfs = bf.split("\n")
    print(f'{len(bfs)} background files found')
    if stp is None:
        stp = get_pickled_file("/usr/local/experimental_methods/support_type_predictor.pickle")
    results = {}
    ts, rs = [], []
    for b in bfs:
        try:
            r = int(search_r.findall(b)[0])
            t = int(search_t.findall(b)[0])
            if t not in ts: ts.append(t)
            if r not in rs: rs.append(r)
            if t not in results:
                results[t] = {}
            results[t][r] = analyze_background(b, **stp)["background lowres integral"]
        except:
            traceback.print_exc()
            print(f"problem with {b}")
    print(f"results\n{results}")
    pylab.figure(figsize=(16, 9))
    pylab.title("effect of transmission on background", fontsize=24)
    for r in sorted(rs):
        curve = []
        for t in sorted(ts):
            try:
                curve.append((t, results[t][r]))
            except:
                pass
        curve = np.array(curve)
        print(curve)
        if len(curve.shape) == 2:
            pylab.plot(curve[:, 0], curve[:, 1], 'o', label=f"run {r:02d}")
    pylab.xlabel("transmission", fontsize=18)
    pylab.ylabel("Lowres integral", fontsize=18) 
    pylab.legend()
    pylab.savefig(os.path.join(directory, "transmission_progression.png"))
    pylab.show()
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--entity",
        default="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA",
        type=str,
        help="entity",
    )

    parser.add_argument(
        "--transmission_progression",
        action="store_true",
        help="analyze transmission effect on the background",
    )
    
    
    args = parser.parse_args()
    print(args)
    
    if args.transmission_progression:
        analyze_progression(args.entity)
    else:
        if os.path.isdir(args.entity):
            background_analysis(args.entity)
        else:
            analyze_background(args.entity)
