#!/usr/bin/env python

import glob
import subprocess
import re
import os
import pickle
import time
import numpy as np

from report import get_xds_results


# table1_pattern = f"     {key}[\ ]*([\d\.]*)[\ ]*([\d\.]*)[\ ]*([\d\.]*)"
special_characters = "()&|+-^"


def get_template(directory):
    _temp = directory.strip("/main")
    return os.path.basename(_temp)


def get_ISa_table(xds_directory):
    correct = os.path.join(xds_directory, "CORRECT.LP")
    if os.path.isfile(correct):
        find_ISa_table = f'grep "     a        b          ISa" -A1 {correct}'
        ISa_table = subprocess.getoutput(find_ISa_table)
    else:
        ISa_table = None
    return ISa_table


def get_autoproc_results(
    autoproc_directory, subdir="HDF5_1", table1="staraniso_alldata-unique.table1"
):
    example_table = """        
     Low resolution limit                      87.390      87.390       3.812
     High resolution limit                      3.423      12.089       3.423
     Rmerge  (all I+ & I-)                      0.600       0.144       2.654
     Rmerge  (within I+/I-)                     0.626       0.147       2.465
     Rmeas   (all I+ & I-)                      0.623       0.152       2.760
     Rmeas   (within I+/I-)                     0.669       0.157       2.646
     Rpim    (all I+ & I-)                      0.164       0.045       0.749
     Rpim    (within I+/I-)                     0.236       0.056       0.959
     Total number of observations               65632        2506        3076
     Total number unique                         4559         228         228
     Mean(I)/sd(I)                                4.5        12.1         1.1
     Completeness (spherical)                    56.2       100.0        10.4
     Completeness (ellipsoidal)                  85.4       100.0        36.7
     Multiplicity                                14.4        11.0        13.5
     CC(1/2)                                    0.987       0.994       0.544
     Anomalous completeness (spherical)          54.7       100.0         9.3
     Anomalous completeness (ellipsoidal)        85.5       100.0        35.1
     Anomalous multiplicity                       8.1         7.7         7.7
     CC(ano)                                    0.017       0.392      -0.084
     |DANO|/sd(DANO)                            0.841       0.800       0.841
"""

    results = {}
    for line in example_table.split("\n"):
        key = line[: len("     Anomalous completeness (ellipsoidal))")].strip(" ")
        if key != "":
            results[key] = None

    tfile = os.path.join(autoproc_directory, subdir, table1)
    if os.path.isfile(tfile):
        table1 = open(tfile).read()
        # print(table1)
        for key in results:
            # pattern = f"     {key}[\ ]*([\d\.\-]*)[\ ]*([\d\.\-]*)[\ ]*([\d\.\-]*)"
            sk = key
            for c in special_characters:
                sk = sk.replace(c, f"\{c}")

            pattern = f".*{sk}.*"
            try:
                found = re.findall(pattern, table1)
                #print(f"{key} found {found}")
                numbers = found[0].replace(key, "").strip(" ").split()
                #print(f"numbers {numbers}")
                try:
                    r = list(map(float, numbers))
                except ValueError:
                    r = numbers
            except:
                r = None, None, None
                traceback.print_exc()
            results[key] = r

    results.update(get_xds_results(os.path.join(autoproc_directory, subdir)))

    return results


def get_xdsme_results(xdsme_directory, subdir="", table1="aimless.SUMM.log"):
    example_table = """
                                           Overall  InnerShell  OuterShell
Low resolution limit                       48.87     48.87      3.84
High resolution limit                       3.74     16.72      3.74

Rmerge  (within I+/I-)                     0.726     0.309     1.193
Rmerge  (all I+ and I-)                    0.779     0.314     1.392
Rmeas (within I+/I-)                       0.891     0.363     1.494
Rmeas (all I+ & I-)                        0.888     0.356     1.610
Rpim (within I+/I-)                        0.504     0.186     0.881
Rpim (all I+ & I-)                         0.411     0.163     0.780
Rmerge in top intensity bin                0.353        -         -
Total number of observations               21207       401       852
Total number unique                         5304       101       232
Mean((I)/sd(I))                              1.5       2.4       1.0
Mn(I) half-set correlation CC(1/2)         0.862     0.945     0.372
Completeness                                71.5      91.6      45.1
Multiplicity                                 4.0       4.0       3.7
Mean(Chi^2)                                 0.93      1.45      0.93

Anomalous completeness                      55.7      90.9      30.5
Anomalous multiplicity                       1.9       2.4       2.3
DelAnom correlation between half-sets     -0.020    -0.499     0.138
Mid-Slope of Anom Normal Probability       0.978       -         -
"""

    template = os.path.basename(xdsme_directory).replace("xdsme_auto_", "")

    results = {}
    for line in example_table.split("\n"):
        key = line[: len("     Anomalous completeness (ellipsoidal))")].strip(" ")
        if key != "":
            results[key] = None

    tfile = os.path.join(xdsme_directory, f"{template}_{table1}")
    if os.path.isfile(tfile):
        table1 = open(tfile).read()
        # print(table1)
        for key in results:
            # pattern = f"     {key}[\ ]*([\d\.\-]*)[\ ]*([\d\.\-]*)[\ ]*([\d\.\-]*)"
            sk = key
            for c in special_characters:
                sk = sk.replace(c, f"\{c}")

            pattern = f".*{sk}.*"
            try:
                found = re.findall(pattern, table1)
                #print(f"{key} found {found}")
                numbers = found[0].replace(key, "").strip(" ").split()
                #print(f"numbers {numbers}")
                try:
                    r = list(map(float, numbers))
                except ValueError:
                    r = numbers
            except:
                r = None, None, None
                traceback.print_exc()
            results[key] = r

    results.update(get_xds_results(xdsme_directory))

    return results


def extract_stats(directory):
    if "main" in os.listdir(directory):
        directory = os.path.join(directory, "main")

    stats = {}

    for item in os.listdir(directory):
        if os.path.isdir(item):
            if "xdsme" in item:
                stats[item] = get_xdsme_results(item)
            elif "autoPROC" in item:
                stats[item] = get_results(item)

    return stats


def some_diffraction(evaluation_dir):
    return os.path.isdir(os.path.join(evaluation_dir, "char"))


def some_sample(evaluation_dir):
    return os.path.isdir(os.path.join(evaluation_dir, "tomo"))


def compare():
    base_directory = "/nfs/data4/2025_Run2/20250023/"
    manual = "2025-04-17"
    automa = "2025-04-20"

    to_look_at = os.path.join(base_directory, automa, "RAW_DATA")
    evaluations = subprocess.getoutput(
        f"find {to_look_at} -mindepth 2 -maxdepth 2 -type d"
    ).split("\n")
    evaluations = [item[len(to_look_at) + 1 :] for item in evaluations]

    ale = {}
    _start = time.time()
    for k, evaluation in enumerate(evaluations):
        print(f"Evaluating {evaluation} ({k+1} of {len(evaluations)})")
        manual_process_dir = os.path.join(
            base_directory, manual, "PROCESSED_DATA", evaluation
        )
        manual_raw_dir = os.path.join(
            base_directory, manual, "RAW_DATA", evaluation
        )
        auto_process_dir = os.path.join(
            base_directory, automa, "PROCESSED_DATA", evaluation, "main"
        )
        auto_raw_dir = os.path.join(
            base_directory, automa, "RAW_DATA", evaluation, "main"
        )

        print(f"{manual_process_dir}")
        print(f"{auto_process_dir}")
        print(f"{manual_raw_dir}")
        print(f"{auto_raw_dir}")
        
        template = get_template(evaluation)
        xdm = os.path.join(manual_process_dir, f"xdsme_auto_{template}_1")
        adm = os.path.join(manual_process_dir, f"autoPROC_{template}_1")
        xdad = os.path.join(
            auto_process_dir, f"xdsme_auto_{template}_default_strategy"
        )
        adad = os.path.join(auto_process_dir, f"autoPROC_{template}_default_strategy")
        xdab = os.path.join(auto_process_dir, f"xdsme_auto_{template}_BEST_strategy")
        adab = os.path.join(auto_process_dir, f"autoPROC_{template}_BEST_strategy")

        print(f"xdm {xdm}")
        print(f"adm {adm}")
        print(f"xdad {xdad}")
        print(f"adad {adad}")
        print(f"xdab {xdab}")
        print(f"adab {adab}")
        
        ale[evaluation] = {
            "manual_xdsme": get_xdsme_results(xdm),
            "manual_aP": get_autoproc_results(adm),
            "auto_default_xdsme": get_xdsme_results(xdad),
            "auto_default_aP": get_autoproc_results(adad),
            "auto_best_xdsme": get_xdsme_results(xdab),
            "auto_best_aP": get_autoproc_results(adab),
        }

        ale[evaluation]["sample_aligned"] = some_sample(os.path.dirname(auto_raw_dir))
        ale[evaluation]["some_diffraction"] = some_diffraction(os.path.dirname(auto_raw_dir))
        print()
        
    duration = time.time() - _start
    print(f"Extracting {len(evaluations)} collects took {duration:.4f} seconds")

    print(ale)
    
    f = open("/tmp/ale.pickle", "wb")
    pickle.dump(ale, f)
    f.close()

def get_table(ale):

    table = []
    for e in ale:
        sa = ale[e]["sample_aligned"]
        sd = ale[e]["some_diffraction"]
        auto_high_res = []
        auto_low_res = []
        for p in ["auto_default_xdsme", "auto_default_aP", "auto_best_xdsme", "auto_best_aP"]:
            try:
                hr = ale[e][p]["High resolution limit"][0]
                lr = ale[e][p]["Low resolution limit"][0]
            except:
                hr = np.inf
                lr = np.inf
            auto_high_res.append(hr)
            auto_low_res.append(lr)
        adx, ada, abx, aba = auto_high_res
        hr = min(auto_high_res)
        lr = auto_low_res[auto_high_res.index(hr)]
    
        manual_high_res = []
        manual_low_res = []
        
        for p in ["manual_xdsme", "manual_aP"]:
            try:
                mhr = ale[e][p]["High resolution limit"][0]
                mlr = ale[e][p]["Low resolution limit"][0]
            except:
                mhr = np.inf
                mlr = np.inf
            manual_high_res.append(mhr)
            manual_low_res.append(mlr)
        mhr = min(manual_high_res)
        mlr = manual_low_res[manual_high_res.index(mhr)]
        
        amhr = ale[e]["ALPX_mp"]
        if "No" in amhr:
            amhr = np.inf
        else:
            amhr = float(amhr)
        aahr = ale[e]["ALPX_wf"]
        if "No" in aahr:
            aahr = np.inf
        else:
            aahr = float(aahr)
            
        e = os.path.basename(e)
        
        table.append([e, int(sa), int(sd), mhr, hr, adx, ada, abx, aba, amhr, aahr])

    return table

def print_table(table):
    e = "sample_name"
    sa = "present"
    sd = "signal"
    mhr = "Manual"
    hr = "Auto"
    adx = "adx"
    ada = "ada"
    abx = "abx"
    aba = "aba"
    amhr = "Manual"
    aahr = "Auto"
    
    #print(f"{e.ljust(33)} {sa.rjust(8)} {sd.rjust(10)} {mhr.rjust(6)} {hr.rjust(6)} {adx.rjust(6)} {ada.rjust(6)} {abx.rjust(6)} {aba.rjust(6)}")
    
    #print(f"{e.ljust(33)} {sa.rjust(8)} {sd.rjust(10)} {mhr.rjust(6)} {hr.rjust(6)}")
    
    #print(f"{e.ljust(33)} {sa.rjust(8)} {sd.rjust(10)} {mhr.rjust(6)} {hr.rjust(6)} {amhr.rjust(6)} {aahr.rjust(6)}")
    
    print(f"{e.ljust(24)} {amhr.rjust(12)} {aahr.rjust(10)} {mhr.rjust(10)} {hr.rjust(10)}")
    print(f'{"(proc by ALPX)".rjust(46)} {"(proc by SOLEIL)".rjust(23)}')
    table.sort(key=lambda x: (x[0][x[0].rindex("_")+1:], x[0][x[0].index("-")+1:]))
    
    for line in table:
        e, sa, sd, mhr, hr, adx, ada, abx, aba, amhr, aahr = line
        sa = str(int(sa))
        sd = str(int(sd))
        mhr = f"{mhr:.1f}"
        hr = f"{hr:.1f}"
        #adx = f"{adx:.2f}"
        #ada = f"{ada:.2f}"
        #abx = f"{abx:.2f}"
        #aba = f"{aba:.2f}"
        amhr = f"{amhr:.1f}"
        aahr = f"{aahr:.1f}"

        e = e[e.index("-")+1:]
        
        #print(f"{e.ljust(33)} {sa.rjust(8)} {sd.rjust(10)} {mhr.rjust(6)} {hr.rjust(6)} {amhr.rjust(6)} {aahr.rjust(6)}")
        #print(f"{e.ljust(33)} {sa.rjust(8)} {sd.rjust(10)} {mhr.rjust(6)} {hr.rjust(6)} {amhr.rjust(6)} {aahr.rjust(6)}")
        print(f"{e.ljust(23)} {amhr.rjust(10)} {aahr.rjust(10)} {mhr.rjust(10)} {hr.rjust(10)}")

import pylab

def plot_results(table):

    samples = [line[0] for line in table]
    mhr = [line[3] for line in table]
    hr = [line[4] for line in table]
    adx = [line[5] for line in table] 
    ada = [line[6] for line in table]
    abx = [line[7] for line in table]
    aba = [line[8] for line in table]
    amp = [line[9] for line in table]
    awf = [line[10] for line in table]
    
    fig = pylab.figure(figsize=(16,9))
    pylab.title("Automated data collection at Proxima2A", fontsize=22)
    pylab.plot(range(0, len(samples), 1), amp, 'd', ms=11, label="Manual (proc by ALPX)")
    pylab.plot(range(0, len(samples), 1), awf, 'o', ms=11, label="Automated (proc by ALPX)")
    pylab.plot(range(0, len(samples), 1), mhr, 'd', ms=7, label="Manual (proc by SOLEIL)")
    pylab.plot(range(0, len(samples), 1), hr, 'o', ms=7, label="Automated (proc by SOLEIL)")
    
    
    pylab.ylim([1, 5])
    pylab.grid(True)
    
    pylab.xlabel("Sample unique identifier", fontsize=18)
    pylab.ylabel("High resolution limit [A]", fontsize=18)
    ax = pylab.gca()
    ax.set_xticks(range(0, len(samples), 1), samples, rotation=45, rotation_mode="anchor", ha="right")
    pylab.legend(loc=1)
    fig.set_tight_layout(True)
    pylab.show()
    
    
    
if __name__ == "__main__":
    compare()

    # import argparse

    # parser.add_argument("-d", "--directory", type=str, default="/nfs/data4/2025_Run2/20250023/2025-04-20/PROCESSED_DATA/PTPN22/PTPN22-CD044623_H10-3_BX033A-07/main")

    # args = parser.parse_args()

    # extract_stats(args.directory)

    # /autoPROC_PTPN22-CD044623_H10-3_BX033A-07_default_strategy/HDF5_1"
