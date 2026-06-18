#!/usr/bin/env python
# coding: utf-8 

import glob
import os
import subprocess
import numpy as np
from useful_routines import (
    get_result_position,
    get_pickled_file,
)

#https://www.w3schools.com/html/tryit.asp?filename=tryhtml_default
template = """
<!DOCTYPE html>
<html>
<head>
<title>Session Report, Proxima2A Synchrotron SOLEIL</title>
</head>

<body>
{sample_reports}
</body>
</html>
"""

sample_report="""
<h1>{sample_id}</h1>
<img src="{sample_snapshot_jpeg}" alt="{sample_description}">
<img src="{dozor_plot}" alt="number of spots per frame">
<h2>User clicks</h2>
{user_clicks}
{uclicks_vs_fit}
<h2>Expert clicks</h2>
{expert_clicks}
{eclicks_vs_fit}
<h2>Alignment summary</h2>
{summary_table}
"""

def _find(directory, template):
    found = subprocess.getoutput(
        f'find {directory} -iname "{template}"'
    ).split("\n")
    return found

def _unpickle_them(items):
    return [get_pickled_file(item) for item in items]

def get_collects(
    directory="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA",
    template="*_parameters.pickle"
):
    collects = _find(directory, template)
    return collects

def get_alignments(
    directory="/nfs/data4/2026_Run3/20260017/2026-06-11/ARCHIVE/opti",
    template="manu_*_parameters.pickle"
):
    alignments = _find(directory, template)
    return _unpickle_them(alignments)

def compare_positions(
    positions,
    keys=["Kappa", "Phi", "AlignmentX", "AlignmentY", "AlignmentZ", "CentringX", "CentringY"]
):
    for key in keys:
        for p in positions:
            if key not in p:
                p[key] = np.inf

        print(f"{key}, {[round(pos[key], 4) for pos in positions]} ")

def determine_alignment_for_collect(collect, alignments, debug=False):

    collect_pars = get_pickled_file(collect)
    t0 = collect_pars["timestamp"]
    relevant = [a for a in alignments if a["timestamp"] < t0]
    relevant.sort(key=lambda x: t0 - x["timestamp"])
    
    a = relevant[0]
    clicks_filename = "%s_clicks.pickle" % os.path.join(a["directory"], a["name_pattern"])
    c = get_pickled_file(clicks_filename)
    click_images = glob.glob(clicks_filename.replace("_clicks.pickle", "*.jpg"))
    if debug:
        print(f"{collect_pars['mounted_sample']} {collect_pars['name_pattern']}")
        print(f"{a['mounted_sample']} {a['name_pattern']}")
        print(f"time difference is {t0 - a['timestamp']:.3f}")
       
        rp = get_result_position(
            c["horizontal_displacements"],
            c["omegas"],
            c["reference_position"],
            along_displacements=c["vertical_discplacements"],
            filename=clicks_filename.replace("_clicks.pickle", "_clicks_fit.png"),
            title=a["name_pattern"],
        )

        compare_positions([collect_pars["position"], c["result_position"], rp[0]])
        print()
    return a, c, click_images



