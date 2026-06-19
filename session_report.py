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
<h2>Optical alignment</h2>
{user_clicks}
{uclicks_vs_fit}
<h2>Expert clicks</h2>
{expert_clicks}
{eclicks_vs_fit}
<h2>Alignment summary</h2>
{summary_table}
"""

summary_table = """
<table>
    <tr>
        <th></th>
        <th>reference</start>
        <th>method1 (used during experiment)</th>
        <th>method2</>
        <th>refractive (user clicks + priors)</th>
        <th>refractive (expert clicks + priors)</th>
    </tr>
    <tr>
        <th>AlignmentY</th>
        <td>{reference_ay:.4f}</td>
        <td>{aligned_ay:.4f}</td>
        <td>{better_ay:.4f}</td>
        <td>{refractive_ay:.4f}</td>
        <td>{refractive_better_ay:.4f}</td>
    </tr>
    <tr>
        <th>AlignmentZ</th>
        <td>{reference_az:.4f}</td>
        <td>{aligned_az:.4f}</td>
        <td>{better_az:.4f}</td>
        <td>{refractive_az:.4f}</td>
        <td>{refractive_better_az:.4f}</td>
    </tr>
    <tr>
        <th>CentringX</th>
        <td>{reference_cx:.4f}</td>
        <td>{aligned_cx:.4f}</td>
        <td>{better_cx:.4f}</td>
        <td>{refractive_cx:.4f}</td>
        <td>{refractive_better_cx:.4f}</td>
    </tr>
    <tr>
        <th>CentringY</th>
        <td>{reference_cy:.4f}</td>
        <td>{aligned_cy:.4f}</td>
        <td>{better_cy:.4f}</td>
        <td>{refractive_cy:.4f}</td>
        <td>{refractive_better_cy:.4f}</td>
    </tr>
</table>
"""

style="""
<style>
table {
  border-collapse: collapse;
}

th {
  text-align: center;
  padding: 8px;
}

td {
  text-align: left;
  padding: 8px;
}

tr:nth-child(odd) {
  background-color: #D6EEEE;
}
</style>
"""

def get_session_report(
    directory="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA",
    template="*_parameters.pickle",
):
    experiments = get_collects(directory, template)
    alignments = get_alignments(directory.replace("RAW_DATA", "ARCHIVE/opti"))
    
    sr = "<!DOCTYPE html>\n"
    sr += "<html>\n<head>\n<title>Session Report, Proxima2A Synchrotron SOLEIL</title>\n</head>\n"
    sr += style
    sr += "<body>\n"
    for experiment in experiments:
        sr += get_experiment_report(experiment, alignments)
    sr += "</body>\n</html>\n"
    
    return sr
                                
def get_experiment_report(experiment, alignments, debug=True):
    a, c, click_images, collect_pars, rp = determine_alignment_for_collect(experiment, alignments, debug=True)
    
    t = os.path.join(collect_pars["directory"], collect_pars["name_pattern"]).replace("RAW_DATA", "ARCHIVE")
    
    sample_snapshot_jpeg = f"{t}_1.snapshot.jpeg"
    diffraction_thubnail = f"{t}_000001.jpeg"
    dozor_plot = f"{t}.png"
    
    er = f'<h1>{os.path.basename(experiment).replace("_parameters.pickle", "")}</h1>\n'
    
    er += '<h2>Snapshot, thumbnail and DOZOR plot</h2>\n'
    er += '<table>\n'
    er += '\t<tr>\n'
    er += 2*'\t' + "<th>optical snapshot</th>\n"
    er += 2*'\t' + "<th>diffraction</th>\n"
    er += 2*'\t' + "<th>dozor plot</th>\n"
    er += '\t</tr>\n'
    er += '\t<tr>\n'
    er += 2*'\t' + "<td>\n"
    er += 3*'\t' + f'<img src="{sample_snapshot_jpeg}" alt="sample optical image just before the collect" style="width:320px;height:320px;">\n'
    er += 2*'\t' + "</td\n>"
    er += 2*'\t' + "<td>\n"
    er += 3*'\t' + f'<img src="{diffraction_thubnail}" alt="diffraction image" style="width:320px;height:320px;">\n'
    er += 2*'\t' + "</td>\n"
    er += 2*'\t' + "<td>\n"
    er += 3*'\t' + f'<img src="{dozor_plot}" alt="number of spots per frame" style="width:320px;height:320px;">\n'
    er += 2*'\t' + "</td>\n"
    er += '\t</tr>\n'
    er += '</table>\n'
    
    er += f'<h2>Sample alignment</h2>\n'
    er += get_click_image_table(click_images)
    
    positions = [
        ("reference", c["reference_position"]),
        ("align", c["result_position"]),
        ("collect", collect_pars["position"]),
        ("align2", rp[0]),
    ]
                 
    er += f'<h2>Aligned positions</h2>\n'
    er += get_positions_table(positions)
    er += 5 * '\n'
    if debug:
        print(er)
    return er

def get_positions_table(positions, keys=["AlignmentY", "AlignmentZ", "CentringX", "CentringY", "Kappa", "Phi"]):
    pt = "<table>\n"
    pt += "\t<tr>\n"
    pt += 2*"\t" + "<th></th>\n"
    for name, position in positions:
        pt += 2*"\t" + f'<th>{name}</th>\n'
    pt += "\t</tr>\n"
    for key in keys:
        pt += "\t<tr>\n"
        pt += 2*"\t" + f'<th>{key}</th>\n'
        for name, position in positions:
            pt += 2*"\t" + f'<td>{position[key]:.4f}</td>\n'
        pt += "\t</tr>\n"
    pt += "</table>\n"

    return pt

def get_click_image_table(click_images, images_per_row=3):
    cit = "<table>\n"
    nrows = len(click_images) // images_per_row
    k = 0
    for row in range(nrows):
        cit += "\t<tr>\n"
        for l in range(images_per_row):
            cit += 2*"\t" + "<td>\n"
            cit += 3*"\t" + f'<img src="{click_images[k]}" alt="user click {k}" style="width:320px;height:320px;">\n'
            cit += 2*"\t" + "</td>\n"
            k += 1
        cit += "\t</tr>\n"
    cit += "</table>\n"
    return cit

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
    return a, c, click_images, collect_pars, rp




def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA",
        help="directory",
    )

    parser.add_argument(
        "-t",
        "--template",
        type=str,
        default="*_parameters.pickle",
        help="template",
    )
    parser.add_argument(
        "-D", "--display", action="store_true", help="display analysis"
    )

    args = parser.parse_args()
    print(f"args {args}")

    sr = get_session_report(
        args.directory,
        args.template,
    )

    f = open(os.path.join(args.directory, "session_report.html"), "w")
    f.write(sr)
    f.close()
             

if __name__ == "__main__":
    main()
