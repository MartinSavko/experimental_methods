#!/usr/bin/env python
# coding: utf-8 

import glob
import os
import subprocess
import numpy as np
from useful_routines import (
    get_result_position,
    get_pickled_file,
    save_pickled_file,
    get_image_size,
)

# search webgl mesh example
# https://imagine.inrialpes.fr/people/Francois.Faure/htmlCourses/WebGL/IntroMeshes.html
# https://imagine.inrialpes.fr/people/Francois.Faure/htmlCourses/WebGL/meshes/cube8.html
# https://team.inria.fr/imagine/gallery/
# https://imagine.inrialpes.fr/people/Francois.Faure/htmlCourses/index.html
# https://imagine.inrialpes.fr/people/Francois.Faure/htmlCourses/FiniteElements.html
# https://graphics.stanford.edu/data/3Dscanrep/
# https://www.webglacademy.com/courses.php?courses=0_1_20_2_3_4_23_5_6_7_10#6
# https://www.w3schools.com/graphics/webgl_intro.asp

# webgl point cloud viewer
# https://sites.icmc.usp.br/fosorio/webgl/webgl-data.html

# https://web.dev/articles/webgl-fundamentals
# https://www.youtube.com/embed/H4c8t6myAWU/?feature=player_detailpage
# VTK js
# https://vimeo.com/375520781
# https://github.com/Kitware/vtk-js
# https://kitware.github.io/vtk-js/docs/tutorial.html

# https://www.w3schools.com/html/tryit.asp?filename=tryhtml_default
# https://stackoverflow.com/questions/13903257/html5-canvas-scale-image-after-drawing-it

script = """
<script type="text/javascript">

function draw_image_and_click(canvas_id, image_id, x, y, click_diameter=5, click_color="blue") {
    const canvas = document.getElementById(canvas_id);
    const ctx = canvas.getContext("2d");
    
    const image = document.getElementById(image_id);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    ctx.beginPath();
    ctx.arc(x, y, click_diameter, 0, 2 * Math.PI);
    ctx.fillStyle = click_color;
    ctx.fill();
};

</script>
    
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
    directory="/nfs/data4/2026_Run3/20260017/2026-06-11",
    template="*_parameters.pickle",
):
    raw = os.path.join(directory, "RAW_DATA")
    archive = os.path.join(directory, "ARCHIVE")
    experiments = get_collects(raw, template)
    alignments = get_alignments(os.path.join(archive, "opti"))
    
    sr = "<!DOCTYPE html>\n"
    sr += "<html>\n"
    sr += 1*"\t" + "<head>\n"
    sr += 2*"\t" + "<title>Session Report, Proxima2A Synchrotron SOLEIL</title>\n"
    sr += 1*"\t" + "</head>\n"
    sr += style
    sr += script
    sr += "<body>\n\n"
    for experiment in experiments:
        sr += get_experiment_report(experiment, alignments)
    sr += "</body>\n"
    sr += "</html>\n"
    
    return sr
                                
def get_experiment_report(experiment, alignments, debug=False):
    a, c, click_images, collect_pars, rp = determine_alignment_for_collect(experiment, alignments, debug=debug)
    
    t = os.path.join(collect_pars["directory"], collect_pars["name_pattern"]).replace("RAW_DATA", "ARCHIVE")
    
    sample_snapshot_jpeg = f"{t}_1.snapshot.jpeg"
    diffraction_thubnail = f"{t}_000001.jpeg"
    dozor_plot = f"{t}.png"
    
    er = f'<h1>{os.path.basename(experiment).replace("_parameters.pickle", "")}</h1>\n'
    
    #er += '<h2>Snapshot, thumbnail and DOZOR plot</h2>\n'
    er += '<table>\n'
    er += '\t<tr>\n'
    er += 2*'\t' + "<th>optical snapshot</th>\n"
    er += 2*'\t' + "<th>diffraction</th>\n"
    er += 2*'\t' + "<th>dozor plot</th>\n"
    er += '\t</tr>\n'
    er += '\t<tr>\n'
    er += 2*'\t' + "<td>\n"
    er += 3*'\t' + f'<img src="{sample_snapshot_jpeg}" alt="sample optical image just before the collect" style="width:340px;height:340px;">\n'
    er += 2*'\t' + "</td>\n"
    er += 2*'\t' + "<td>\n"
    er += 3*'\t' + f'<img src="{diffraction_thubnail}" alt="diffraction image" style="width:340px;height:340px;">\n'
    er += 2*'\t' + "</td>\n"
    er += 2*'\t' + "<td>\n"
    er += 3*'\t' + f'<img src="{dozor_plot}" alt="number of spots per frame" style="width:340px;height:340px;">\n'
    er += 2*'\t' + "</td>\n"
    er += '\t</tr>\n'
    er += '</table>\n'
    
    #er += f'<h2>Sample alignment</h2>\n'
    er += f'<h3>alignment movie</h3>\n'
    er += get_alignment_video(a)
    er += f'<h3>alignment clicks</h3>\n'
    #er += get_click_image_table(click_images)
    er += get_click_image_table_with_overlays(click_images, c)
    
    #er += get_click_fits()
    clicks_fit_figure = f'{os.path.join(a["directory"], a["name_pattern"])}_clicks_fit.png'
    er += f'<img src="{clicks_fit_figure}" alt="clicks fit" style="width:1120px;height:630px;">\n'
    positions = [
        ("reference", c["reference_position"]),
        ("align", c["result_position"]),
        ("collect", collect_pars["position"]),
        ("align2", rp[0]),
    ]
    
    #er += f'<h2>Aligned positions</h2>\n'
    er += get_positions_table(positions)
    er += 5 * '\n'
    if debug:
        print(er)
    return er

def get_positions_table(positions, keys=["AlignmentY", "AlignmentZ", "CentringX", "CentringY", "Kappa", "Phi"]):
    pt = "<table>\n"
    pt += "\t<caption>Aligned positions</caption>\n"
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

def _get_video_item(src, width, height):
    av = f'<video width="{width}" height="{height}" controls>\n'
    av += f'\t<source src="{src}" type="video/mp4">\n'
    av += "\tYour browser does not support the video tag.\n"
    av += "</video>\n"
    
def get_alignment_video(a, width=1360//3, height=1024//3):
    movie = f'{os.path.join(a["directory"], a["name_pattern"])}_sample_view_movie.mp4'
    murko = movie.replace("_sample_view_movie.mp4", "_murko_movie.mp4")
    oav_element = _get_video_item(movie, width, height)
    murko_element = _get_video_item(murko, width, height)
    av = "<table>\n"
    av += "\t<tr>\n"
    av += 2*"\t" + f'<th>oav</th>\n'
    av += 2*"\t" + f'<th>murko</th>\n'
    av += "\t</tr>\n"
    av += "\t<tr>\n"
    av += 2*"\t" + f"<td>\n"
    av += oav_element
    av += 2*"\t" + f"</td>\n"
    av += 2*"\t" + f"<td>\n"
    av += murko_element
    av += 2*"\t" + f"</td>\n"
    av += "\t</tr>\n"
    av += "</table>\n"
    return av 

def get_click_image_table(click_images, images_per_row=3):
    cit = "<table>\n"
    nrows = len(click_images) // images_per_row
    k = 0
    for row in range(nrows):
        cit += "\t<tr>\n"
        for l in range(images_per_row):
            cit += 2*"\t" + "<td>\n"
            cit += 3*"\t" + f'<img src="{click_images[k]}" alt="user click {k}" style="width:340px;height:256px;">\n'
            cit += 2*"\t" + "</td>\n"
            k += 1
        cit += "\t</tr>\n"
    cit += "</table>\n"
    return cit

import re
click = re.compile(".*_y_([\d]+)_x_([\d]+).*")

def get_click_image_table_with_overlays(click_images, c, images_per_row=3, click_diameter=5, scale=0.25):
    cit = '<div style="display:none;">\n'
    for k, imagepath in enumerate(click_images):
        #ih, iw = get_image_size(imagepath)
        #w = int(iw*scale)
        #h = int(ih*scale)
        cit += f'\t<img id="image_{os.path.basename(imagepath)}" src="{imagepath}">\n'
        #style="width:{w}px;height:{h}px;>\n'
    cit += '</div>\n'
    
    cit += "<table>\n"
    nrows = len(click_images) // images_per_row
    k = 0
    for row in range(nrows):
        cit += "\t<tr>\n"
        for l in range(images_per_row):
            cit += 2*"\t" + "<td>\n"
            cit += 3*"\t" 
            imagepath = click_images[k]
            ih, iw = get_image_size(imagepath)
            iname = os.path.basename(imagepath)
            canvas_id = f"canvas_{iname}"
            cit += f'<canvas id="{canvas_id}" width="{int(iw*scale)}" height="{int(ih*scale)}"></canvas>\n'
            cit += 2*"\t" + "</td>\n"
            k += 1
        cit += "\t</tr>\n"
    cit += "</table>\n"
    
    #https://stackoverflow.com/questions/19869639/how-to-call-a-javascript-function-within-an-html-body
    
    for k, img in enumerate(click_images):
        iname = os.path.basename(img)
        canvas_id = f"canvas_{iname}"
        image_id = f"image_{iname}"
        y, x = (np.array(click.findall(iname)[0]).astype(float) * scale).astype(int)
        cit += "<script>\n"
        cit += f'\tdraw_image_and_click("{canvas_id}", "{image_id}", {x}, {y}, {click_diameter});\n'
        cit += "</script>\n"
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

def determine_alignment_for_collect(collect, alignments, debug=False, force=False):

    collect_pars = get_pickled_file(collect)
    t0 = collect_pars["timestamp"]
    relevant = [a for a in alignments if a["timestamp"] < t0]
    relevant.sort(key=lambda x: t0 - x["timestamp"])
    
    a = relevant[0]
    clicks_filename = "%s_clicks.pickle" % os.path.join(a["directory"], a["name_pattern"])
    c = get_pickled_file(clicks_filename)
    click_images = glob.glob(clicks_filename.replace("_clicks.pickle", "*.jpg"))
    click_images.sort(key=lambda x: x[x.index("click"):])
    print(f"{collect_pars['mounted_sample']} {collect_pars['name_pattern']}")
    print(f"{a['mounted_sample']} {a['name_pattern']}")
    print(f"time difference is {t0 - a['timestamp']:.3f}")
    used = c["orthogonal_optimal_parameters"]
    rp_filename = clicks_filename.replace("_clicks.pickle", "_result_position.pickle")
    if force or not os.path.isfile(rp_filename):
        rp = get_result_position(
            c["horizontal_displacements"],
            c["omegas"],
            c["reference_position"],
            alignmenty_direction=1.0,
            alignmentz_direction=-1.0,
            centringx_direction=-1.0,
            centringy_direction=-1.0,
            click_label="click",
            along_displacements=c["vertical_discplacements"],
            filename=clicks_filename.replace("_clicks.pickle", "_clicks_fit.png"),
            title=a["name_pattern"],
            comparative_model=(used["c"], used["r"], used["alpha"])
        )
        save_pickled_file(rp_filename, rp)
    else:
        rp = get_pickled_file(rp_filename)
    if debug:
        compare_positions([collect_pars["position"], c["result_position"], rp[0]])

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
        default="/nfs/data4/2026_Run3/20260017/2026-06-11",
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
