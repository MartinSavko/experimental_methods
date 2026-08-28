#!/usr/bin/env python
# coding: utf-8 

import glob
import os
import subprocess
import datetime
import numpy as np

from useful_routines import (
    get_result_position,
    get_pickled_file,
    save_pickled_file,
    get_image_size,
    generate_thumbnails,
    timing,
)

from background_analysis import analyze_background

stp = get_pickled_file("/usr/local/experimental_methods/support_type_predictor.pickle")

# search webgl mesh example
# https://asalga.github.io/XB-PointStream
# https://learnwebgl.brown37.net #!!!!
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


# Include html document into another
#https://css-tricks.com/the-simplest-ways-to-handle-html-includes/
#https://www.filamentgroup.com/lab/html-includes/

"""
<iframe 
    src="header.html" 
    onload="this.before((this.contentDocument.body||this.contentDocument).children[0]);this.remove()">
</iframe>
"""

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

def get_head(title, favicon=None):
    head = "<head>\n"
    head += 1*"\t" + f"<title>{title}</title>\n"
    if favicon:
        head += 1*"\t" + f'<link rel="icon" type="image/x-icon" href="{favicon}">\n'
    head += "</head>\n"
    head += style
    head += script
    
    return head
    
def include_page(page):
    #_ = f'<div><iframe src="{page}" onload="this.before((this.contentDocument.body||this.contentDocument).children[0]);this.remove()"></iframe></div>\n'
    #_ = f'<iframe src="{page}" onload="this.insertAdjacentHTML(\'afterend\', (this.contentDocument.body||this.contentDocument).innerHTML);this.remove()"></iframe>\n'
                                                               
    #_ = f'<?php include "{page}" ?>\n'
    
    html = open(page).read()
    start = html.index("<body>") + len("<body>")
    end = html.index("</body>")
    _ = html[start: end]
    return _

def get_session_report_body(experiments, alignments, selfrely=False):
    srb = "<body>\n\n"
    for experiment in experiments:
        if selfrely:
            srb += get_experiment_report(experiment, alignments, standalone=True)
        else:
            srb += include_page(get_report_filename(experiment))
            
    srb += "</body>\n"
    return srb
    
def make_standalone(
    head_and_body,
    header = "<!DOCTYPE html>\n<html>\n",
    footer = "</html>\n",
):
    return header + head_and_body + footer

def get_session_report(
    directory="/nfs/data4/2026_Run3/20260017/2026-06-11",
    template="*_parameters.pickle",
    title="Session Report, Proxima2A Synchrotron SOLEIL",
    favicon=None,
    relative=True,
):
    raw = os.path.join(directory, "RAW_DATA")
    archive = os.path.join(directory, "ARCHIVE")
    experiments = get_collects(raw, template)
    alignments = get_alignments(os.path.join(archive, "opti"))
    
    head = get_head(title, favicon)
    body = get_session_report_body(experiments, alignments)

    if relative:
        body = body.replace(directory, ".")
        
    sr = head + body
    
    return make_standalone(sr)
       
def make_image_table(images, headers=[], items_per_row=3, _="", width=340, height=256):
    _ += "<table>\n"
    
    if headers:
        items_per_row = len(headers)
        _ += '\t<tr>\n'
        for th in headers:
            _ += 2 * '\t' + f"<th>{th}</th>\n"
        _ += '\t</tr>\n'
        
    for k, image in enumerate(images):
        if k % items_per_row == 0:
            _ += '\t<tr>\n'
            new_line = True
        else:
            new_line = False
        _ += 2*'\t' + "<td>\n"
        _ += 3*'\t' + f'<a href="{image}">\n'
        _ += 4*'\t' + f'<img src="{image}" style="width:{width}px;height:{height}px;">\n'
        _ += 3*'\t' + '</a>\n'
        _ += 2*'\t' + "</td>\n"
        if (k % items_per_row == 0 and not new_line) or k == len(images) - 1:
            _ += '\t</tr>\n'
         
    _ += "</table>\n"
    return _

def get_dozor_analysis(directory, name_pattern, _=""):
    
    dozor_directory = os.path.join(directory, "process", f"dozor_{name_pattern}")
    dozor_background_mtv = os.path.join(dozor_directory, "dozor_background.mtv")
    dozor_average_mtv = os.path.join(dozor_directory, "dozor_average.mtv")
    if not os.path.isfile(dozor_background_mtv):
        line = f"diffraction_experiment_analysis.py -d {directory} -n {name_pattern} -f"
        print(f"running DOZOR analysis {line}")
        os.system(line)
        
    about_background = analyze_background(dozor_background_mtv, **stp)
    _ += '<table>\n'
    _ += '<caption style="text-align:left"><b>Background analysis</b></caption>\n'
    for key in ["support type", "background maximum", "background lowres integral"]:
        if "background" in key:
            value = f'{about_background[key]:.3f}'
        else:
            value = about_background[key]
        _ += 1*"\t" + "<tr>\n"
        _ += 2*"\t" + f'<th>{key.replace("background", "").replace("integral", "integral (>7A)")}</th>\n'
        _ += 2*"\t" + f'<td>{value}</td>\n'
        _ += 1*"\t" + "</tr>\n"
    _ += "</table>\n"
    _ += "<br></br>\n"
    
    if not os.path.isfile(os.path.join(dozor_directory, "dozor_background_background_plot.png")):
        line = f'plotmtv.py -d {dozor_background_mtv}'
        os.system(line)
    if not os.path.isfile(os.path.join(dozor_directory, "dozor_average_spot_number_vs._rot.angle.png")):
        line = f'plotmtv.py -d {dozor_average_mtv}'
        os.system(line)
        
    images = [
        "dozor_background_background_plot.png",
        "dozor_background_background_vs.rotation_angle.png",
        "dozor_average_spot_number_vs._rot.angle.png",
        "dozor_average_b-factor_vs.rot.angle.png",
        "dozor_average_av.wilson_intensity_vs.rot.angle.png",
        "dozor_average_resolution_vs.rot.angle.png",
    ]
    
    images = [os.path.join(dozor_directory, image) for image in images]
    
    _ += make_image_table(images)
    return _

def get_processing(directory, name_pattern):
    pa = ""
    return pa

def get_parameters_table(
    parameters,
    selected_parameters=[
        [
            "directory",
            "name_pattern",
        ],
        [
            
            "photon_energy",
            "wavelength",
            "detector_distance",
            "resolution",
            "scan_range",
            "angle_per_frame",
            "scan_exposure_time",
            "frame_time",
            "scan_speed",
            "transmission",
            "timestamp",
        ],
    ],
    display_options={
        "transmission": {"unit": "%", "round": 1},
        "photon_energy": {"unit": "keV", "round": 4, "multiply": 1e-3},
        "wavelength": {"unit": "A", "round": 4},
        "detector_distance": {"unit": "mm", "round": 1},
        "resolution": {"unit": "A", "round": 3},
        "scan_range": {"name": "Total range", "unit": "deg.", "round": 4},
        "angle_per_frame": {"name": "Frame range", "unit": "deg", "round": 2},
        "scan_exposure_time": {"name": "Total exposure", "unit": "s", "round": 4},
        "frame_time": {"name": "Frame time", "unit": "s", "round": 4},
        "scan_speed": {"unit": "deg/s", "round": 1},
    },
    items_per_row=(1, 2),
    _="",
):
    
    for l, sp in enumerate(selected_parameters):
        _ += "<table>\n"
        for k, param in enumerate(sp):
            if k % items_per_row[l] == 0:
                _ += '\t<tr>\n'
                new_line = True
            else:
                new_line = False
                
            th = param.capitalize().replace("_", " ")
            unit = ""
            rnd = 4
            multiply = False
            if param in display_options:
                if param in display_options:
                    if "name" in display_options[param]:
                        th = display_options[param]["name"]
                    if "unit" in display_options[param]:
                        unit = f' {display_options[param]["unit"]}'
                    if "round" in display_options[param]:
                        rnd = display_options[param]["round"]
                    if "multiply" in display_options[param]:
                        multiply = display_options[param]["multiply"]
            
            value = parameters[param]
            
            if multiply:
                value *= multiply
            if param == "timestamp":
                value = datetime.datetime.isoformat(datetime.datetime.fromtimestamp(value))
            if param == "directory":
                value = f'<a href="{value}">{value}</a>'
            if isinstance(value, float):
                td = f'{round(value, rnd)}'
            else:
                td = value
            
            _ += 2*'\t' + f'<th>{th}</th> <td>{td}{unit}</td>\n'
            if (k % items_per_row[l] == 0 and not new_line) or k == len(sp) - 1:
                _ += '\t</tr>\n'
         
        _ += "</table>\n"

    return _
        
        
def get_experiment_report(experiment, alignments, standalone=True, debug=False, er=""):
    a, c, click_images, collect_pars, rp = determine_alignment_for_collect(experiment, alignments, debug=debug)
    
    directory, name_pattern = collect_pars["directory"], collect_pars["name_pattern"] 
    template = os.path.join(directory, name_pattern).replace("RAW_DATA", "ARCHIVE")
    
    er += f'<h1>{name_pattern}</h1>\n'
    er += '<h2>Overview</h2>\n'
    er += get_visit_card(directory, name_pattern)
    
    er += get_parameters_table(collect_pars)
    
    er += '<h2>DOZOR analysis overview</h2>\n'
    er += get_dozor_analysis(directory, name_pattern)
    
    er += '<h2>Processing</h2>\n'
    er += get_processing(directory, name_pattern)
    
    er += '<h2>Alignment</h2>\n'
    er += get_alignment_overview(a, c, click_images, collect_pars, rp)
    er += 5 * '\n'
        
    if standalone:
        head = get_head(title=f"{collect_pars['name_pattern']} report")
        body = "<body>\n" + er + "</body>\n"
        er = make_standalone(head + body)
        
        report_filename = get_report_filename(experiment)
        print(f"saving report to {report_filename}")
        f = open(report_filename, "w")
        f.write(er)
        f.close()
    
    if debug:
        print(er)
        
    return er

def get_visit_card(directory, name_pattern):
    
    template = os.path.join(directory, name_pattern).replace("RAW_DATA", "ARCHIVE")
    
    sample_snapshot_jpeg = f"{template}_1.snapshot.jpeg"
    diffraction_thumbnail = f"{template}_000001.jpeg"
    dozor_plot = f"{template}.png"
    
    if not os.path.isfile(diffraction_thumbnail):
        print("thumbnails do not exist, will try to generate them")
        generate_thumbnails(directory, name_pattern)
        
    if not os.path.isfile(dozor_plot):
        print("dozor plot is not present, will try to generate it")
        line = f"diffraction_experiment_analysis.py -d {directory} -n {name_pattern} &"
        print(line)
        os.system(line)
    
    images = [
        sample_snapshot_jpeg,
        diffraction_thumbnail,
        dozor_plot,
    ]

    headers = [
        "optical snapshot",
        "diffraction",
        "dozor plot",
    ]
    
    vc = make_image_table(images, headers)
    
    #vc = '<table>\n'
    #vc += '\t<tr>\n'
    #vc += 2*'\t' + "<th>optical snapshot</th>\n"
    #vc += 2*'\t' + "<th>diffraction</th>\n"
    #vc += 2*'\t' + "<th>dozor plot</th>\n"
    #vc += '\t</tr>\n'
    #vc += '\t<tr>\n'
    #vc += 2*'\t' + "<td>\n"
    #vc += 3*'\t' + f'<img src="{sample_snapshot_jpeg}" alt="sample optical image just before the collect" style="width:340px;height:340px;">\n'
    #vc += 2*'\t' + "</td>\n"
    #vc += 2*'\t' + "<td>\n"
    #vc += 3*'\t' + f'<img src="{diffraction_thumbnail}" alt="diffraction image" style="width:340px;height:340px;">\n'
    #vc += 2*'\t' + "</td>\n"
    #vc += 2*'\t' + "<td>\n"
    #vc += 3*'\t' + f'<img src="{dozor_plot}" alt="number of spots per frame" style="width:340px;height:340px;">\n'
    #vc += 2*'\t' + "</td>\n"
    #vc += '\t</tr>\n'
    #vc += '</table>\n'
    
    return vc

def get_alignment_overview(a, c, click_images, collect_pars, rp):
    
    if a is not None:
        ao = f'<h3>alignment movie</h3>\n'
        ao += get_alignment_video(a)
        ao += f'<h3>alignment clicks</h3>\n'
        ao += f'<br>number of clicks: {len(click_images)}</br>\n'
        ao += get_click_image_table_with_overlays(click_images, c)
        
        clicks_fit_figure = f'{os.path.join(a["directory"], a["name_pattern"])}_clicks_fit.png'
        ao += f'<img src="{clicks_fit_figure}" alt="clicks fit" style="width:1120px;height:630px;">\n'
        positions = [
            ("reference", c["reference_position"]),
            ("align", c["result_position"]),
            ("collect", collect_pars["position"]),
            ("align2", rp[0]),
        ]
        
        ao += get_positions_table(positions)
    else:
        ao = ""
    return ao

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

def _get_video_item(src, width, height, _type="video/mp4"):
    av = f'<video width="{width}" height="{height}" controls>\n'
    av += f'\t<source src="{src}" type="{_type}">\n'
    av += "\tYour browser does not support the video tag.\n"
    av += "</video>\n"
    return av

def get_alignment_video(a, width=1360//3, height=1024//3, generate=True):
    movie = f'{os.path.join(a["directory"], a["name_pattern"])}_sample_view_movie.mp4'
    murko = movie.replace("_sample_view_movie.mp4", "_murko_movie.webm")
    if generate and not os.path.isfile(murko):
        line = f"murko_movie.py -e {movie} &"
        print("murko movie is not present, will try to generate it ")
        print(line)
        os.system(line)
    oav_element = _get_video_item(movie, width, height)
    murko_element = _get_video_item(murko, width, height, _type="video/webm")
    av = "<table>\n"
    av += "\t<tr>\n"
    av += 2*"\t" + f'<th><a href="{movie}">oav</a></th>\n'
    av += 2*"\t" + f'<th><a href="{murko}">murko</a></th>\n'
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

def get_click_image_table(click_images, items_per_row=3):
    cit = "<table>\n"
    nrows = len(click_images) // items_per_row
    k = 0
    for row in range(nrows):
        cit += "\t<tr>\n"
        for l in range(items_per_row):
            cit += 2*"\t" + "<td>\n"
            cit += 3*"\t" + f'<img src="{click_images[k]}" alt="user click {k}" style="width:340px;height:256px;">\n'
            cit += 2*"\t" + "</td>\n"
            k += 1
        cit += "\t</tr>\n"
    cit += "</table>\n"
    return cit

import re
click = re.compile(".*_y_([\d]+)_x_([\d]+).*")

def get_click_image_table_with_overlays(click_images, c, items_per_row=3, click_diameter=5, scale=0.25):
    cit = '<div style="display:none;">\n'
    for k, imagepath in enumerate(click_images):
        #ih, iw = get_image_size(imagepath)
        #w = int(iw*scale)
        #h = int(ih*scale)
        cit += f'\t<a href="{imagepath}"><img id="image_{os.path.basename(imagepath)}" src="{imagepath}"></a>\n'
        #style="width:{w}px;height:{h}px;>\n'
    cit += '</div>\n'
    
    cit += "<table>\n"
    nrows, reminder = divmod(len(click_images), items_per_row)
    if reminder > 0:
        nrows += 1
    k = 0
    for row in range(nrows):
        cit += "\t<tr>\n"
        for l in range(items_per_row):
            cit += 2*"\t" + "<td>\n"
            cit += 3*"\t" 
            try:
                imagepath = click_images[k]
            except:
                break
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

@timing
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
    if relevant:
        a = relevant[0]
    
        clicks_filename = "%s_clicks.pickle" % os.path.join(a["directory"], a["name_pattern"])
        c = get_pickled_file(clicks_filename)
        click_images = glob.glob(clicks_filename.replace("_clicks.pickle", "*.jpg"))
        click_images.sort(key=lambda x: x[x.index("click"):])
        print(f"{collect_pars['mounted_sample']} {collect_pars['name_pattern']}")
        print(f"{a['mounted_sample']} {a['name_pattern']}")
        print(f"time difference is {t0 - a['timestamp']:.3f}")
        used = c["orthogonal_optimal_parameters"]
        if isinstance(used, dict):
            pass
        else:
            center, radius, phase = used
            used = {"c": center, "r": radius, "alpha": phase}
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

    else:
        a = None
        c = None
        click_images = []
        rp = None
        
    return a, c, click_images, collect_pars, rp

def get_report_filename(experiment):
    
    report_filename = os.path.realpath(experiment).replace("_parameters.pickle", "_report.html").replace("RAW_DATA", "ARCHIVE")
    
    if not os.path.isdir(os.path.dirname(report_filename)):
        report_filename = os.path.realpath(experiment).replace("_parameters.pickle", "_report.html")
        
    return report_filename

def _experiment_report(
    #experiment="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA/IRF5/IRF5-MT260973_H01-1_BX028A-02/IRF5-MT260973_H01-1_BX028A-02_1_parameters.pickle",
    experiment="/nfs/data4/2026_Run3/20100023/2026-06-14/RAW_DATA/Manual/8_15_13_parameters.pickle",
):
    experiment = os.path.realpath(experiment)
    directory = experiment[:experiment.index("/RAW_DATA")]
    archive = os.path.join(directory, "ARCHIVE")
    alignments = get_alignments(os.path.join(archive, "opti"))
    
    er = get_experiment_report(experiment, alignments)
    
    #print(er)
    

def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--session",
        type=str,
        #default="/nfs/data4/2026_Run3/20260017/2026-06-11",
        default=None,
        help="session",
    )

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA/IRF5/IRF5-MT260973_H01-1_BX028A-02/IRF5-MT260973_H01-1_BX028A-02_1_parameters.pickle",
        #default="/nfs/data4/2026_Run3/20100023/2026-06-14/RAW_DATA/Manual/8_15_13_parameters.pickle"
        help="experiment",
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

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force",
    )
    
    args = parser.parse_args()
    print(f"args {args}")

    if args.session is not None:
        sr = get_session_report(
            args.session,
            args.template,
        )

        f = open(os.path.join(args.session, "session_report.html"), "w")
        f.write(sr)
        f.close()
    else:
        report_filename = get_report_filename(args.experiment)
        if os.path.isfile(report_filename) and not args.force:
            print(f"report {report_filename} already present, moving on ...")
        else:
            print(f"report {report_filename} not present or force")
            _experiment_report(args.experiment)

if __name__ == "__main__":
    main()
    #test()
    
