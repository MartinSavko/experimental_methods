#!/usr/bin/env python
# coding: utf-8

import os
import pylab
import seaborn as sns
sns.set(color_codes=True)
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

def get_curves(mtv_file="dozor_background.mtv"):
    dbm = open(mtv_file, "r").readlines()
    curves = []
    curve = None
    while dbm:
        line = dbm.pop(0)
        if line.startswith("$"):
            if curve is not None:
                curves.append(curve)
            curve = {"x": [], "y": [], "comment": ""}
            curve["type"] = line.split("=")[-1].strip("\n")
        elif line.startswith("#"):
            c = f"{line[1:].strip()}"
            curve["comment"] += c if not curve["comment"] else f"\n{c}"
        elif line.startswith("%"):
            ls = line.strip("% ")
            s = ls.find("=")
            key = ls[:s].strip()
            value = ls[s+1:].strip()
            curve[key.strip()] = value.strip("' \n")
        else:
            x, y = line.split()
            curve["x"].append(float(x))
            curve["y"].append(float(y))
    else:
        if curve is not None:
           curves.append(curve)
           
    return curves

def plot_curve(curve, datafile=None, directory="./", figsize=(16, 9), fontsize=22):
    fig = pylab.figure(figsize=figsize)

    fig.suptitle(curve["toplabel"], fontsize=fontsize)
    if "subtitle" in curve:
        pylab.title(curve["subtitle"], fontsize=fontsize)
    
    pylab.plot(curve["x"], curve["y"], lw=float(curve["linewidth"]), label=curve["linelabel"] if "linelabel" in curve else "data")
    pylab.xlabel(curve["xlabel"], fontsize=int(0.95*fontsize))
    pylab.ylabel(curve["ylabel"], fontsize=int(0.95*fontsize))
    #pylab.set_xmin(float(curve["xmin"]))
    pylab.legend(fontsize=int(0.9*fontsize))
    
    if datafile is not None:
        directory = os.path.dirname(datafile)
        fname = os.path.join(directory, os.path.basename(datafile).strip(".mtv") + "_" + curve["toplabel"].lower().replace(" ", "_") + ".png")
    pylab.savefig(fname)
    
def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--datafile",
        type=str,
        default="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA/IRF5/IRF5-MT260973_H06-2_BX028A-03/process/dozor_IRF5-MT260973_H06-2_BX028A-03_1/dozor_background.mtv",
        help="mtv file",
    )

    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="display the plots",
    )
    
    args = parser.parse_args()
    print(f"args {args}")
    
    curves = get_curves(args.datafile)
    for curve in curves:
        plot_curve(curve, datafile=args.datafile)
        
    if args.show:
        pylab.show()
        
if __name__ == "__main__":
    main()
    
