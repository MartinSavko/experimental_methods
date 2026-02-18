#!/usr/bin/env python
import os
import glob


def rename(source):
    directory = os.path.dirname(source)
    basename = os.path.basename(source)
    template = basename.replace("_master.h5", "")

    desttemplate = directory.split("/")[-2] + "_BEST_strategy"

    dest = "%s_master.h5" % desttemplate
    # h5 rename
    hrl = "cd {directory}; h5rename {basename} {dest}".format(
        directory=directory, basename=basename, dest=dest
    )
    os.system(hrl)
    os.chdir(directory)
    allfiles = glob.glob("%s*" % template) + glob.glob("*/%s*" % template)

    for f in allfiles:
        os.rename(f, f.replace(template, desttemplate))

    lnp = "sed -i \"s/.*name_pattern.*/ 'name_pattern': '{desttemplate}',/g\" {desttemplate}.log".format(
        desttemplate=desttemplate
    )
    print(lnp)
    os.system(lnp)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument(
        "-s",
        "--source",
        default="/nfs/data4/2025_Run2/20250023/2025-04-20/RAW_DATA/PTPN22/PTPN22-CD044622_H09-2_BX029A-02/main/omega_scan_best_1_pos_02__master.h5",
        type=str,
        help="source",
    )

    args = parser.parse_args()
    rename(args.source)
