#!/usr/bin/env python

import os
import subprocess

def get_pids(template):
    ps = subprocess.getoutput(
        f"ps aux | grep -i {template} | grep -v grep | grep -v skill"
    )
    
    if ps:
        print(ps)
        ps = ps.split("\n")
        pids = [int(item.split()[1]) for item in ps]
    else:
        pids = []
        print(f"No {template} processes found ")
        
    return pids

def kill(pid, signal):
    print(f"sending {signal:d} to {pid:d}")
    try:
        os.kill(pid, signal)
    except:
        print(f"could not kill {pid}, please check")
        
def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("-t", "--template", type=str, default="baloo", help="template of processes to stop")
    
    parser.add_argument("-s", "--signal", type=int, default=15, help="signal to send")
    
    args = parser.parse_args()
    print(args)
    
    pids = get_pids(args.template)
    
    for pid in pids:
        kill(pid, args.signal)
        
if __name__ == "__main__":
    main()
