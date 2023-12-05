#!/usr/bin/env python

import sys
import time
import os
import subprocess
import re
import pickle
import HTMLParser
import traceback
import logging
import gevent
from datetime import date
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from subprocess import Popen, PIPE

sys.path.insert(0, '/nfs/data/experimental_methods')
import eiger

# essential variables
eiger_IP = '172.19.10.26'
downloader_log_directory = '/nfs/data2/downloader'
time_out = 10.
check_time = 1.
acceptable_apparent_file_size_difference = 209715.2 # bytes: 0.2 * 1024 * 1024
base_url = 'http://{eiger_IP:s}/data/'.format(eiger_IP=eiger_IP)
default_destination = '/nfs/data2/orphaned_collects'
fix_master_executable = './fix_master.py'


COLOR_W = '\033[93m'
COLOR_C = '\033[91m'
COLOR_E = '\033[91m'
END_COLOR = '\033[0m'

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
cslFormatter = logging.Formatter("%(message)s")
prntLog = logging.getLogger()
prntLog.setLevel(INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(cslFormatter)
prntLog.addHandler(consoleHandler)
fileHandler = None

def get_name_pattern(filename):
    return re.findall('(.*)_(master|data_\d{6}).h5', filename)[0][0]

def is_master(filename):
    #return 'master.h5' == filename[-9:]
    return re.match('(.*)_master.h5', filename)
    
def is_datafile(filename):
    return re.match('(.*)_(data_\d{6}).h5', filename)

def size_from_h_string(size_string, size_dict={'M':1024**2, 'K':1024, 'G':1024**3, '':1}):
    factor = size_dict[re.findall('[\d\.]*([GMK]?)', size_string)[0]]
    size = float(re.findall('([\d\.]*)[GMK]?', size_string)[0])*factor
    return size

def check_directory(directory):
    if not os.path.isdir(directory):
       try:
           os.makedirs(directory)
       except:
           log_print('Can not create directory')
           log_print(traceback.print_exc())

def run_exec_str(execstr):
    outp = Popen([execstr], stdout=PIPE, stderr=PIPE, shell=True).communicate()[1]
    return outp

def download(remote_filename, base_url='http://172.19.10.26/data/', cut_dirs=2):
    try:
        wget_running = subprocess.getoutput('ps aux | grep wget | grep -v "grep --color=auto" | grep %s' % remote_filename)
    except:
        wget_running = ''
        log_print(traceback.print_exc()) #traceback.format_exc())
    if remote_filename not in wget_running:
       wget_line = '(time wget -nH -nv --cut-dirs=%d --backups=3 \'%s%s\') 2>&1 ' % (cut_dirs, base_url, remote_filename)
       log_print('wget_line: %s ' % wget_line)
       return run_exec_str(wget_line)
    
def log_print(msg, lvl=INFO, timing=False):
    "Function used to replace all print"
    prnt_msg = msg
    if lvl == WARNING:
        prnt_msg = COLOR_W + " WARNING: " + msg + END_COLOR + "\n"
        prnt_func = prntLog.warning
    elif lvl == ERROR:
        prnt_msg = COLOR_E + "!ERROR! " + msg + END_COLOR +  "\n"
        prnt_func = prntLog.error
    elif lvl == CRITICAL:
        prnt_msg = COLOR_C + "!CRITICAL! " + msg + END_COLOR + "\n"
        prnt_func = prntLog.critical
    elif lvl == INFO:
        prnt_func = prntLog.info
    elif lvl == DEBUG:
        prnt_func = prntLog.debug
    if not fileHandler:
        print(prnt_msg)
        return
    if timing and fileHandler:
        fileHandler.setFormatter(logFormatter)
    elif fileHandler:
        fileHandler.setFormatter(cslFormatter)
    prnt_func(prnt_msg)
    if lvl == CRITICAL:
        sys.exit(0)
     
def main():
    
    h = HTMLParser.HTMLParser()
    print('eiger_IP', eiger_IP)
    print('port', 80)
    e = eiger.eiger(host=eiger_IP, port=80)
 
    username = os.getlogin()
    owner_uid = os.getuid()
    
    # remove proxy_definition for wget
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    today = date.today()
    LOGNAME = os.path.join(downloader_log_directory, "eiger_DL_logs",  "%s.downloader" % username)
    WGETLOG = os.path.join(downloader_log_directory, "eiger_DL_logs",  "%s.WGET" % username)
   
    _strdate = date.today().isoformat()
    fileHandler = logging.FileHandler("%s_%s.log" % (LOGNAME, _strdate))
    fileHandler.setFormatter(logFormatter)
    prntLog.addHandler(fileHandler)
    
    with open(WGETLOG+"_%s.log" % (_strdate),"a") as wgetlog:
        while True:
            try:
                filenames = e.get_filenames()
                file_size_and_name = e.get_file_size_and_name()
                if filenames:
                    to_download = len(file_size_and_name)
                    log_print("+++ New files to download: #%d" % to_download)
            except e:
                log_print('Can not connect to the detector', ERROR)
                log_print(traceback.print_exc())
                log_print('Waiting for connection to come up again')
                log_print('Checking again in %s seconds' % (time_out))
                raise e
                time.sleep(time_out)	 

            for k, filename in enumerate(file_size_and_name):
                log_print('%d (of %d) filename %s' % (k+1, to_download, filename), timing=True)
                filesystem_filename = h.unescape(filename).replace('nfsruche', 'nfs/ruche')
                log_print('filesystem_filename %s' % filesystem_filename)
                filesystem_filename = filesystem_filename.replace('\t', '_').replace('\n','_')
                log_print('filesystem_filename %s' % filesystem_filename)
                rfilename = file_size_and_name[filename]['rfilename']
                log_print('rfilename %s' % rfilename)
                file_on_disk = os.path.basename(filesystem_filename)
                
                if rfilename != file_on_disk:
                    log_print('Unconventional filename rfilename %s, file_on_disk: %s' % (rfilename, file_on_disk), WARNING)
                
                try:
                    filesystem_filename_without_potential_uid = filesystem_filename[filesystem_filename.index('/'):]
                    log_print('filesystem_filename_without_potential_uid %s' % filesystem_filename_without_potential_uid)
                    potential_uid = filesystem_filename[:filesystem_filename.index('/')]
                    if potential_uid.isdigit():
                        uid = potential_uid
                        if uid != str(owner_uid):
                            log_print('File %s belongs to user %s this instance of downloader belongs to user %s, skipping ...' % (filename, uid, owner_uid))
                            continue
                    else:
                        uid = ''
                        filesystem_filename_without_potential_uid = '/%s' % filesystem_filename
                        
                    destination = os.path.dirname(filesystem_filename_without_potential_uid)
                   
                    log_print('destination from filename %s ' % destination)
                except ValueError:
                    destination = default_destination
                    filesystem_filename_without_potential_uid = filesystem_filename
                    
                name_pattern = get_name_pattern(filename)
                log_print('name_pattern %s' % name_pattern)
                try:
                    log_print('downloading %s into %s' % (filename, destination))
                    #check_directory(os.path.dirname(destination))
                    check_directory(destination)
                    os.chdir(destination)
                    wlog = download(filename, cut_dirs=2)
                    wgetlog.write(wlog)
                    #download_speed = wlog.splitlines()[-1][21:]
                    #log_print('    Download speed: %s' % download_speed)
                    
                    if is_master(filesystem_filename_without_potential_uid):
                        command_line = '/home/experiences/proxima2a/com-proxima2a/bin/fix_master.py -m %s' % filesystem_filename_without_potential_uid
                        log_print('executing %s' % command_line)
                        #os.system(command_line)
                        run_exec_str(command_line)
                except:
                    destination = os.path.join(default_destination, date.today().isoformat())
                    check_directory(destination)
                    log_print('destination_default %s' % destination)
                    log_print('downloading_default %s into %s' % (filesystem_filename_without_potential_uid, destination))
                    os.chdir(destination)
                    wlog = download(filename, cut_dirs=1)
                    wgetlog.write(wlog)
                    #download_speed = wlog.splitlines()[-1][21:]
                    #log_print('    Download speed: %s' % download_speed)
                    log_print(traceback.print_exc())
                    log_print('Unknown destination for file %s, please inspect' % name_pattern, ERROR)
                    
                log_print('Veryfing existence of file %s' % filesystem_filename_without_potential_uid, DEBUG)

                if os.path.exists(filesystem_filename_without_potential_uid):
                    size_on_disk = os.stat(filesystem_filename_without_potential_uid).st_size
                    size_on_server = size_from_h_string(file_size_and_name[filename]['size'])
                    size_difference = abs(size_on_disk - size_on_server)
                    try:
                        size_from_webdav = os.stat('/mnt/eiger/%s' % filename).st_size
                        log_print('size_from_webdav %s' % size_from_webdav)
                        size_webdav_difference = abs(size_on_disk - size_from_webdav)
                        log_print('size_webdav_difference %s' % size_webdav_difference)
                    except:
                        size_from_webdav = None
                        size_webdav_difference = -1
                    log_print('approximate file size on the server %s' % size_on_server)
                    log_print('exact file size on the disk %s' % size_on_disk)
                    log_print('difference in webdav partition size and exact local size %6.4f kB' % (size_webdav_difference/1024.,))
                    log_print('difference in approximate server and exact local size %6.4f kB' % (size_difference/1024.,))
                    if size_webdav_difference == 0:
                        remove_line = '%s' % filename
                        #remove_line = '%s' % rfilename
                        log_print("File size for downloaded and original matches perfectly for %s. Removing from server..." % (rfilename,))
                        log_print("removing %s" % remove_line)
                        e.remove_files(remove_line)
                    elif size_difference < acceptable_apparent_file_size_difference:
                        remove_line = '%s' % filename
                        #remove_line = '%s' % rfilename
                        log_print("File size match acceptable (%.1f Mb) for %s. Removing from server..." % (size_difference, rfilename))
                        log_print("removing %s" % remove_line)
                        e.remove_files(remove_line)
                    else:
                        log_print("File size doesn't match (%.1f kB) for: %s" % (size_difference/1024., rfilename), ERROR)
                log_print(3*'\n')
            time.sleep(check_time)

if __name__ == '__main__':
    main()
