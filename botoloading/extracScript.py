#!/opt/anaconda3/envs/goesenv/bin/python

from extractROIfiles import processFilesToROI
import datetime
from subprocess import Popen

def main():
    cmdstr = """
./extractROIfiles.py --platform=17 --dayofyear=101,102,103,104 --band=8 --config=extract.yaml --data-dir=/mnt/g/GOESProj/data/sharedData/scratch/april_data
./extractROIfiles.py --platform=16 --dayofyear=101,102,103,104 --band=8 --config=extract.yaml --data-dir=/mnt/g/GOESProj/data/sharedData/scratch/april_data 
./extractROIfiles.py --platform=17 --dayofyear=104,103,102,101 --band=8 --config=extract.yaml --data-dir=/mnt/g/GOESProj/data/sharedData/scratch/april_data
./extractROIfiles.py --platform=16 --dayofyear=104,103,102,101 --band=8 --config=extract.yaml --data-dir=/mnt/g/GOESProj/data/sharedData/scratch/april_data
    """
    cmds = [cmd for cmd in cmdstr.split('\n') if cmd]
    for cmd in cmds:
        pid = Popen(cmd, shell=True)
if __name__ == '__main__':
    main()