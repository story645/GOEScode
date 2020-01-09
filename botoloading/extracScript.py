#!/opt/anaconda3/envs/goesenv/bin/python

from extractROIfiles import processFilesToROI
import datetime
from subprocess import Popen

def main():
    cmdstr = """
./extractROIfiles.py --platform=17 --dayofyear=240,241 --band=8 --config=extract.yaml
./extractROIfiles.py --platform=17 --dayofyear=242,243 --band=8 --config=extract.yaml
./extractROIfiles.py --platform=17 --dayofyear=244,245 --band=8 --config=extract.yaml
./extractROIfiles.py --platform=16 --dayofyear=240,241 --band=8 --config=extract.yaml
./extractROIfiles.py --platform=16 --dayofyear=242,243 --band=8 --config=extract.yaml
./extractROIfiles.py --platform=16 --dayofyear=244,245 --band=8 --config=extract.yaml
    """
    cmds = [cmd for cmd in cmdstr.split('\n') if cmd]
    for cmd in cmds:
        pid = Popen(cmd, shell=True)
if __name__ == '__main__':
    main()