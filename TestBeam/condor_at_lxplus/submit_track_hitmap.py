import os, re
from pathlib import Path
import argparse
from glob import glob

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='submit condor job!',
        )

parser.add_argument(
    '-d',
    '--inputdir',
    nargs='+',
    metavar = 'DIRNAME',
    type = str,
    help = 'input directory name',
    required = True,
    dest = 'dirname',
)

parser.add_argument(
    '-r',
    '--row',
    metavar = 'NUM',
    type = int,
    help = 'Row of pixel',
    default = 8,
    dest = 'row',
)

parser.add_argument(
    '-c',
    '--col',
    metavar = 'NUM',
    type = int,
    help = 'Column of pixel',
    default = 11,
    dest = 'col',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()
current_dir = Path('./')

dirs = args.dirname

listfile = current_dir / 'input_list_for_hitmap.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
    for idir in dirs:
        files = glob(f'{idir}/*feather')
        for ifile in files:
            pattern = r'Run_(\d+)'
            fname = ifile.split('/')[-1]
            matches = re.findall(pattern, ifile)
            save_string = f"run{matches[0]}, {fname}, {ifile}"
            listfile.write(save_string + '\n')

bash_script = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

echo "python track_hitmap.py -f ${{1}} -r ${{2}} --row {0} --col {1}"
python track_hitmap.py -f ${{1}} -r ${{2}} --row {0} --col {1}
""".format(args.row, args.col)

with open('run_track_hitmap.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if log_dir.exists():
    os.system('rm condor_logs/*log')
    os.system('rm condor_logs/*stdout')
    os.system('rm condor_logs/*stderr')
    os.system('ls condor_logs | wc -l')

jdl = """universe              = vanilla
executable            = run_track_hitmap.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run)
transfer_Input_Files  = track_hitmap.py,$(path)
output                = {0}/$(ClusterId).$(ProcId).hitmap.stdout
error                 = {0}/$(ClusterId).$(ProcId).hitmap.stderr
log                   = {0}/$(ClusterId).$(ProcId).hitmap.log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue run,fname,path from input_list_for_hitmap.txt
""".format(str(log_dir))

with open(f'condor_track_hitmap.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if not args.dryrun:
    os.system(f'condor_submit condor_track_hitmap.jdl')
