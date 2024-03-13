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
    metavar = 'DIRNAME',
    type = str,
    help = 'input directory name',
    required = True,
    dest = 'dirname',
)

parser.add_argument(
    '-t',
    '--track',
    metavar = 'NAME',
    type = str,
    help = 'csv file including track candidates',
    required = True,
    dest = 'track',
)

parser.add_argument(
    '--refID',
    metavar = 'ID',
    type = int,
    help = 'reference board ID',
    required = True,
    dest = 'refID',
)

parser.add_argument(
    '--dutID',
    metavar = 'ID',
    type = int,
    help = 'DUT board ID',
    required = True,
    dest = 'dutID',
)

parser.add_argument(
    '--ignoreID',
    metavar = 'ID',
    type = int,
    help = 'board ID be ignored',
    required = True,
    dest = 'ignoreID',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()
current_dir = Path('./')

files = glob(f'{args.dirname}/*feather')
listfile = current_dir / 'input_files.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
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

echo "python dataSelectionByTrack.py -f ${{1}} -r ${{2}} -t {0} --refID {1} --dutID {2} --ignoreID {3}"
python dataSelectionByTrack.py -f ${{1}} -r ${{2}} -t {0} --refID {1} --dutID {2} --ignoreID {3}
""".format(args.track, args.refID, args.dutID, args.ignoreID)

with open('run.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

jdl = """universe              = vanilla
executable            = run.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run)
transfer_Input_Files  = dataSelectionByTrack.py,{1},$(path)
output                = {0}/$(ClusterId).$(ProcId).stdout
error                 = {0}/$(ClusterId).$(ProcId).stderr
log                   = {0}/$(ClusterId).$(ProcId).log
MY.WantOS             = "el9"
+JobFlavour           = "espresso"
Queue run,fname,path from input_files.txt
""".format(str(log_dir), args.track)

with open(f'condor_dataSelection.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if not args.dryrun:
    os.system(f'condor_submit condor_dataSelection.jdl')
