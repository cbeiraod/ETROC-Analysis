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
    '-t',
    '--track',
    metavar = 'NAME',
    type = str,
    help = 'csv file including track candidates',
    required = True,
    dest = 'track',
)

parser.add_argument(
    '--trigID',
    metavar = 'ID',
    type = int,
    help = 'trigger board ID',
    required = True,
    dest = 'trigID',
)

parser.add_argument(
    '--refID',
    metavar = 'ID',
    type = int,
    help = 'reference board ID',
    default = 3,
    dest = 'refID',
)

parser.add_argument(
    '--dutID',
    metavar = 'ID',
    type = int,
    help = 'DUT board ID',
    default = 1,
    dest = 'dutID',
)

parser.add_argument(
    '--ignoreID',
    metavar = 'ID',
    type = int,
    help = 'board ID be ignored',
    default = 2,
    dest = 'ignoreID',
)

parser.add_argument(
    '--trigTOTLower',
    metavar = 'NUM',
    type = int,
    help = 'Lower TOT selection boundary for the trigger board',
    default = 100,
    dest = 'trigTOTLower',
)

parser.add_argument(
    '--trigTOTUpper',
    metavar = 'NUM',
    type = int,
    help = 'Upper TOT selection boundary for the trigger board',
    default = 200,
    dest = 'trigTOTUpper',
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

listfile = current_dir / 'input_list_for_dataSelection.txt'
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

echo "python track_data_selection.py -f ${{1}} -r ${{2}} -t {0} --trigID {1} --refID {2} --dutID {3} --ignoreID {4} --trigTOTLower {5} --trigTOTUpper {6}"
python track_data_selection.py -f ${{1}} -r ${{2}} -t {0} --trigID {1} --refID {2} --dutID {3} --ignoreID {4} --trigTOTLower {5} --trigTOTUpper {6}
""".format(args.track, args.trigID, args.refID, args.dutID, args.ignoreID, args.trigTOTLower, args.trigTOTUpper)

with open('run_track_data_selection.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if log_dir.exists():
    os.system('rm condor_logs/*log')
    os.system('rm condor_logs/*stdout')
    os.system('rm condor_logs/*stderr')
    os.system('ls condor_logs | wc -l')

jdl = """universe              = vanilla
executable            = run_track_data_selection.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(fname) $(run)
transfer_Input_Files  = track_data_selection.py,{1},$(path)
output                = {0}/$(ClusterId).$(ProcId).stdout
error                 = {0}/$(ClusterId).$(ProcId).stderr
log                   = {0}/$(ClusterId).$(ProcId).log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue run,fname,path from input_list_for_dataSelection.txt
""".format(str(log_dir), args.track)

with open(f'condor_track_data_selection.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if not args.dryrun:
    os.system(f'condor_submit condor_track_data_selection.jdl')
