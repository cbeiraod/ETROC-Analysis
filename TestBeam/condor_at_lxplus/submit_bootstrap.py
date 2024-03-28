import shutil, os
from pathlib import Path
import argparse
from glob import glob

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='offline translate script',
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
    '-i',
    '--iteration',
    metavar = 'ITERATION',
    type = int,
    help = 'Number of bootstrapping',
    default = 100,
    dest = 'iteration',
)

parser.add_argument(
    '-s',
    '--sampling',
    metavar = 'SAMPLING',
    type = int,
    help = 'Random sampling fraction',
    default = 75,
    dest = 'sampling',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

args = parser.parse_args()
current_dir = Path('./')

files = glob(f'{args.dirname}/*pkl')
listfile = current_dir / 'input_list_for_bootstrap.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
    for ifile in files:
        name = ifile.split('/')[-1].split('.')[0]
        save_string = f"{name}, {ifile}"
        listfile.write(save_string + '\n')

outdir = current_dir / f'resolution_{args.dirname}'
outdir.mkdir(exist_ok = False)

bash_script = """#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

echo "python bootstrap.py -f ${{1}}.pkl -i {0} -s {1}"
python bootstrap.py -f ${{1}}.pkl -i {0} -s {1}
""".format(args.iteration, args.sampling)

with open('run_bootstrap.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if log_dir.exists():
    os.system('rm condor_logs/*log')
    os.system('rm condor_logs/*stdout')
    os.system('rm condor_logs/*stderr')
    os.system('ls condor_logs | wc -l')

jdl = """universe              = vanilla
executable            = run_bootstrap.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(ifile)
transfer_Input_Files  = bootstrap.py,$(path)
TransferOutputRemaps = "$(ifile)_resolution.pkl={1}/$(ifile)_resolution.pkl"
output                = {0}/$(ClusterId).$(ProcId).stdout
error                 = {0}/$(ClusterId).$(ProcId).stderr
log                   = {0}/$(ClusterId).$(ProcId).log
MY.WantOS             = "el9"
+JobFlavour           = "microcentury"
Queue ifile,path from input_list_for_bootstrap.txt
""".format(str(log_dir), str(outdir))

with open(f'condor_bootstrap.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if not args.dryrun:
    os.system(f'condor_submit condor_bootstrap.jdl')
