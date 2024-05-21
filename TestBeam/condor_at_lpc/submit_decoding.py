import subprocess, sys, os
from pathlib import Path
import argparse
from glob import glob
from jinja2 import Template
from natsort import natsorted

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='submit decoding jobs on condor',
        )

parser.add_argument(
    '--data_era',
    metavar = 'NAME',
    type = str,
    help = "Which TB dataset, e.g. desy_tb_data_Feb2024. Please check by the command: eosls /store/group/lpcmtdstudies",
    required = True,
    dest = 'data_era',
)

parser.add_argument(
    '--run_name',
    metavar = 'NAME',
    type = str,
    help = 'extra run information for output directory name. Example: Run_X. X can be any number.',
    required = True,
    dest = 'run_name',
)

parser.add_argument(
    '--dryrun',
    action = 'store_true',
    help = 'If set, condor submission will not happen',
    dest = 'dryrun',
)

parser.add_argument(
    '--clear_condor_logs',
    action = 'store_true',
    help = 'If set, clear condor log files',
    dest = 'clear_condor_logs',
)

args = parser.parse_args()

current_dir = Path('./')
eosls_command = ["eos", "root://cmseos.fnal.gov", "ls", f"/store/group/lpcmtdstudies/{args.data_era}/{args.run_name}",]
result = subprocess.run(eosls_command, capture_output=True, text=True)
dir_list = natsorted(result.stdout.split())

if len(dir_list) == 0:
    print('No input directories are found')
    print('Double-check the input path')
    sys.exit()

listfile = current_dir / 'input_list_for_decoding.txt'
if listfile.is_file():
    listfile.unlink()

with open(listfile, 'a') as listfile:
    for idir in dir_list:
        path = f"root://cmseos.fnal.gov//store/group/lpcmtdstudies/{args.data_era}/{args.run_name}/{idir}"
        save_string = f"{idir}, {path}"
        listfile.write(save_string + '\n')

outdir = f'{args.run_name}_feather'
print(f"Make output directory {outdir} at /store/group/lpcmtdstudies/{args.data_era}")
os.system(f'eos root://cmseos.fnal.gov mkdir -p /store/group/lpcmtdstudies/{args.data_era}/{outdir}')
os.system(f"eos root://cmseos.fnal.gov ls /store/group/lpcmtdstudies/{args.data_era} | grep --color \"feather\"")
print()

eosls_command = ["eos", "root://cmseos.fnal.gov", "ls", f"/store/group/lpcmtdstudies/{args.data_era}/{args.run_name}_feather",]
result = subprocess.run(eosls_command, capture_output=True, text=True)
file_list = natsorted(result.stdout.split())

if len(file_list) > 0:
    print('It looks like feather files already exists.')
    print('First flie:', file_list[0], '/ Last file:', file_list[-1])
    print('Do you really want to submit jobs?')
    print('If so, please remove or rename the directory first')
    print('EOS rename command:')
    print('eos root://cmseos.fnal.gov file rename /store/user/username/testing_old.root /store/user/username/testing_Name_v2.txt')
    print('EOS mv command:')
    print('eos root://cmseos.fnal.gov mv /store/user/tonjes/step4_test.root /store/group/lpcci2dileptons/tonjes_test.root')
    sys.exit()

# Define the bash script template
bash_template = """#!/bin/bash

# Check current directory to make sure that input files are transferred
ls -ltrh
echo ""

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

# Copy input data from EOS to local work node
xrdcp -r {{ input_dir_path }} ./

# Untar python environment
tar -xf python_lib.tar

# Check untar output
ls -ltrh

# Set custom python environment
export PYTHONPATH=${PWD}/local/lib/python3.9/site-packages:$PYTHONPATH
echo "${PYTHONPATH}"
echo ""

echo "python decoding.py -d {{ input_dir_name }}"
python decoding.py -d {{ input_dir_name }}

# Check output is there
ls -ltrh
echo ""

# Copy output using xrdcp
echo "xrdcp {{ input_dir_name }}.feather root://cmseos.fnal.gov//store/group/lpcmtdstudies/{{ data_era }}/{{ output_dir }}"
xrdcp {{ input_dir_name }}.feather root://cmseos.fnal.gov//store/group/lpcmtdstudies/{{ data_era }}/{{ output_dir }}
XRDEXIT=$?
if [[ $XRDEXIT -ne 0 ]]; then
    echo "exit code $XRDEXIT, failure in xrdcp"
    exit $XRDEXIT
fi
rm {{ input_dir_name }}.feather

"""

# Prepare the data for the template
options = {
    'input_dir_name': '${1}',
    'input_dir_path': '${2}',
    'data_era': args.data_era,
    'output_dir': outdir,
}

# Render the template with the data
bash_script = Template(bash_template).render(options)

with open('run_decode.sh','w') as bashfile:
    bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

if args.clear_condor_logs:
    os.system('rm condor_logs/*decoding*log')
    os.system('rm condor_logs/*decoding*stdout')
    os.system('rm condor_logs/*decoding*stderr')
    os.system('ls condor_logs/*decoding*log | wc -l')

jdl = """universe              = vanilla
executable            = run_decode.sh
should_Transfer_Files = YES
whenToTransferOutput  = ON_EXIT
arguments             = $(name) $(path)
transfer_Input_Files  = decoding.py, python_lib.tar
output                = {0}/$(ClusterId).$(ProcId).decoding.stdout
error                 = {0}/$(ClusterId).$(ProcId).decoding.stderr
log                   = {0}/$(ClusterId).$(ProcId).decoding.log
Queue name, path from input_list_for_decoding.txt
""".format(str(log_dir))

with open(f'condor_decoding.jdl','w') as jdlfile:
    jdlfile.write(jdl)

if args.dryrun:
    print('=============== Input list ===============')
    os.system('cat input_list_for_decoding.txt')
    print()
    print('=============== bash script ===============')
    os.system('cat run_decode.sh')
else:
    os.system(f'condor_submit condor_decoding.jdl')
