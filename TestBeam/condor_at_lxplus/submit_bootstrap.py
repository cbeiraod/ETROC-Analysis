import os
from pathlib import Path
import argparse
from glob import glob
from jinja2 import Template

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
    '--board_id_for_TOA_cut',
    metavar = 'NUM',
    type = int,
    help = 'TOA range cut will be applied to a given board ID',
    default = 1,
    dest = 'board_id_for_TOA_cut',
)

parser.add_argument(
    '--minimum_nevt',
    metavar = 'NUM',
    type = int,
    help = 'Minimum number of events for bootstrap',
    default = 1000,
    dest = 'minimum_nevt',
)

parser.add_argument(
    '--trigTOALower',
    metavar = 'NUM',
    type = int,
    help = 'Lower TOA selection boundary for the trigger board',
    default = 100,
    dest = 'trigTOALower',
)

parser.add_argument(
    '--trigTOAUpper',
    metavar = 'NUM',
    type = int,
    help = 'Upper TOA selection boundary for the trigger board',
    default = 500,
    dest = 'trigTOAUpper',
)

parser.add_argument(
    '--autoTOTcuts',
    action = 'store_true',
    help = 'If set, select 80 percent of data around TOT median value of each board',
    dest = 'autoTOTcuts',
)

parser.add_argument(
    '--noTrig',
    action = 'store_true',
    help = 'If set, trigger will not be considered for the analysis',
    dest = 'noTrig',
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

# Define the bash script template
bash_template = """
#!/bin/bash

ls -ltrh
echo ""
pwd

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-el9-gcc13-opt/setup.sh

echo "python bootstrap.py -f {{ filename }} -i {{ iteration }} -s {{ sampling }} --board_id_for_TOA_cut {{ board_id_for_TOA_cut }} --minimum_nevt {{ minimum_nevt }} --trigTOALower {{ trigTOALower }} --trigTOAUpper {{ trigTOAUpper }}{% if autoTOTcuts %} --autoTOTcuts{% endif %}{% if noTrig %} --noTrig{% endif %}"
python bootstrap.py -f {{ filename }} -i {{ iteration }} -s {{ sampling }} --board_id_for_TOA_cut {{ board_id_for_TOA_cut }} --minimum_nevt {{ minimum_nevt }} --trigTOALower {{ trigTOALower }} --trigTOAUpper {{ trigTOAUpper }}{% if autoTOTcuts %} --autoTOTcuts{% endif %}{% if noTrig %} --noTrig{% endif %}
"""

# Prepare the data for the template
options = {
    'filename': '${1}.pkl',
    'iteration': args.iteration,
    'sampling': args.sampling,
    'board_id_for_TOA_cut': args.board_id_for_TOA_cut,
    'minimum_nevt': args.minimum_nevt,
    'trigTOALower': args.trigTOALower,
    'trigTOAUpper': args.trigTOAUpper,
    'autoTOTcuts': args.autoTOTcuts,
    'noTrig': args.noTrig
}

# Render the template with the data
bash_script = Template(bash_template).render(options)

print('\n========= Run option =========')
print(f'Bootstrap iteration: {args.iteration}')
print(f'{args.sampling}% of random sampling')
print(f"TOA cut for a 'NEW' trigger is {args.trigTOALower}-{args.trigTOAUpper}")
print(f'Number of events larger than {args.minimum_nevt} will be considered')
if args.autoTOTcuts:
    print(f'Automatic TOT cuts will be applied')
if args.noTrig:
    print('Trigger board will not be considered')
print('========= Run option =========\n')

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