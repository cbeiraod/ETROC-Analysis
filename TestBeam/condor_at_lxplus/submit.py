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

args = parser.parse_args()

current_dir = Path('./')
base_dir = current_dir / 'condor_submit'
base_dir.mkdir(exist_ok=False)

files = glob(f'{args.dirname}/*pkl')
file_names = [os.path.basename(ifile) for ifile in files]

for idx, ifile in enumerate(files):

    single_name = file_names[idx].split('.')[0]
    submit_dir = base_dir / single_name
    submit_dir.mkdir(exist_ok=False)

    shutil.copyfile(src=ifile, dst=submit_dir / file_names[idx])
    shutil.copyfile(src='bootstrap.py', dst=submit_dir / 'bootstrap.py')

    bash_script = """
#!/bin/bash

# Load python environment from work node
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh

echo "python3 bootstrap.py -f {0} -i {1} -s {2}"
python3 bootstrap.py -f {0} -i {1} -s {2}
""".format(ifile, 100, 0.75)

    with open(submit_dir / 'run.sh','w') as bashfile:
        bashfile.write(bash_script)

log_dir = current_dir / 'condor_logs'
log_dir.mkdir(exist_ok=True)

jdl = """
universe              = vanilla
Executable            = run.sh
Should_Transfer_Files = YES
WhenToTransferOutput  = ON_EXIT
Transfer_Input_Files  = bootstrap.py, $(filename)
Output                = {0}/$(ClusterId).$(ProcId).stdout
Error                 = {0}/$(ClusterId).$(ProcId).stderr
Log                   = {0}/$(ClusterId).$(ProcId).log
+JobFlavour           = "espresso"
Queue filename matching ({1}/*/*.pkl)
""".format(str(log_dir), str(base_dir))

with open(f'condor_submit.sub','w') as jdlfile:
    jdlfile.write(jdl)

os.system(f'condor_submit condor_submit.sub')