import os
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

rootdir = args.dirname
dirs = glob(rootdir+'/TID_testing_candidate_*')

logpath = './log'
if not os.path.exists(logpath):
    os.mkdir(logpath)

outdir = './translated_outputs'
if not os.path.exists(outdir):
    os.mkdir(outdir)

for i, idir in enumerate(dirs):
    name = idir.split('/')[1]
    #print(f"tar --exclude-caches-all --exclude-vcs --exclude='*translated*' -cf {idir}.tar -C {rootdir} {name}")
    if not os.path.exists(f'{idir}.tar'):
        os.system(f"tar --exclude-caches-all --exclude-vcs --exclude='*translated*' -cf {idir}.tar -C {rootdir} {name}")

    os.environ['DIR'] = name

    jdl = """
    universe              = vanilla
    Executable            = run.sh
    Should_Transfer_Files = YES
    WhenToTransferOutput  = ON_EXIT
    Transfer_Input_Files  = run.sh, simplified_standalone_translate_etroc2_data.py, {1}.tar
    transfer_output_files = {2}_translated.tgz
    TransferOutputRemaps = "{2}_translated.tgz=translated_outputs/{2}_translated.tgz"
    arguments             = $ENV(DIR)
    Output                = {0}/$(Cluster)_$(Process).stdout
    Error                 = {0}/$(Cluster)_$(Process).stderr
    Log                   = {0}/$(Cluster)_$(Process).log
    +JobFlavour           = "espresso"
    Queue 1
    """.format(logpath, f'{idir}', name)

    with open(f'{logpath}/condor_{name}.sub','w') as jdlfile:
        jdlfile.write(jdl)

    #print(f'condor_submit {logpath}/condor_{name}.sub')
    os.system(f'condor_submit {logpath}/condor_{name}.sub')
