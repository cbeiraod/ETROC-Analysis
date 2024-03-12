from collections import defaultdict
from natsort import natsorted
import argparse
from glob import glob
import pandas as pd
import re

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='merge individual bootstrap results',
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
    '-o',
    '--outputname',
    metavar = 'OUTNAME',
    type = str,
    help = 'output file name',
    required = True,
    dest = 'outname',
)

args = parser.parse_args()

final_dict = defaultdict(list)
files = natsorted(glob(args.dirname+'/*pkl'))

for ifile in files:

    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, ifile)

    df = pd.read_pickle(ifile)
    columns = df.columns

    for idx in range(len(columns)):
        final_dict[f'row{columns[idx]}'].append(matches[idx][0])
        final_dict[f'col{columns[idx]}'].append(matches[idx][1])

        final_dict[f'res{columns[idx]}'].append(df[columns[idx]].mean())
        final_dict[f'err{columns[idx]}'].append(df[columns[idx]].std())

final_df = pd.DataFrame(final_dict)
final_df.to_pickle(args.outname+'.pkl')