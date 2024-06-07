import pandas as pd
import argparse
import re
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from tqdm import tqdm

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
    dest = 'inputdir',
)

args = parser.parse_args()

input_dir = Path(args.inputdir)
files = natsorted(list(input_dir.glob('track*pkl')))

final_dict = defaultdict(list)

for ifile in tqdm(files):
    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, str(ifile))

    tmp_df = pd.read_pickle(ifile)

    for idx in range(len(matches)):
        final_dict[f'row{idx}'].append(matches[idx][0])
        final_dict[f'col{idx}'].append(matches[idx][1])
    final_dict['nevt'].append(tmp_df.shape[0])

    del tmp_df

track_nevt_df = pd.DataFrame(data=final_dict)
track_nevt_df.to_csv(f'{args.inputdir}_nevt_per_track.csv', index=False)
