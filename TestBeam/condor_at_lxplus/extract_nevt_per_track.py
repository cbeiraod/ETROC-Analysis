import pandas as pd
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
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

parser.add_argument(
    '-o',
    '--outputdir',
    metavar = 'DIRNAME',
    type = str,
    help = 'output directory name',
    required = True,
    dest = 'outputdir',
)

args = parser.parse_args()

input_dir = Path(args.inputdir)
files = natsorted(list(input_dir.glob('track*pkl')))

final_dict = defaultdict(list)

def process_file(ifile):
    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, str(ifile))

    file_dict = defaultdict(list)
    for idx in range(len(matches)):
        file_dict[f'row{idx}'].append(matches[idx][0])
        file_dict[f'col{idx}'].append(matches[idx][1])

    tmp_df = pd.read_pickle(ifile)
    file_dict['nevt'].append(tmp_df.shape[0])

    del tmp_df
    return file_dict

# Process files in parallel
with tqdm(files) as pbar:
    with ProcessPoolExecutor() as executor:
        future_to_file = [executor.submit(process_file, ifile) for ifile in files]

    for future in as_completed(future_to_file):
        pbar.update(1)
        result = future.result()
        for key, value in result.items():
            final_dict[key].extend(value)

track_nevt_df = pd.DataFrame(data=final_dict)
track_nevt_df.to_csv(f'{args.outputdir}_nevt_per_track.csv', index=False)
