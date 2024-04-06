from natsort import natsorted
from glob import glob
from pathlib import Path
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='merge individual dataSelection results',
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
    '--outdir',
    metavar = 'OUTNAME',
    type = str,
    help = 'output directory name',
    required = True,
    dest = 'outdir',
)

args = parser.parse_args()

base_dir = Path('./')
outputdir = base_dir / args.outdir
outputdir.mkdir(exist_ok=False)

def merge_dataframes(files):
    merged_data = {}  # Dictionary to store merged DataFrames
    keys = None  # To store keys from the first file
    for file in tqdm(files):
        with open(file, 'rb') as f:
            data_dict = pickle.load(f)  # Load dictionary from file (assuming files are pickled)
            if keys is None:
                keys = data_dict.keys()  # Store keys from the first file
            else:
                if keys != data_dict.keys():
                    raise ValueError("Keys in dictionaries are not the same across all files")
            # Merge DataFrames for each key
            for key in data_dict.keys():
                if key not in merged_data:
                    merged_data[key] = data_dict[key]
                else:
                    merged_data[key] = pd.concat([merged_data[key], data_dict[key]], ignore_index=True)
    return merged_data

# final_dict = defaultdict(list)
files = natsorted(glob(args.dirname+'/*pickle'))

# Merge the dataframes
merged_dict = merge_dataframes(files)

for ikey in tqdm(merged_dict.keys()):
    board_ids = merged_dict[ikey].columns.get_level_values('board').unique().tolist()
    outname = f"track_{ikey}"
    for board_id in board_ids:
        irow = merged_dict[ikey]['row'][board_id].unique()[0]
        icol = merged_dict[ikey]['col'][board_id].unique()[0]
        outname += f"_R{irow}C{icol}"

    merged_dict[ikey].to_pickle(outputdir / f'{outname}.pkl')
