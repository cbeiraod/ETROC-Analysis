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

board_names = ['ET2_EPIR_Pair1', 'ET2_BAR_4', 'ET2_EPIR_Pair4']

base_dir = Path('./')
outputdir = base_dir / args.outdir
outputdir.mkdir(exist_ok=False)

def merge_dataframes(files):
    merged_data = {}  # Dictionary to store merged DataFrames
    keys = None  # To store keys from the first file
    for file in files:
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
merged_dict = merge_dataframes(files[:2])

for ikey in tqdm(merged_dict.keys()):
    print(merged_dict[ikey])
    break
    # merged_dict[ikey].drop(columns=['evt'], inplace=True)

    # board_ids = merged_dict[ikey].columns.get_level_values('board').unique().tolist()
    # merged_dict[ikey]['evt'] = range(merged_dict[ikey].shape[0])
    # merged_dict[ikey] = merged_dict[ikey].astype('uint64')
    # positions = [f'R{merged_dict[ikey]["row"][idx].unique()[0]}C{merged_dict[ikey]["col"][idx].unique()[0]}' for idx in board_ids]

    # merged_dict[ikey].to_pickle(outputdir / f'track_{ikey}_{board_names[0]}_{positions[0]}_{board_names[1]}_{positions[1]}_{board_names[2]}_{positions[2]}.pkl')
