from natsort import natsorted
from glob import glob
from pathlib import Path
import pandas as pd
import pickle
import argparse
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

def merge_dataframes_from_pickles(pickle_files: list[str]):
    dict_list = []
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            dictionary = pickle.load(f)
            dict_list.append(dictionary)

    merged_dict = {}
    for key in dict_list[0].keys():
        merged_df = pd.concat([d[key] for d in dict_list], ignore_index=True)
        merged_dict[key] = merged_df

    return merged_dict

# final_dict = defaultdict(list)
files = natsorted(glob(args.dirname+'/*pickle'))

# Merge the dataframes
merged_dict = merge_dataframes_from_pickles(files)

for ikey in merged_dict.keys():
    merged_dict[ikey].drop(columns=['evt'], inplace=True)

    board_ids = merged_dict[ikey].columns.get_level_values('board').unique().tolist()
    merged_dict[ikey]['evt'] = range(merged_dict[ikey].shape[0])
    merged_dict[ikey] = merged_dict[ikey].astype('uint64')
    positions = [f'R{merged_dict[ikey]["row"][idx].unique()[0]}C{merged_dict[ikey]["col"][idx].unique()[0]}' for idx in board_ids]

    merged_dict[ikey].to_pickle(outputdir / f'track_{board_names[0]}_{positions[0]}_{board_names[1]}_{positions[1]}_{board_names[2]}_{positions[2]}.pkl')