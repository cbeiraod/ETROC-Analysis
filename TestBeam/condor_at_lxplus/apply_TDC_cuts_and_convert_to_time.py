from natsort import natsorted
from pathlib import Path
import pandas as pd
import numpy as np
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

parser.add_argument(
    '--setTrigBoardID',
    metavar = 'NUM',
    type = int,
    help = 'Set the offline trigger board ID',
    required = True,
    dest = 'setTrigBoardID',
)

parser.add_argument(
    '--setDUTBoardID',
    metavar = 'NUM',
    type = int,
    help = 'Set the DUT board ID',
    required = True,
    dest = 'setDUTBoardID',
)

parser.add_argument(
    '--setRefBoardID',
    metavar = 'NUM',
    type = int,
    help = 'Set the offline reference board ID',
    required = True,
    dest = 'setRefBoardID',
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

args = parser.parse_args()

base_dir = Path('./')
outputdir = base_dir / args.outdir
outputdir.mkdir(exist_ok=False)

track_dir = outputdir / 'tracks'
track_dir.mkdir(exist_ok=False)

time_dir = outputdir / 'time'
time_dir.mkdir(exist_ok=False)

board_ids = [args.setTrigBoardID, args.setDUTBoardID, args.setRefBoardID]

## --------------------------------------
def tdc_event_selection_pivot(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict
    ):
    # Create boolean masks for each board's filtering criteria
    masks = {}
    for board, cuts in tdc_cuts_dict.items():
        mask = (
            input_df['cal'][board].between(cuts[0], cuts[1]) &
            input_df['toa'][board].between(cuts[2], cuts[3]) &
            input_df['tot'][board].between(cuts[4], cuts[5])
        )
        masks[board] = mask

    # Combine the masks using logical AND
    combined_mask = pd.concat(masks, axis=1).all(axis=1)
    del masks

    # Apply the combined mask to the DataFrame
    tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)
    del combined_mask
    return tdc_filtered_df

## --------------------------------------
def merge_dataframes(files):
    merged_data = {}  # Dictionary to store merged DataFrames
    merged_data_in_time = {}
    keys = None  # To store keys from the first file
    for file in tqdm(files):
        with open(file, 'rb') as f:
            # data_dict = pickle.load(f)  # Load dictionary from file (assuming files are pickled)
            data_dict = pd.read_pickle(f)
            if keys is None:
                keys = data_dict.keys()  # Store keys from the first file
            else:
                if keys != data_dict.keys():
                    raise ValueError("Keys in dictionaries are not the same across all files")
            # Merge DataFrames for each key
            for key in data_dict.keys():

                tot_cuts = {}
                for idx in board_ids:
                    if args.autoTOTcuts:
                        lower_bound = data_dict[key]['tot'][idx].quantile(0.01)
                        upper_bound = data_dict[key]['tot'][idx].quantile(0.96)
                        tot_cuts[idx] = [round(lower_bound), round(upper_bound)]
                    else:
                        tot_cuts[idx] = [0, 600]

                ## Selecting good hits with TDC cuts
                tdc_cuts = {}
                for idx in board_ids:
                    if idx == args.setTrigBoardID:
                        tdc_cuts[idx] = [0, 1100, args.trigTOALower, args.trigTOAUpper, tot_cuts[idx][0], tot_cuts[idx][1]]
                    else:
                        tdc_cuts[idx] = [0, 1100, 0, 1100, tot_cuts[idx][0], tot_cuts[idx][1]]

                interest_df = tdc_event_selection_pivot(data_dict[key], tdc_cuts_dict=tdc_cuts)

                df_in_time = pd.DataFrame()
                for idx in board_ids:
                    bins = 3.125/interest_df['cal'][idx].mean()
                    df_in_time[f'toa_b{str(idx)}'] = (12.5 - interest_df['toa'][idx] * bins)*1e3
                    df_in_time[f'tot_b{str(idx)}'] = ((2*interest_df['tot'][idx] - np.floor(interest_df['tot'][idx]/32)) * bins)*1e3

                if key not in merged_data:
                    merged_data[key] = data_dict[key]
                    merged_data_in_time[key] = df_in_time
                else:
                    merged_data[key] = pd.concat([merged_data[key], data_dict[key]], ignore_index=True)
                    merged_data_in_time[key] = pd.concat([merged_data_in_time[key], df_in_time], ignore_index=True)

    return merged_data, merged_data_in_time

# final_dict = defaultdict(list)
files = natsorted(list(Path(args.dirname).glob('run*pickle')))

# Merge the dataframes
merged_track, merged_track_in_time = merge_dataframes(files)

for ikey in tqdm(merged_track.keys()):
    board_ids = merged_track[ikey].columns.get_level_values('board').unique().tolist()
    outname = f"track_{ikey}"
    for board_id in board_ids:
        irow = merged_track[ikey]['row'][board_id].unique()[0]
        icol = merged_track[ikey]['col'][board_id].unique()[0]
        outname += f"_R{irow}C{icol}"

    merged_track[ikey].to_pickle(track_dir / f'{outname}.pkl')
    merged_track_in_time[ikey].to_pickle(time_dir / f'{outname}.pkl')
