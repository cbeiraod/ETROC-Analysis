import pandas as pd
import numpy as np
from glob import glob
import random
from tqdm import tqdm
from pathlib import Path

## --------------------------------------
def tdc_event_selection(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict,
        select_by_hit: bool = False,
    ):

    if select_by_hit:

        # Create boolean masks for each board's filtering criteria
        masks = {}
        for board, cuts in tdc_cuts_dict.items():
            mask = (
                (input_df['board'] == board) &
                input_df['cal'].between(cuts[0], cuts[1]) &
                input_df['toa'].between(cuts[2], cuts[3]) &
                input_df['tot'].between(cuts[4], cuts[5])
            )
            masks[board] = mask

        # Combine the masks using logical OR
        combined_mask = pd.concat(masks, axis=1).any(axis=1)

        # Apply the combined mask to the DataFrame
        tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)

        return tdc_filtered_df

    else:
        from functools import reduce

        # Create boolean masks for each board's filtering criteria
        masks = {}
        for board, cuts in tdc_cuts_dict.items():
            mask = (
                (input_df['board'] == board) &
                input_df['cal'].between(cuts[0], cuts[1]) &
                input_df['toa'].between(cuts[2], cuts[3]) &
                input_df['tot'].between(cuts[4], cuts[5])
            )
            masks[board] = input_df[mask]['evt'].unique()

        common_elements = reduce(np.intersect1d, list(masks.values()))
        tdc_filtered_df = input_df.loc[input_df['evt'].isin(common_elements)].reset_index(drop=True)

        return tdc_filtered_df

## --------------------------------------
def singlehit_event_clear(
        input_df: pd.DataFrame,
        ignore_boards: list[int] = None
    ):

    ana_df = input_df
    if ignore_boards is not None:
        for board in ignore_boards:
            ana_df = ana_df.loc[ana_df['board'] != board].copy()

    ## event has one hit from each board
    event_board_counts = ana_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None
    for board in event_board_counts:
        if event_selection_col is None:
            event_selection_col = (event_board_counts[board] == 1)
        else:
            event_selection_col = event_selection_col & (event_board_counts[board] == 1)
    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = ana_df[ana_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    del ana_df, event_board_counts, event_selection_col

    return selected_subset_df

## --------------------------------------
def making_pivot(
        input_df: pd.DataFrame,
        index: str,
        columns: str,
        drop_columns: tuple,
        ignore_boards: list[int] = None
    ):
        ana_df = input_df
        if ignore_boards is not None:
            for board in ignore_boards:
                ana_df = ana_df.loc[ana_df['board'] != board].copy()
        pivot_data_df = ana_df.pivot(
        index = index,
        columns = columns,
        values = list(set(ana_df.columns) - drop_columns),
        )
        pivot_data_df.columns = ["{}_{}".format(x, y) for x, y in pivot_data_df.columns]

        return pivot_data_df

## --------------------------------------
def finding_tracks(
        input_files: list[Path],
        columns_to_read: list[str],
        ignore_board_ids: list[int],
        outfile_name: str,
        group_for_pivot: list[str],
        drop_for_pivot: list[str],
        iteration: int = 10,
        sampling_fraction: int = 20,
        minimum_number_of_tracks: int = 1000,
        dut_id: int = 1,
        ref_id: int = 3,
        red_2nd_id: int = -1,
    ):

    final_list = []
    sampling_fraction = sampling_fraction * 0.01

    for isampling in tqdm(range(iteration)):
        files = random.sample(input_files, k=int(sampling_fraction*len(input_files)))

        last_evt = 0
        dataframes = []
        num_failed_files = 0

        for idx, ifile in enumerate(files):
            tmp_df = pd.read_feather(ifile, columns=columns_to_read)

            if tmp_df.empty:
                num_failed_files += 1
                print('file is empty, move on to the next file')
                continue

            event_board_counts = filtered_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
            event_selection_col = None

            if red_2nd_id == -1:
                trig_selection = (event_board_counts[0] == 1)
                ref_selection = (event_board_counts[ref_id] == 1)
                event_selection_col = trig_selection & ref_selection
            else:
                trig_selection = (event_board_counts[0] == 1)
                ref_selection = (event_board_counts[ref_id] == 1)
                ref_2nd_selection = (event_board_counts[red_2nd_id] == 1)
                event_selection_col = trig_selection & ref_selection & ref_2nd_selection

            selected_event_numbers = event_board_counts[event_selection_col].index
            tmp_df = filtered_df[filtered_df['evt'].isin(selected_event_numbers)]
            tmp_df.reset_index(inplace=True, drop=True)

            if idx > 0:
                tmp_df['evt'] += last_evt
            last_evt += np.uint64(tmp_df['evt'].nunique())

            ## Selecting good hits
            tdc_cuts = {}
            if ignore_board_ids is None:
                ids_to_loop = [0, 1, 2, 3]
            else:
                ids_to_loop = set([0, 1, 2, 3])-set(ignore_board_ids)

            print(tmp_df.head())
            for idx in ids_to_loop:
                # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
                if idx == 0:
                    tdc_cuts[idx] = [tmp_df.loc[tmp_df['board'] == idx]['cal'].mode()[0]-50, tmp_df.loc[tmp_df['board'] == idx]['cal'].mode()[0]+50,  100, 500, 50, 250]
                elif idx == ref_id:
                    tdc_cuts[idx] = [tmp_df.loc[tmp_df['board'] == idx]['cal'].mode()[0]-50, tmp_df.loc[tmp_df['board'] == idx]['cal'].mode()[0]+50,  0, 1100, 50, 250]
                else:
                    tdc_cuts[idx] = [tmp_df.loc[tmp_df['board'] == idx]['cal'].mode()[0]-50, tmp_df.loc[tmp_df['board'] == idx]['cal'].mode()[0]+50,  0, 1100, 0, 600]

            filtered_df = tdc_event_selection(tmp_df, tdc_cuts_dict=tdc_cuts)
            del tmp_df

            if filtered_df.empty:
                num_failed_files += 1
                continue

            event_board_counts = filtered_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
            event_selection_col = None

            if red_2nd_id == -1:
                trig_selection = (event_board_counts[0] == 1)
                ref_selection = (event_board_counts[ref_id] == 1)
                event_selection_col = trig_selection & ref_selection
            else:
                trig_selection = (event_board_counts[0] == 1)
                ref_selection = (event_board_counts[ref_id] == 1)
                ref_2nd_selection = (event_board_counts[red_2nd_id] == 1)
                event_selection_col = trig_selection & ref_selection & ref_2nd_selection

            selected_event_numbers = event_board_counts[event_selection_col].index
            selected_subset_df = filtered_df[filtered_df['evt'].isin(selected_event_numbers)]
            selected_subset_df.reset_index(inplace=True, drop=True)

            dataframes.append(selected_subset_df)
            del event_board_counts, selected_event_numbers, selected_subset_df, event_selection_col

        df = pd.concat(dataframes)
        df.reset_index(inplace=True, drop=True)
        del dataframes

        single_filtered_df = singlehit_event_clear(df, ignore_boards=ignore_board_ids)
        pivot_data_df = making_pivot(single_filtered_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot'}), ignore_boards=ignore_board_ids)
        del single_filtered_df

        min_hit_counter = minimum_number_of_tracks*(len(files)-num_failed_files)/len(input_files)
        combinations_df = pivot_data_df.groupby(group_for_pivot).count()
        combinations_df['count'] = combinations_df['toa_0']
        combinations_df.drop(drop_for_pivot, axis=1, inplace=True)
        track_df = combinations_df.loc[combinations_df['count'] > min_hit_counter]
        track_df.reset_index(inplace=True)
        del pivot_data_df, combinations_df

        if red_2nd_id == -1:
            row_delta_TR = np.abs(track_df['row_0'] - track_df[f'row_{ref_id}']) <= 1
            row_delta_TD = np.abs(track_df['row_0'] - track_df[f'row_{dut_id}']) <= 1
            col_delta_TR = np.abs(track_df['col_0'] - track_df[f'col_{ref_id}']) <= 1
            col_delta_TD = np.abs(track_df['col_0'] - track_df[f'col_{dut_id}']) <= 1

            track_condition = (row_delta_TR) & (col_delta_TR) & (row_delta_TD) & (col_delta_TD)

        else:
            row_delta_TR = np.abs(track_df['row_0'] - track_df[f'row_{ref_id}']) <= 1
            row_delta_TR2 = np.abs(track_df['row_0'] - track_df[f'row_{red_2nd_id}']) <= 1
            row_delta_TD = np.abs(track_df['row_0'] - track_df[f'row_{dut_id}']) <= 1
            col_delta_TR = np.abs(track_df['col_0'] - track_df[f'col_{ref_id}']) <= 1
            col_delta_TR2 = np.abs(track_df['col_0'] - track_df[f'col_{red_2nd_id}']) <= 1
            col_delta_TD = np.abs(track_df['col_0'] - track_df[f'col_{dut_id}']) <= 1

            track_condition = (row_delta_TR) & (col_delta_TR) & (row_delta_TD) & (col_delta_TD) & (row_delta_TR2) & (col_delta_TR2)

        track_df = track_df[track_condition]

        final_list.append(track_df)
        del track_condition, track_df

    final_df = pd.concat(final_list)

    final_df.drop(columns=['count'], inplace=True)
    final_df = final_df.drop_duplicates(subset=columns_want_to_group, keep='first')
    final_df.to_csv(f'{outfile_name}.csv', index=False)


## --------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                prog='finding tracks',
                description='find track candidates!',
            )

    parser.add_argument(
        '-p',
        '--path',
        metavar = 'PATH',
        type = str,
        help = 'path to directory including feather files',
        required = True,
        dest = 'path',
    )

    parser.add_argument(
        '-o',
        '--outfilename',
        metavar = 'NAME',
        type = str,
        help = 'name for output csv file',
        required = True,
        dest = 'outfilename',
    )

    parser.add_argument(
        '-i',
        '--iteration',
        metavar = 'ITERATION',
        type = int,
        help = 'Number of iteration to find tracks',
        default = 10,
        dest = 'iteration',
    )

    parser.add_argument(
        '-s',
        '--sampling',
        metavar = 'SAMPLING',
        type = int,
        help = 'Random sampling fraction',
        default = 20,
        dest = 'sampling',
    )

    parser.add_argument(
        '-m',
        '--minimum',
        metavar = 'NUM',
        type = int,
        help = 'Minimum number of tracks for selection',
        default = 1000,
        dest = 'track',
    )

    parser.add_argument(
        '--refID',
        metavar = 'ID',
        type = int,
        help = 'reference board ID',
        default = 3,
        dest = 'refID',
    )

    parser.add_argument(
        '--dutID',
        metavar = 'ID',
        type = int,
        help = 'DUT board ID',
        default = 1,
        dest = 'dutID',
    )

    parser.add_argument(
        '--ignoreID',
        metavar = 'ID',
        type = int,
        help = 'board ID be ignored',
        default = 2,
        dest = 'ignoreID',
    )

    parser.add_argument(
        '--four_board',
        action = 'store_true',
        help = 'data will be selected based on 4-board combination',
        dest = 'four_board',
    )

    args = parser.parse_args()

    input_files = list(Path(f'{args.path}').glob('*/*feather'))
    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']



    if not args.four_board:

        list_of_ignore_boards = [args.ignoreID]
        columns_want_to_drop = [f'toa_{i}' for i in set([0, 1, 2, 3])-set(list_of_ignore_boards)]

        columns_want_to_group = []
        for i in set([0, 1, 2, 3])-set(list_of_ignore_boards):
            columns_want_to_group.append(f'row_{i}')
            columns_want_to_group.append(f'col_{i}')

        print('*************** 3-board track finding ********************')
        print(f'Output csv file name is: {args.outfilename}')
        print(f'Number track finding iteration: {args.iteration}')
        print(f'Sampling fraction is: {args.sampling*0.01}')
        print(f'Minimum number of track for selection is: {args.track}')
        print(f'Reference board ID is: {args.refID}')
        print(f'Device Under Test board ID is: {args.dutID}')
        print(f'Board ID {args.ignoreID} will be ignored')
        print('*************** 3-board track finding ********************')

        finding_tracks(input_files=input_files, columns_to_read=columns_to_read, ignore_board_ids=list_of_ignore_boards,
                    outfile_name=args.outfilename, iteration=args.iteration, sampling_fraction=args.sampling, minimum_number_of_tracks=args.track,
                    dut_id=args.dutID, ref_id=args.refID, group_for_pivot=columns_want_to_group, drop_for_pivot=columns_want_to_drop)
    else:

        columns_want_to_drop = [f'toa_{i}' for i in [0,1,2,3]]

        columns_want_to_group = []
        for i in [0, 1, 2, 3]:
            columns_want_to_group.append(f'row_{i}')
            columns_want_to_group.append(f'col_{i}')

        print('*************** 4-board track finding ********************')
        print(f'Output csv file name is: {args.outfilename}')
        print(f'Number track finding iteration: {args.iteration}')
        print(f'Sampling fraction is: {args.sampling*0.01}')
        print(f'Minimum number of track for selection is: {args.track}')
        print(f'Reference board ID is: {args.refID}')
        print(f'2nd reference board ID is: {args.ignoreID}')
        print(f'Device Under Test board ID is: {args.dutID}')
        print('*************** 4-board track finding ********************')

        finding_tracks(input_files=input_files, columns_to_read=columns_to_read, ignore_board_ids=None,
                    outfile_name=args.outfilename, iteration=args.iteration, sampling_fraction=args.sampling, minimum_number_of_tracks=args.track,
                    dut_id=args.dutID, ref_id=args.refID, group_for_pivot=columns_want_to_group, drop_for_pivot=columns_want_to_drop, red_2nd_id=args.ignoreID)
