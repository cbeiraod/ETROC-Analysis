import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

## --------------------------------------
def pixel_filter(
        input_df: pd.DataFrame,
        pixel_dict: dict,
        filter_by_area: bool = False,
        pixel_buffer: int = 2,
    ):

    masks = {}
    if filter_by_area:
        for board, pix in pixel_dict.items():
            mask = (
                (input_df['board'] == board)
                & (input_df['row'] >= pix[0]-pixel_buffer) & (input_df['row'] <= pix[0]+pixel_buffer)
                & (input_df['col'] >= pix[1]-pixel_buffer) & (input_df['col'] <= pix[1]+pixel_buffer)
            )
            masks[board] = mask
    else:
        for board, pix in pixel_dict.items():
            mask = (
                (input_df['board'] == board) & (input_df['row'] == pix[0]) & (input_df['col'] == pix[1])
            )
            masks[board] = mask

    # Combine the masks using logical OR
    combined_mask = pd.concat(masks, axis=1).any(axis=1)

    # Apply the combined mask to the DataFrame
    filtered = input_df[combined_mask].reset_index(drop=True)
    return filtered


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
def find_toa_linear_fit_params(
        input_df: pd.DataFrame,
        ref_id: int,
    ):

    event_board_counts = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None

    trig_selection = (event_board_counts[0] == 1)
    ref_selection = (event_board_counts[ref_id] == 1)
    event_selection_col = trig_selection & ref_selection

    sub_df = input_df.loc[input_df['evt'].isin(event_board_counts[event_selection_col].index)]
    sub_df.reset_index(inplace=True, drop=True)
    del event_board_counts, event_selection_col, trig_selection, ref_selection

    ## Selecting good hits
    tdc_cuts = {}
    for idx in [0, ref_id]:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        if idx == 0:
            tdc_cuts[idx] = [sub_df.loc[sub_df['board'] == idx]['cal'].mode()[0]-50, sub_df.loc[sub_df['board'] == idx]['cal'].mode()[0]+50,  100, 500, 0, 600]
        else:
            tdc_cuts[idx] = [sub_df.loc[sub_df['board'] == idx]['cal'].mode()[0]-50, sub_df.loc[sub_df['board'] == idx]['cal'].mode()[0]+50,  0, 1100, 0, 600]

    filtered_df = tdc_event_selection(sub_df, tdc_cuts_dict=tdc_cuts)
    del sub_df

    params = np.polyfit(filtered_df.loc[filtered_df['board'] == 0]['toa'].reset_index(drop=True), filtered_df.loc[filtered_df['board'] == ref_id]['toa'].reset_index(drop=True), 1)
    del filtered_df

    return params


## --------------------------------------
def data_3board_selection_by_track(
        input_df: pd.DataFrame,
        pix_dict: dict,
        dut_id: int,
        ref_id: int,
        board_to_analyze: list[int],
        trig_ref_params: np.array,
        trig_dut_params: np.array,
        toa_cuts: list[int],
    ):

    track_tmp_df = pixel_filter(input_df, pix_dict, filter_by_area=True, pixel_buffer=2)

    event_board_counts = track_tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[0] == 1) & (event_board_counts[dut_id] == 1) & (event_board_counts[ref_id] == 1)

    isolated_df = track_tmp_df.loc[track_tmp_df['evt'].isin(event_board_counts[event_selection_col].index)]

    track_tmp_df = pixel_filter(isolated_df, pix_dict, filter_by_area=False)

    ## Selecting good hits with TDC cuts
    tdc_cuts = {}
    for idx in board_to_analyze:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        if idx == 0:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    toa_cuts[0], toa_cuts[1], 0, 600]
        else:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    0, 1100, 0, 600]

    track_tmp_df = tdc_event_selection(track_tmp_df, tdc_cuts_dict=tdc_cuts)

    x = track_tmp_df.loc[track_tmp_df['board'] == 0]['toa'].reset_index(drop=True)
    y = track_tmp_df.loc[track_tmp_df['board'] == ref_id]['toa'].reset_index(drop=True)

    trig_ref_distance = (x*trig_ref_params[0] - y + trig_ref_params[1])/(np.sqrt(trig_ref_params[0]**2 + 1))

    x = track_tmp_df.loc[track_tmp_df['board'] == 0]['toa'].reset_index(drop=True)
    y = track_tmp_df.loc[track_tmp_df['board'] == dut_id]['toa'].reset_index(drop=True)

    trig_dut_distance = (x*trig_dut_params[0] - y + trig_dut_params[1])/(np.sqrt(trig_dut_params[0]**2 + 1))

    pivot_table = track_tmp_df.pivot(index=["evt"], columns=["board"], values=["row", "col", "toa", "tot", "cal"])
    pivot_table = pivot_table.reset_index()
    pivot_table = pivot_table[(trig_ref_distance.abs() < 3*np.std(trig_ref_distance)) & (trig_dut_distance.abs() < 3*np.std(trig_dut_distance))]
    pivot_table = pivot_table.reset_index(drop=True)

    ## Pivot Table to make tracks
    return pivot_table


## --------------------------------------
def data_4board_selection_by_track(
        input_df: pd.DataFrame,
        pix_dict: dict,
        dut_id: int,
        ref_id: int,
        ref_2nd_id: int,
        board_to_analyze: list[int],
        trig_ref_params: np.array,
        trig_dut_params: np.array,
        toa_cuts: list[int],
    ):

    track_tmp_df = pixel_filter(input_df, pix_dict, filter_by_area=True, pixel_buffer=2)

    event_board_counts = track_tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[0] == 1) & (event_board_counts[dut_id] == 1) & (event_board_counts[ref_id] == 1) & (event_board_counts[ref_2nd_id] == 1)

    isolated_df = track_tmp_df.loc[track_tmp_df['evt'].isin(event_board_counts[event_selection_col].index)]

    track_tmp_df = pixel_filter(isolated_df, pix_dict, filter_by_area=False)

    board_to_analyze = list(set(board_to_analyze) - set([ref_2nd_id]))

    ## Selecting good hits with TDC cuts
    tdc_cuts = {}
    for idx in board_to_analyze:
        # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
        if idx == 0:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    toa_cuts[0], toa_cuts[1], 0, 600]
        else:
            tdc_cuts[idx] = [track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]-3, track_tmp_df.loc[track_tmp_df['board'] == idx]['cal'].mode()[0]+3,
                    0, 1100, 0, 600]

    track_tmp_df = tdc_event_selection(track_tmp_df, tdc_cuts_dict=tdc_cuts)

    x = track_tmp_df.loc[track_tmp_df['board'] == 0]['toa'].reset_index(drop=True)
    y = track_tmp_df.loc[track_tmp_df['board'] == ref_id]['toa'].reset_index(drop=True)

    trig_ref_distance = (x*trig_ref_params[0] - y + trig_ref_params[1])/(np.sqrt(trig_ref_params[0]**2 + 1))

    x = track_tmp_df.loc[track_tmp_df['board'] == 0]['toa'].reset_index(drop=True)
    y = track_tmp_df.loc[track_tmp_df['board'] == dut_id]['toa'].reset_index(drop=True)

    trig_dut_distance = (x*trig_dut_params[0] - y + trig_dut_params[1])/(np.sqrt(trig_dut_params[0]**2 + 1))

    pivot_table = track_tmp_df.pivot(index=["evt"], columns=["board"], values=["row", "col", "toa", "tot", "cal"])
    pivot_table = pivot_table.reset_index()
    pivot_table = pivot_table[(trig_ref_distance.abs() < 3*np.std(trig_ref_distance)) & (trig_dut_distance.abs() < 3*np.std(trig_dut_distance))]
    pivot_table = pivot_table.reset_index(drop=True)

    ## Pivot Table to make tracks
    return pivot_table



## --------------------------------------
if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
                prog='data selection',
                description='save data by track',
            )

    parser.add_argument(
        '-f',
        '--inputfile',
        metavar = 'NAME',
        type = str,
        help = 'input feather file',
        required = True,
        dest = 'inputfile',
    )

    parser.add_argument(
        '-r',
        '--runinfo',
        metavar = 'NAME',
        type = str,
        help = 'run information for the output name',
        required = True,
        dest = 'runinfo',
    )

    parser.add_argument(
        '-t',
        '--track',
        metavar = 'NAME',
        type = str,
        help = 'csv file including track candidates',
        required = True,
        dest = 'track',
    )

    parser.add_argument(
        '--refID',
        metavar = 'ID',
        type = int,
        help = 'reference board ID',
        required = True,
        dest = 'refID',
    )

    parser.add_argument(
        '--dutID',
        metavar = 'ID',
        type = int,
        help = 'DUT board ID',
        required = True,
        dest = 'dutID',
    )

    parser.add_argument(
        '--ignoreID',
        metavar = 'ID',
        type = int,
        help = 'board ID be ignored',
        required = True,
        dest = 'ignoreID',
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

    args = parser.parse_args()

    board_ids = [0,1,2,3]
    ignore_boards = [args.ignoreID]
    toa_cuts = [args.trigTOALower, args.trigTOAUpper]

    ref_id = args.refID
    dut_id = args.dutID

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']

    run_df = pd.read_feather(args.inputfile, columns=columns_to_read)

    if run_df.empty:
        print('Empty input file!')
        exit(0)

    ### Find parameters for diagonal cut
    ref_params = find_toa_linear_fit_params(run_df, ref_id=ref_id)
    dut_params = find_toa_linear_fit_params(run_df, ref_id=dut_id)

    track_df = pd.read_csv(args.track)

    track_pivots = defaultdict(pd.DataFrame)

    if track_df.shape[1] == 8:
        board_to_analyze = board_ids

        for itrack in tqdm(range(track_df.shape[0])):
            pix_dict = {}
            for idx in board_to_analyze:
                pix_dict[idx] = [track_df.iloc[itrack][f'row_{idx}'], track_df.iloc[itrack][f'col_{idx}']]

            table = data_4board_selection_by_track(input_df=run_df, pix_dict=pix_dict, dut_id=dut_id, ref_id=ref_id, ref_2nd_id=args.ignoreID,
                                                   board_to_analyze=board_to_analyze, trig_ref_params=ref_params, trig_dut_params=dut_params, toa_cuts=toa_cuts)
            track_pivots[itrack] = table
    else:
        board_to_analyze = list(set(board_ids) - set(ignore_boards))

        ### Drop the un-interested board id
        run_df = run_df.loc[~(run_df['board'] == ignore_boards[0])]

        ### Run data selection
        for itrack in tqdm(range(track_df.shape[0])):

            ## Filter only the pixels of interest, dropping other hits on the boards of interest as well as boards not of interest
            pix_dict = {}
            for idx in board_to_analyze:
                pix_dict[idx] = [track_df.iloc[itrack][f'row_{idx}'], track_df.iloc[itrack][f'col_{idx}']]

            table = data_3board_selection_by_track(input_df=run_df, pix_dict=pix_dict, dut_id=dut_id, ref_id=ref_id, board_to_analyze=board_to_analyze,
                                            trig_ref_params=ref_params, trig_dut_params=dut_params, toa_cuts=toa_cuts)
            track_pivots[itrack] = table

    fname = args.inputfile.split('.')[0]
    ### Save python dictionary in pickle format
    with open(f'{args.runinfo}_{fname}.pickle', 'wb') as output:
        pickle.dump(track_pivots, output, protocol=pickle.HIGHEST_PROTOCOL)
