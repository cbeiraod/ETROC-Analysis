import pandas as pd
import numpy as np

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
def dut1_hitmap(
        input_df: pd.DataFrame,
        pixel: list[int],
    ):
    input_df = input_df.loc[(input_df['board'] == 0) | (input_df['board'] == 1)]
    event_board_counts = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None

    trig_selection = (event_board_counts[0] == 1)
    ref_selection = (event_board_counts[1] == 1)
    event_selection_col = trig_selection & ref_selection

    selected_subset_df = input_df.loc[input_df['evt'].isin(event_board_counts[event_selection_col].index)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    evts = selected_subset_df[(selected_subset_df['board'] == 0) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])]['evt'].values
    interest_df = selected_subset_df[selected_subset_df['evt'].isin(evts)].reset_index(drop=True)
    pivot_data_df = making_pivot(interest_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot', 'ea', 'bcid', 'l1a_counter'}), ignore_boards=None)

    combinations_df = pivot_data_df.groupby(['row_1', 'col_1']).count()
    combinations_df['count'] = combinations_df['toa_0']
    combinations_df.drop(['toa_0', 'toa_1', 'row_0', 'col_0'], axis=1, inplace=True)
    combinations_df = combinations_df.astype('int64')
    combinations_df.reset_index(inplace=True)

    return combinations_df

## --------------------------------------
def dut2_hitmap(
        input_df: pd.DataFrame,
        pixel: list[int],
    ):

    event_board_counts = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = (event_board_counts[0] == 1) & (event_board_counts[1] == 1) & (event_board_counts[2] == 1) & (event_board_counts[3] == 1)

    selected_subset_df = input_df.loc[input_df['evt'].isin(event_board_counts[event_selection_col].index)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    board1_condition = (selected_subset_df['board'] == 0) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])
    board2_condition = (selected_subset_df['board'] == 1) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])
    board3_condition = (selected_subset_df['board'] == 3) & (selected_subset_df['row'] == pixel[0]) & (selected_subset_df['col'] == pixel[1])

    evts1 = selected_subset_df[board1_condition]['evt'].values
    evts2 = selected_subset_df[board2_condition]['evt'].values
    evts3 = selected_subset_df[board3_condition]['evt'].values

    evts = np.intersect1d(np.intersect1d(evts1, evts2), evts3)

    interest_df = selected_subset_df[selected_subset_df['evt'].isin(evts)].reset_index(drop=True)
    pivot_data_df = making_pivot(interest_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot', 'ea', 'bcid', 'l1a_counter'}), ignore_boards=None)

    combinations_df = pivot_data_df.groupby(['row_2', 'col_2']).count()
    combinations_df['count'] = combinations_df['toa_2']
    combinations_df.drop(['toa_0', 'toa_1', 'toa_2', 'toa_3', 'row_0', 'col_0', 'row_1', 'col_1', 'row_3', 'col_3'], axis=1, inplace=True)
    combinations_df = combinations_df.astype('int64')
    combinations_df.reset_index(inplace=True)

    return combinations_df

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
                prog='bootstrap',
                description='find time resolution!',
            )

    parser.add_argument(
        '-f',
        '--file',
        metavar = 'PATH',
        type = str,
        help = 'feather file',
        required = True,
        dest = 'file',
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
        '--row',
        metavar = 'NUM',
        type = int,
        help = 'Row of pixel',
        default = 8,
        dest = 'row',
    )

    parser.add_argument(
        '--col',
        metavar = 'NUM',
        type = int,
        help = 'Column of pixel',
        default = 11,
        dest = 'col',
    )

    args = parser.parse_args()

    columns_to_read = ['evt', 'board', 'row', 'col', 'toa', 'tot', 'cal']
    run_df = pd.read_feather(args.file, columns=columns_to_read)

    if run_df.empty:
        print('Empty input file!')
        exit(0)

    results = {}

    results['dut1'] = dut1_hitmap(input_df=run_df, pixel=[args.row, args.col])
    results['dut2'] = dut2_hitmap(input_df=run_df, pixel=[args.row, args.col])

    fname = args.file.split('.')[0]
    with open(f'{args.runinfo}_{fname}_hitmap.pickle', 'wb') as output:
        pickle.dump(results, output, protocol=pickle.HIGHEST_PROTOCOL)