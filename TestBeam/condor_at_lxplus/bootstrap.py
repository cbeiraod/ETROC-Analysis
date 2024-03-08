import beamtest_analysis_helper as helper
import pandas as pd
import numpy as np
from collections import defaultdict

def bootstrap(
        input_df: pd.DataFrame,
        board_to_analyze: list[int],
        iteration: int = 100,
        sampling_fraction: float = 0.75,
    ):

    resolution_from_bootstrap = defaultdict(list)

    for iloop in range(iteration):

        tdc_filtered_df = input_df

        n = int(sampling_fraction*tdc_filtered_df.shape[0])
        indices = np.random.choice(tdc_filtered_df['evt'].unique(), n, replace=False)
        tdc_filtered_df = tdc_filtered_df.loc[tdc_filtered_df['evt'].isin(indices)]

        if tdc_filtered_df.shape[0] < iteration/(3.*(1-sampling_fraction)):
            print('Warning!! Sampling size is too small. Skipping this track')
            break

        d = {
            'evt': tdc_filtered_df['evt'].unique(),
        }

        for idx in board_to_analyze:
            bins = 3.125/tdc_filtered_df['cal'][idx].mean()
            d[f'toa_b{str(idx)}'] = (12.5 - tdc_filtered_df['toa'][idx] * bins)*1e3
            d[f'tot_b{str(idx)}'] = ((2*tdc_filtered_df['tot'][idx] - np.floor(tdc_filtered_df['tot'][idx]/32)) * bins)*1e3

        df_in_time = pd.DataFrame(data=d)
        del d, tdc_filtered_df

        if(len(board_to_analyze)==3):
            corr_toas = helper.three_board_iterative_timewalk_correction(df_in_time, 5, 3, board_list=board_to_analyze)
        elif(len(board_to_analyze)==4):
            corr_toas = helper.four_board_iterative_timewalk_correction(df_in_time, 5, 3)
        else:
            print("You have less than 3 boards to analyze")
            break

        diffs = {}
        for board_a in board_to_analyze:
            for board_b in board_to_analyze:
                if board_b <= board_a:
                    continue
                name = f"{board_a}{board_b}"
                diffs[name] = np.asarray(corr_toas[f'toa_b{board_a}'] - corr_toas[f'toa_b{board_b}'])

        try:
            fit_params = {}
            for key in diffs.keys():
                params = helper.fwhm_based_on_gaussian_mixture_model(diffs[key], n_components=2, each_component=False, plotting=False)
                fit_params[key] = float(params[0]/2.355)

            del params, diffs, corr_toas

            resolutions = helper.return_resolution_three_board_fromFWHM(fit_params, var=list(fit_params.keys()), board_list=board_to_analyze)

            if any(np.isnan(val) for key, val in resolutions.items()):
                print('fit results is not good, skipping this iteration')
                continue

            for key in resolutions.keys():
                resolution_from_bootstrap[key].append(resolutions[key])

        except Exception as inst:
            print(inst)
            del diffs, corr_toas

    else:
        print('Track is not validate for bootstrapping')

    resolution_from_bootstrap_df = pd.DataFrame(resolution_from_bootstrap)

    return resolution_from_bootstrap_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                prog='bootstrap',
                description='find time resolution!',
            )

    parser.add_argument(
        '-f',
        '--file',
        metavar = 'PATH',
        type = str,
        help = 'pickle file with tdc data based on selected track',
        required = True,
        dest = 'file',
    )

    parser.add_argument(
        '-i',
        '--iteration',
        metavar = 'ITERATION',
        type = int,
        help = 'Number of bootstrapping',
        default = 100,
        dest = 'iteration',
    )

    parser.add_argument(
        '-s',
        '--sampling',
        metavar = 'SAMPLING',
        type = float,
        help = 'Random sampling fraction',
        default = 0.75,
        dest = 'sampling',
    )

    parser.add_argument(
        '--csv',
        action = 'store_true',
        help = 'If set, save final dataframe in csv format',
        dest = 'do_csv',
    )

    args = parser.parse_args()

    fname = args.file.split('.')[0]
    df = pd.read_pickle(args.file)
    board_ids = df.columns.get_level_values('board').unique().tolist()

    resolution_df = bootstrap(input_df=df, board_to_analyze=board_ids, iteration=args.iteration, sampling_fraction=args.sampling)

    if not args.do_csv:
        resolution_df.to_pickle(fname+'_resolution.pkl')
    else:
        resolution_df.to_csv(fname+'_resolution.csv', index=False)
    pass