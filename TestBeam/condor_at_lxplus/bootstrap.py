import pandas as pd
import numpy as np
from collections import defaultdict

## --------------------------------------
def three_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    board_list: list,
):

    corr_toas = {}
    corr_b0 = input_df[f'toa_b{board_list[0]}'].values
    corr_b1 = input_df[f'toa_b{board_list[1]}'].values
    corr_b2 = input_df[f'toa_b{board_list[2]}'].values

    del_toa_b0 = (0.5*(input_df[f'toa_b{board_list[1]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[0]}']).values
    del_toa_b1 = (0.5*(input_df[f'toa_b{board_list[0]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[1]}']).values
    del_toa_b2 = (0.5*(input_df[f'toa_b{board_list[1]}'] + input_df[f'toa_b{board_list[2]}']) - input_df[f'toa_b{board_list[0]}']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df[f'tot_b{board_list[0]}'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df[f'tot_b{board_list[1]}'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df[f'tot_b{board_list[2]}'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        corr_b0 = corr_b0 + poly_func_b0(input_df[f'tot_b{board_list[0]}'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df[f'tot_b{board_list[1]}'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df[f'tot_b{board_list[2]}'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b2) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b2) - corr_b1)
        del_toa_b2 = (0.5*(corr_b0 + corr_b1) - corr_b2)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b{board_list[0]}'] = corr_b0
            corr_toas[f'toa_b{board_list[1]}'] = corr_b1
            corr_toas[f'toa_b{board_list[2]}'] = corr_b2

    return corr_toas

## --------------------------------------
def four_board_iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
    ):

    corr_toas = {}
    corr_b0 = input_df['toa_b0'].values
    corr_b1 = input_df['toa_b1'].values
    corr_b2 = input_df['toa_b2'].values
    corr_b3 = input_df['toa_b3'].values

    del_toa_b3 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b3']).values
    del_toa_b2 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b1'] + input_df['toa_b3']) - input_df['toa_b2']).values
    del_toa_b1 = ((1/3)*(input_df['toa_b0'] + input_df['toa_b3'] + input_df['toa_b2']) - input_df['toa_b1']).values
    del_toa_b0 = ((1/3)*(input_df['toa_b3'] + input_df['toa_b1'] + input_df['toa_b2']) - input_df['toa_b0']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df['tot_b0'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df['tot_b1'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b2 = np.polyfit(input_df['tot_b2'].values, del_toa_b2, poly_order)
        poly_func_b2 = np.poly1d(coeff_b2)

        coeff_b3 = np.polyfit(input_df['tot_b3'].values, del_toa_b3, poly_order)
        poly_func_b3 = np.poly1d(coeff_b3)

        corr_b0 = corr_b0 + poly_func_b0(input_df['tot_b0'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df['tot_b1'].values)
        corr_b2 = corr_b2 + poly_func_b2(input_df['tot_b2'].values)
        corr_b3 = corr_b3 + poly_func_b3(input_df['tot_b3'].values)

        del_toa_b3 = ((1/3)*(corr_b0 + corr_b1 + corr_b2) - corr_b3)
        del_toa_b2 = ((1/3)*(corr_b0 + corr_b1 + corr_b3) - corr_b2)
        del_toa_b1 = ((1/3)*(corr_b0 + corr_b3 + corr_b2) - corr_b1)
        del_toa_b0 = ((1/3)*(corr_b3 + corr_b1 + corr_b2) - corr_b0)

        if i == iterative_cnt-1:
            corr_toas[f'toa_b0'] = corr_b0
            corr_toas[f'toa_b1'] = corr_b1
            corr_toas[f'toa_b2'] = corr_b2
            corr_toas[f'toa_b3'] = corr_b3

    return corr_toas

## --------------------------------------
def fwhm_based_on_gaussian_mixture_model(
        input_data: np.array,
        n_components: int = 2,
        each_component: bool = False,
        plotting: bool = False,
    ):

    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt

    x_range = np.linspace(input_data.min(), input_data.max(), 1000).reshape(-1, 1)
    models = GaussianMixture(n_components=n_components).fit(input_data.reshape(-1, 1))

    logprob = models.score_samples(x_range)
    pdf = np.exp(logprob)

    peak_height = np.max(pdf)

    # Find the half-maximum points.
    half_max = peak_height*0.5
    half_max_indices = np.where(pdf >= half_max)[0]

    # Calculate the FWHM.
    fwhm = x_range[half_max_indices[-1]] - x_range[half_max_indices[0]]

    ### GMM - sigma
    # Get the standard deviations (sigma) for each component
    sigma_values = np.sqrt(models.covariances_.diagonal())

    # Get the mixing coefficients and Calculate the combined sigma using a weighted sum
    combined_sigma = np.sqrt(np.sum(models.weights_ * sigma_values**2))

    ### Draw plot
    if each_component:
        # Compute PDF for each component
        responsibilities = models.predict_proba(x_range)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

    if plotting:

        fig, ax = plt.subplots(figsize=(10,10))

        # Plot data histogram
        bins, _, _ = ax.hist(input_data, bins=30, density=True, histtype='stepfilled', alpha=0.4, label='Data')

        # Plot PDF of whole model
        ax.plot(x_range, pdf, '-k', label='Mixture PDF')

        if each_component:
            # Plot PDF of each component
            ax.plot(x_range, pdf_individual, '--', label='Component PDF')

        # Plot
        ax.vlines(x_range[half_max_indices[0]],  ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red', label='FWHM')
        ax.vlines(x_range[half_max_indices[-1]], ymin=0, ymax=np.max(bins)*0.75, lw=1.5, colors='red')

        ax.legend(loc='best', fontsize=14)

    return fwhm, combined_sigma

## --------------------------------------
def return_resolution_three_board_fromFWHM(
        fit_params: dict,
        var: list,
        board_list:list,
    ):

    results = {
        board_list[0]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[1]]**2 - fit_params[var[2]]**2)),
        board_list[1]: np.sqrt(0.5*(fit_params[var[0]]**2 + fit_params[var[2]]**2 - fit_params[var[1]]**2)),
        board_list[2]: np.sqrt(0.5*(fit_params[var[1]]**2 + fit_params[var[2]]**2 - fit_params[var[0]]**2)),
    }

    return results

## --------------------------------------
def return_resolution_four_board_fromFWHM(
        fit_params: dict,
    ):

    results = {
        0: np.sqrt((1/6)*(2*fit_params['01']**2+2*fit_params['02']**2+2*fit_params['03']**2-fit_params['12']**2-fit_params['13']**2-fit_params['23']**2)),
        1: np.sqrt((1/6)*(2*fit_params['01']**2+2*fit_params['12']**2+2*fit_params['13']**2-fit_params['02']**2-fit_params['03']**2-fit_params['23']**2)),
        2: np.sqrt((1/6)*(2*fit_params['02']**2+2*fit_params['12']**2+2*fit_params['23']**2-fit_params['01']**2-fit_params['03']**2-fit_params['13']**2)),
        3: np.sqrt((1/6)*(2*fit_params['03']**2+2*fit_params['13']**2+2*fit_params['23']**2-fit_params['01']**2-fit_params['02']**2-fit_params['12']**2)),
    }

    return results

## --------------------------------------
def bootstrap(
        input_df: pd.DataFrame,
        board_to_analyze: list[int],
        iteration: int = 100,
        sampling_fraction: int = 75,
    ):

    resolution_from_bootstrap = defaultdict(list)
    random_sampling_fraction = sampling_fraction*0.01

    for iloop in range(iteration):

        tdc_filtered_df = input_df.reset_index()

        n = int(random_sampling_fraction*tdc_filtered_df.shape[0])
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
            corr_toas = three_board_iterative_timewalk_correction(df_in_time, 5, 3, board_list=board_to_analyze)
        elif(len(board_to_analyze)==4):
            corr_toas = four_board_iterative_timewalk_correction(df_in_time, 5, 3)
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
                params = fwhm_based_on_gaussian_mixture_model(diffs[key], n_components=2, each_component=False, plotting=False)
                fit_params[key] = float(params[0]/2.355)

            del params, diffs, corr_toas

            if(len(board_to_analyze)==3):
                resolutions = return_resolution_three_board_fromFWHM(fit_params, var=list(fit_params.keys()), board_list=board_to_analyze)
            elif(len(board_to_analyze)==4):
                resolutions = return_resolution_four_board_fromFWHM(fit_params)
            else:
                print("You have less than 3 boards to analyze")
                break

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
        type = int,
        help = 'Random sampling fraction',
        default = 75,
        dest = 'sampling',
    )

    parser.add_argument(
        '--csv',
        action = 'store_true',
        help = 'If set, save final dataframe in csv format',
        dest = 'do_csv',
    )

    args = parser.parse_args()

    output_name = args.file.split('.')[0]
    df = pd.read_pickle(args.file)
    board_ids = df.columns.get_level_values('board').unique().tolist()

    resolution_df = bootstrap(input_df=df, board_to_analyze=board_ids, iteration=args.iteration, sampling_fraction=args.sampling)

    if not args.do_csv:
        resolution_df.to_pickle(f'{output_name}_resolution.pkl')
    else:
        resolution_df.to_csv(f'{output_name}_resolution.csv', index=False)
