import beamtest_analysis_helper as helper
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob
from natsort import natsorted
import hist
import mplhep as hep
hep.style.use('CMS')

# !!!!!!!!!!!!
# It is very important to correctly set the chip name, this value is stored with the data

chip_labels = ["0", "1", "2", "3"]
chip_names = ["ET2_EPIR_Pair1", "ET2_BAR_4", "ET2_BAR_6", "ET2_EPIR_Pair4"]

chip_fignames = chip_names
chip_figtitles = ["(Trigger) ETROC2 EPIR Pair1 HV260V OS:15","ETROC2 W15 2-3 HV260V OS:15", "ETROC2 W15 4-3 HV260V OS:15", "ETROC2 EPIR Pair4 HV260V OS:15"]

today = datetime.date.today().isoformat()
fig_outdir = Path('../../ETROC-figures')
fig_outdir = fig_outdir / (today + '_Array_Test_Results')
fig_outdir.mkdir(exist_ok=True)
fig_path = str(fig_outdir)


files = glob('./desy_TB_run2/filtered/Run_2_loop_*_filtered.feather')
files = natsorted(files)
print(files)

dataframes = []

for ifile in files:
    tmp_df = pd.read_feather(ifile)
    if tmp_df.empty:
        continue

    event_board_counts = tmp_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None
    for board in [0, 1, 2, 3]:
        if board not in event_board_counts:
            event_selection_col = 0
            continue
        if event_selection_col is None:
            event_selection_col = (event_board_counts[board] == 1)
        else:
            event_selection_col = event_selection_col & (event_board_counts[board] == 1)
    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = tmp_df[tmp_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    dataframes.append(selected_subset_df)
    del tmp_df, selected_subset_df

df = pd.concat(dataframes)
df.reset_index(inplace=True, drop=True)
del dataframes

single_filtered_df = helper.singlehit_event_clear_func(df)
pivot_data_df = helper.making_pivot(single_filtered_df, 'evt', 'board', set({'board', 'evt', 'cal', 'tot'}))
del single_filtered_df

min_hit_counter = 10500
combinations_df = pivot_data_df.groupby(['row_0', 'col_0', 'row_1', 'col_1', 'row_2', 'col_2', 'row_3', 'col_3', ]).count()
combinations_df['count'] = combinations_df['toa_0']
combinations_df.drop(['toa_0', 'toa_1', 'toa_2', 'toa_3'], axis=1, inplace=True)
track_df = combinations_df.loc[combinations_df['count'] > min_hit_counter]
track_df.reset_index(inplace=True)

del pivot_data_df, combinations_df

@profile
def myfunction(input_df, out_dict):
    for i in range(len(track_df)):

        pix_dict = {
            # board ID: [row, col]
            0: [input_df.iloc[i]['row_0'], input_df.iloc[i]['col_0']],
            1: [input_df.iloc[i]['row_1'], input_df.iloc[i]['col_1']],
            2: [input_df.iloc[i]['row_2'], input_df.iloc[i]['col_2']],
            3: [input_df.iloc[i]['row_3'], input_df.iloc[i]['col_3']],
        }

        pix_filtered_df = helper.pixel_filter(df, pix_dict)

        tdc_cuts = {
            # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
            0: [pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].std(), 100, 450,    0, 600],
            1: [pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].std(),   0, 1100,   0, 600],
            2: [pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].std(),   0, 1100,   0, 600],
            3: [pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].std(),   0, 1100,   0, 600], # pixel ()
        }

        tdc_filtered_df = helper.tdc_event_selection(pix_filtered_df, tdc_cuts)
        tdc_filtered_df = helper.singlehit_event_clear_func(tdc_filtered_df)
        del pix_filtered_df,

        cal_means = {boardID:{} for boardID in chip_labels}

        for boardID in chip_labels:
            groups = tdc_filtered_df.loc[tdc_filtered_df['board'] == int(boardID)].groupby(['row', 'col'])
            for (row, col), group in groups:
                cal_mean = group['cal'].mean()
                cal_means[boardID][(row, col)] = cal_mean
            del groups

        bin0 = (3.125/cal_means["0"][(pix_dict[0][0], pix_dict[0][1])])
        bin1 = (3.125/cal_means["1"][(pix_dict[1][0], pix_dict[1][1])])
        bin2 = (3.125/cal_means["2"][(pix_dict[2][0], pix_dict[2][1])])
        bin3 = (3.125/cal_means["3"][(pix_dict[3][0], pix_dict[3][1])])

        del pix_dict, tdc_cuts

        toa_in_time_b0 = 12.5 - tdc_filtered_df.loc[tdc_filtered_df['board'] == 0]['toa'] * bin0
        toa_in_time_b1 = 12.5 - tdc_filtered_df.loc[tdc_filtered_df['board'] == 1]['toa'] * bin1
        toa_in_time_b2 = 12.5 - tdc_filtered_df.loc[tdc_filtered_df['board'] == 2]['toa'] * bin2
        toa_in_time_b3 = 12.5 - tdc_filtered_df.loc[tdc_filtered_df['board'] == 3]['toa'] * bin3

        tot_in_time_b0 = (2*tdc_filtered_df.loc[tdc_filtered_df['board'] == 0]['tot'] - np.floor(tdc_filtered_df.loc[tdc_filtered_df['board'] == 0]['tot']/32)) * bin0
        tot_in_time_b1 = (2*tdc_filtered_df.loc[tdc_filtered_df['board'] == 1]['tot'] - np.floor(tdc_filtered_df.loc[tdc_filtered_df['board'] == 1]['tot']/32)) * bin1
        tot_in_time_b2 = (2*tdc_filtered_df.loc[tdc_filtered_df['board'] == 2]['tot'] - np.floor(tdc_filtered_df.loc[tdc_filtered_df['board'] == 2]['tot']/32)) * bin2
        tot_in_time_b3 = (2*tdc_filtered_df.loc[tdc_filtered_df['board'] == 3]['tot'] - np.floor(tdc_filtered_df.loc[tdc_filtered_df['board'] == 3]['tot']/32)) * bin3

        d = {
            'evt': tdc_filtered_df['evt'].unique(),
            'toa_b0': toa_in_time_b0.to_numpy(),
            'tot_b0': tot_in_time_b0.to_numpy(),
            'toa_b1': toa_in_time_b1.to_numpy(),
            'tot_b1': tot_in_time_b1.to_numpy(),
            'toa_b2': toa_in_time_b2.to_numpy(),
            'tot_b2': tot_in_time_b2.to_numpy(),
            'toa_b3': toa_in_time_b3.to_numpy(),
            'tot_b3': tot_in_time_b3.to_numpy(),
        }

        df_in_time = pd.DataFrame(data=d)
        del d, tdc_filtered_df
        del toa_in_time_b0, toa_in_time_b1, toa_in_time_b2, toa_in_time_b3
        del tot_in_time_b0, tot_in_time_b1, tot_in_time_b2, tot_in_time_b3

        corr_toas = helper.four_board_iterative_timewalk_correction(df_in_time, 5, 3)

        tmp_dict = {
            'evt': df_in_time['evt'].values,
            'corr_toa_b0': corr_toas[0],
            'corr_toa_b1': corr_toas[1],
            'corr_toa_b2': corr_toas[2],
            'corr_toa_b3': corr_toas[3],
        }

        df_in_time_corr = pd.DataFrame(tmp_dict)
        del tmp_dict, df_in_time

        diff_b01 = df_in_time_corr['corr_toa_b0'] - df_in_time_corr['corr_toa_b1']
        diff_b02 = df_in_time_corr['corr_toa_b0'] - df_in_time_corr['corr_toa_b2']
        diff_b03 = df_in_time_corr['corr_toa_b0'] - df_in_time_corr['corr_toa_b3']
        diff_b12 = df_in_time_corr['corr_toa_b1'] - df_in_time_corr['corr_toa_b2']
        diff_b13 = df_in_time_corr['corr_toa_b1'] - df_in_time_corr['corr_toa_b3']
        diff_b23 = df_in_time_corr['corr_toa_b2'] - df_in_time_corr['corr_toa_b3']

        dTOA_b01 = hist.Hist(hist.axis.Regular(80, diff_b01.mean().round(2)-0.8, diff_b01.mean().round(2)+0.8, name="TWC_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))
        dTOA_b02 = hist.Hist(hist.axis.Regular(80, diff_b02.mean().round(2)-0.8, diff_b02.mean().round(2)+0.8, name="TWC_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))
        dTOA_b03 = hist.Hist(hist.axis.Regular(80, diff_b03.mean().round(2)-0.8, diff_b03.mean().round(2)+0.8, name="TWC_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))
        dTOA_b12 = hist.Hist(hist.axis.Regular(80, diff_b12.mean().round(2)-0.8, diff_b12.mean().round(2)+0.8, name="TWC_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))
        dTOA_b13 = hist.Hist(hist.axis.Regular(80, diff_b13.mean().round(2)-0.8, diff_b13.mean().round(2)+0.8, name="TWC_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))
        dTOA_b23 = hist.Hist(hist.axis.Regular(80, diff_b23.mean().round(2)-0.8, diff_b23.mean().round(2)+0.8, name="TWC_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))

        dTOA_b01.fill(diff_b01)
        dTOA_b02.fill(diff_b02)
        dTOA_b03.fill(diff_b03)
        dTOA_b12.fill(diff_b12)
        dTOA_b13.fill(diff_b13)
        dTOA_b23.fill(diff_b23)

        del df_in_time_corr

        fit_params_lmfit = {}
        params = helper.lmfit_gaussfit_with_pulls(diff_b01, dTOA_b01, std_range_cut=0.4, width_factor=1.25, fig_title='Board 0 - Board 1',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True)
        fit_params_lmfit['01'] = params
        params = helper.lmfit_gaussfit_with_pulls(diff_b02, dTOA_b02, std_range_cut=0.4, width_factor=1.25, fig_title='Board 0 - Board 2',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True)
        fit_params_lmfit['02'] = params
        params = helper.lmfit_gaussfit_with_pulls(diff_b03, dTOA_b03, std_range_cut=0.4, width_factor=1.25, fig_title='Board 0 - Board 3',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True)
        fit_params_lmfit['03'] = params
        params = helper.lmfit_gaussfit_with_pulls(diff_b12, dTOA_b12, std_range_cut=0.4, width_factor=1.25, fig_title='Board 1 - Board 2',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True)
        fit_params_lmfit['12'] = params
        params = helper.lmfit_gaussfit_with_pulls(diff_b13, dTOA_b13, std_range_cut=0.4, width_factor=1.25, fig_title='Board 1 - Board 3',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True)
        fit_params_lmfit['13'] = params
        params = helper.lmfit_gaussfit_with_pulls(diff_b23, dTOA_b23, std_range_cut=0.4, width_factor=1.25, fig_title='Board 2 - Board 3',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True)
        fit_params_lmfit['23'] = params

        del params
        del dTOA_b01, dTOA_b02, dTOA_b03, dTOA_b12, dTOA_b13, dTOA_b23
        del diff_b01, diff_b02, diff_b03, diff_b12, diff_b13, diff_b23

        res_b0 = helper.return_resolution_four_board(fit_params_lmfit, ['01', '02', '03', '12', '13', '23'])
        res_b1 = helper.return_resolution_four_board(fit_params_lmfit, ['01', '12', '13', '02', '03', '23'])
        res_b2 = helper.return_resolution_four_board(fit_params_lmfit, ['02', '12', '23', '01', '03', '13'])
        res_b3 = helper.return_resolution_four_board(fit_params_lmfit, ['03', '13', '23', '01', '02', '12'])

        del fit_params_lmfit

        out_dict[(input_df.iloc[i]['row_0'], input_df.iloc[i]['col_0'])] = {}
        out_dict[(input_df.iloc[i]['row_0'], input_df.iloc[i]['col_0'])][(input_df.iloc[i]['row_1'], input_df.iloc[i]['col_1'])] = {}
        out_dict[(input_df.iloc[i]['row_0'], input_df.iloc[i]['col_0'])][(input_df.iloc[i]['row_1'], input_df.iloc[i]['col_1'])][(input_df.iloc[i]['row_2'], input_df.iloc[i]['col_2'])] = {}
        out_dict[(input_df.iloc[i]['row_0'], input_df.iloc[i]['col_0'])][(input_df.iloc[i]['row_1'], input_df.iloc[i]['col_1'])][(input_df.iloc[i]['row_2'], input_df.iloc[i]['col_2'])][(input_df.iloc[i]['row_3'], input_df.iloc[i]['col_3'])] = (res_b0, res_b1, res_b2, res_b3)

        del res_b0, res_b1, res_b2, res_b3

output = {}
myfunction(track_df, output)
