import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import pandas as pd
from natsort import natsorted
from glob import glob
import hist
import mplhep as hep
plt.style.use(hep.style.CMS)
import boost_histogram as bh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import copy

## --------------------------------------
def toSingleDataFrame(
        files: list,
        do_blockMix: bool = False,
        do_savedf: bool = False,
    ):
    evt = -1
    previous_bcid = -1
    df_count = 0
    d = {
        'evt': [],
        'board': [],
        'col': [],
        'row': [],
        'toa': [],
        'tot': [],
        'cal': [],
    }

    files = natsorted(files)
    df = pd.DataFrame(d)

    if do_blockMix:
        files = files[1:]

    for ifile in files:
        file_d = copy.deepcopy(d)
        with open(ifile, 'r') as infile:
            for line in infile:
                if line.split(' ')[2] == 'HEADER':
                    current_bcid = line.strip().split(' ')[-1]
                    if current_bcid != previous_bcid or df_count>=3:
                            evt += 1
                            df_count = 0
                    previous_bcid = current_bcid
                    df_count += 1
                elif line.split(' ')[2] == 'DATA':
                    id  = int(line.split(' ')[1])
                    col = int(line.split(' ')[6])
                    row = int(line.split(' ')[8])
                    toa = int(line.split(' ')[10])
                    tot = int(line.split(' ')[12])
                    cal = int(line.split(' ')[14])
                    file_d['evt'].append(evt)
                    file_d['board'].append(id)
                    file_d['row'].append(row)
                    file_d['col'].append(col)
                    file_d['toa'].append(toa)
                    file_d['tot'].append(tot)
                    file_d['cal'].append(cal)
                elif line.split(' ')[2] == 'TRAILER':
                    pass
        if len(file_d['evt']) > 0:
            file_df = pd.DataFrame(file_d)
            df = pd.concat((df, file_df), ignore_index=True)
            del file_df
        del file_d

    ## Under develop
    if do_savedf:
        pass

    return df

## --------------------------------------
def toSingleDataFramePerDirectory(
        root: str,
        path_pattern: str,
    ):

    evt = -1
    previous_bcid = -1
    df_count = 0
    name_pattern = "*translated*.dat"

    dirs = glob(f"{root}/{path_pattern}")
    dirs = natsorted(dirs)
    print(dirs[:3])

    d = {
        'evt': [],
        'board': [],
        'col': [],
        'row': [],
        'toa': [],
        'tot': [],
        'cal': [],
    }

    for dir in dirs:
        df = pd.DataFrame(d)
        name = dir.split('/')[-1]
        files = glob(f"{dir}/{name_pattern}")

        for ifile in files:
            file_d = copy.deepcopy(d)
            with open(ifile, 'r') as infile:
                for line in infile.readlines():
                    if line.split(' ')[2] == 'HEADER':
                        current_bcid = line.strip().split(' ')[-1]
                        if current_bcid != previous_bcid or df_count>=3:
                            evt += 1
                            df_count = 0
                        previous_bcid = current_bcid
                        df_count += 1
                    elif line.split(' ')[2] == 'DATA':
                        id  = int(line.split(' ')[1])
                        col = int(line.split(' ')[6])
                        row = int(line.split(' ')[8])
                        toa = int(line.split(' ')[10])
                        tot = int(line.split(' ')[12])
                        cal = int(line.split(' ')[14])
                        file_d['evt'].append(evt)
                        file_d['board'].append(id)
                        file_d['row'].append(row)
                        file_d['col'].append(col)
                        file_d['toa'].append(toa)
                        file_d['tot'].append(tot)
                        file_d['cal'].append(cal)
                    elif line.split(' ')[2] == 'TRAILER':
                        pass
            if len(file_d['evt']) > 0:
                file_df = pd.DataFrame(file_d)
                df = pd.concat((df, file_df), ignore_index=True)

        df.to_parquet(name+'.pqt', index=False)
        del df

## --------------------------------------
def making_heatmap_byPandas(
        input_df: pd.DataFrame,
        chipLabels: list,
        figtitle: list,
        figtitle_tag: str
    ):
    # Group the DataFrame by 'col,' 'row,' and 'board,' and count the number of hits in each group
    hits_count_by_col_row_board = input_df.groupby(['col', 'row', 'board'])['evt'].count().reset_index()

    # Rename the 'evt' column to 'hits'
    hits_count_by_col_row_board = hits_count_by_col_row_board.rename(columns={'evt': 'hits'})

    for idx, id in enumerate(chipLabels):
        # Create a pivot table to reshape the data for plotting
        pivot_table = hits_count_by_col_row_board[hits_count_by_col_row_board['board'] == int(id)].pivot_table(
            index='row',
            columns='col',
            values='hits',
            fill_value=0  # Fill missing values with 0 (if any)
        )

        if (pivot_table.shape[0] != 16) or (pivot_table.shape[1]!= 16):
            pivot_table = pivot_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
            pivot_table = pivot_table.reindex(columns=np.arange(0,16))
            pivot_table = pivot_table.fillna(-1)

        # Create a heatmap to visualize the count of hits
        fig, ax = plt.subplots(dpi=100, figsize=(20, 20))
        ax.cla()
        im = ax.imshow(pivot_table, cmap="viridis", interpolation="nearest")

        # Add color bar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hits', fontsize=20)

        for i in range(16):
            for j in range(16):
                value = pivot_table.iloc[i, j]
                if value == -1: continue
                text_color = 'black' if value > (pivot_table.values.max() + pivot_table.values.min()) / 2 else 'white'
                text = str("{:.0f}".format(value))
                plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=17)

        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        ax.set_xlabel('Column (col)', fontsize=20)
        ax.set_ylabel('Row (row)', fontsize=20)
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f"{figtitle[idx]}, Heat map {figtitle_tag}", loc="right", size=20)
        ax.tick_params(axis='x', which='both', length=5, labelsize=17)
        ax.tick_params(axis='y', which='both', length=5, labelsize=17)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()

## --------------------------------------
def singlehit_event_clear_func(
        input_df: pd.DataFrame
    ):
    # Group the DataFrame by 'evt' and count unique 'board' values in each group
    unique_board_counts = input_df.groupby('evt')['board'].nunique()

    ## event has two unique board ID
    event_numbers_with_three_unique_boards = unique_board_counts[unique_board_counts == 3].index
    subset_df = input_df[input_df['evt'].isin(event_numbers_with_three_unique_boards)]
    subset_df.reset_index(inplace=True, drop=True)

    ## event has one hit from each board
    event_board_counts = subset_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    selected_event_numbers = event_board_counts[(event_board_counts[0] == 1) & (event_board_counts[1] == 1) & (event_board_counts[3] == 1)].index
    selected_subset_df = subset_df[subset_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)
    del subset_df

    return selected_subset_df

## --------------------------------------
def tdc_event_selection(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict
    ):
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

## --------------------------------------
def pixel_filter(
        input_df: pd.DataFrame,
        pixel_dict: dict
    ):
    # Create boolean masks for each board's filtering criteria
    masks = {}
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
def find_maximum_event_combination(
        input_df: pd.DataFrame,
        board_pixel_info: list,
    ):
    # Step 1: Filter the rows where board is 1, col is 6, and row is 15
    selected_rows = input_df[(input_df['board'] == board_pixel_info[0]) & (input_df['row'] == board_pixel_info[1]) & (input_df['col'] == board_pixel_info[2])]

    # Step 2: Get the unique "evt" values from the selected rows
    unique_evts = selected_rows['evt'].unique()

    # Step 3: Filter rows where board is 0 or 3 and "evt" is in unique_evts
    filtered_rows = input_df[(input_df['board'].isin([0, 3])) & (input_df['evt'].isin(unique_evts))]

    result_df = pd.concat([selected_rows, filtered_rows], ignore_index=True)
    result_df = result_df.sort_values(by="evt")
    result_df.reset_index(inplace=True, drop=True)

    test_group = result_df.groupby(['board', 'row', 'col'])
    count_df = test_group.size().reset_index(name='count')

    row0 = count_df.loc[count_df[count_df['board'] == 0]['count'].idxmax()]
    row3 = count_df.loc[count_df[count_df['board'] == 3]['count'].idxmax()]

    print(f"Board 0, Row: {row0['row']}, Col: {row0['col']}, Count: {row0['count']}")
    print(f"Board 3, Row: {row3['row']}, Col: {row3['col']}, Count: {row3['count']}")

    del selected_rows, unique_evts, filtered_rows, test_group, count_df, row0, row3
    return result_df

## --------------------------------------
def iterative_timewalk_correction(
    input_df: pd.DataFrame,
    iterative_cnt: int,
    poly_order: int,
):

    corr_toas = []
    corr_b0 = input_df['toa_b0'].values
    corr_b1 = input_df['toa_b1'].values
    corr_b3 = input_df['toa_b3'].values

    del_toa_b0 = (0.5*(input_df['toa_b1'] + input_df['toa_b3']) - input_df['toa_b0']).values
    del_toa_b1 = (0.5*(input_df['toa_b0'] + input_df['toa_b3']) - input_df['toa_b1']).values
    del_toa_b3 = (0.5*(input_df['toa_b0'] + input_df['toa_b1']) - input_df['toa_b3']).values

    for i in range(iterative_cnt):
        coeff_b0 = np.polyfit(input_df['tot_b0'].values, del_toa_b0, poly_order)
        poly_func_b0 = np.poly1d(coeff_b0)

        coeff_b1 = np.polyfit(input_df['tot_b1'].values, del_toa_b1, poly_order)
        poly_func_b1 = np.poly1d(coeff_b1)

        coeff_b3 = np.polyfit(input_df['tot_b3'].values, del_toa_b3, poly_order)
        poly_func_b3 = np.poly1d(coeff_b3)

        corr_b0 = corr_b0 + poly_func_b0(input_df['tot_b0'].values)
        corr_b1 = corr_b1 + poly_func_b1(input_df['tot_b1'].values)
        corr_b3 = corr_b3 + poly_func_b3(input_df['tot_b3'].values)

        del_toa_b0 = (0.5*(corr_b1 + corr_b3) - corr_b0)
        del_toa_b1 = (0.5*(corr_b0 + corr_b3) - corr_b1)
        del_toa_b3 = (0.5*(corr_b0 + corr_b1) - corr_b3)

        if i == iterative_cnt-1:
            corr_toas.append(corr_b0)
            corr_toas.append(corr_b1)
            corr_toas.append(corr_b3)

    return corr_toas

## --------------------------------------
def making_3d_heatmap_byPandas(
        input_df: pd.DataFrame,
        chipLabels: list,
        figtitle: list,
        figtitle_tag: str,
    ):
    # Create a 3D subplot

    for idx, id in enumerate(chipLabels):

        # Create the 2D heatmap for the current chip label
        hits_count_by_col_row_board = input_df[input_df['board'] == int(id)].groupby(['col', 'row'])['evt'].count().reset_index()
        hits_count_by_col_row_board = hits_count_by_col_row_board.rename(columns={'evt': 'hits'})
        pivot_table = hits_count_by_col_row_board.pivot_table(index='row', columns='col', values='hits', fill_value=0)

        if pivot_table.shape[1] != 16:
            continue

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for the 3D surface
        x, y = np.meshgrid(np.arange(16), np.arange(16))
        z = pivot_table.values
        dx = dy = 0.75  # Width and depth of the bars

        # Create a 3D surface plot
        ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z).flatten(), dx, dy, z.flatten(), shade=True)

        # Customize the 3D plot settings as needed
        ax.set_xlabel('COL', fontsize=15, labelpad=15)
        ax.set_ylabel('ROW', fontsize=15, labelpad=15)
        ax.set_zlabel('Hits', fontsize=15, labelpad=-35)
        ax.invert_xaxis()
        ticks = range(0, 16)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticks(ticks=range(16), labels=[], minor=True)
        ax.set_yticks(ticks=range(16), labels=[], minor=True)
        ax.tick_params(axis='x', labelsize=8)  # You can adjust the 'pad' value
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        ax.set_title(f"Heat map 3D {figtitle[idx]}", fontsize=16)
        plt.tight_layout()

## --------------------------------------
def return_hist(
        input_df: pd.DataFrame,
        chip_names: list,
        chip_labels: list,
        hist_bins: list = [50, 64, 64]
):
    h = {chip_names[idx]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
        )
    for idx, boardID in enumerate(chip_labels)}
    for idx, boardID in enumerate(chip_labels):
        tmp_df = input_df[input_df['board'] == int(boardID)]
        h[chip_names[idx]].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values)

    return h
## --------------------------------------
def return_resolution_ps(sig_a, err_a, sig_b, err_b, sig_c, err_c):
    res = np.sqrt(0.5)*(np.sqrt(sig_a**2 + sig_b**2 - sig_c**2))
    var_res = (1/4)*(1/res**2)*(((sig_a**2)*(err_a**2))+((sig_b**2)*(err_b**2))+((sig_c**2)*(err_c**2)))
    return res*1e3, np.sqrt(var_res)*1e3

## --------------------------------------
def draw_hist_plot_pull(
    input_hist: hist.Hist,
    fig_title: str,
):
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
    main_ax = fig.add_subplot(grid[0])
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    main_ax_artists, sublot_ax_arists = input_hist.plot_pull(
        "gaus",
        eb_ecolor="steelblue",
        eb_mfc="steelblue",
        eb_mec="steelblue",
        eb_fmt="o",
        eb_ms=6,
        eb_capsize=1,
        eb_capthick=2,
        eb_alpha=0.8,
        fp_c="hotpink",
        fp_ls="-",
        fp_lw=2,
        fp_alpha=0.8,
        bar_fc="royalblue",
        pp_num=3,
        pp_fc="royalblue",
        pp_alpha=0.618,
        pp_ec=None,
        ub_alpha=0.2,
        fit_fmt= r"{name} = {value:.4g} $\pm$ {error:.4g}",
        ax_dict= {"main_ax":main_ax,"pull_ax":subplot_ax},
    )
    hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)
    main_ax.set_title(f'{fig_title}', loc="right", size=25)

## --------------------------------------
def lmfit_gaussfit_with_pulls(
        input_data: pd.Series,
        input_hist: hist.Hist,
        std_range_cut: float,
        width_factor: float,
        fig_title: str,
        use_pred_uncert: bool,
    ):

    from lmfit.models import GaussianModel
    from lmfit.lineshapes import gaussian

    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])

    main_ax = fig.add_subplot(grid[0])
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    mod = GaussianModel()

    reduced_data = input_data[(input_data > input_data.mean()-std_range_cut) & (input_data < input_data.mean()+std_range_cut)]

    fit_min = reduced_data.mean()-width_factor*reduced_data.std()
    fit_max = reduced_data.mean()+width_factor*reduced_data.std()
    del reduced_data

    input_hist_peak = input_hist[(fit_min)*1j:(fit_max)*1j]
    centers = input_hist.axes[0].centers
    fit_centers = input_hist_peak.axes[0].centers
    pars = mod.guess(input_hist.values(), x=centers)
    out = mod.fit(input_hist_peak.values(), pars, x=fit_centers, weights=1/np.sqrt(input_hist_peak.values()))

    hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)
    main_ax.set_title(f'{fig_title}', loc="right", size=25)
    main_ax.errorbar(centers, input_hist.values(), np.sqrt(input_hist.variances()),
                    ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                    ms=6, capsize=1, capthick=2, alpha=0.8)
    main_ax.plot(fit_centers, out.best_fit, color="hotpink", ls="-", lw=2, alpha=0.8,
                label=fr"$\mu:{out.params['center'].value:.3f}, \sigma: {abs(out.params['sigma'].value):.3f}$")
    main_ax.set_ylabel('Counts')
    main_ax.set_ylim(-20, None)

    popt = [par for name, par in out.best_values.items()]
    pcov = out.covar

    if np.isfinite(pcov).all():
        n_samples = 100
        vopts = np.random.multivariate_normal(popt, pcov, n_samples)
        sampled_ydata = np.vstack([gaussian(fit_centers, *vopt).T for vopt in vopts])
        model_uncert = np.nanstd(sampled_ydata, axis=0)
    else:
        model_uncert = np.zeros_like(np.sqrt(input_hist.variances()))

    main_ax.fill_between(
        fit_centers,
        out.eval(x=fit_centers) - model_uncert,
        out.eval(x=fit_centers) + model_uncert,
        color="hotpink",
        alpha=0.2,
        label='Uncertainty'
    )
    main_ax.legend(fontsize=20, loc='upper right')

    ### Calculate pull
    if use_pred_uncert:
        pulls = (input_hist.values() - out.eval(x=centers))/np.sqrt(out.eval(x=centers))
    else:
        pulls = (input_hist.values() - out.eval(x=centers))/np.sqrt(input_hist.variances())
    pulls[np.isnan(pulls) | np.isinf(pulls)] = 0

    left_edge = centers[0]
    right_edge = centers[-1]

    # Pull: plot the pulls using Matplotlib bar method
    width = (right_edge - left_edge) / len(pulls)

    subplot_ax.axvline(fit_centers[0], c='red', lw=2)
    subplot_ax.axvline(fit_centers[-1], c='red', lw=2)
    subplot_ax.axhline(1, c='black', lw=0.75)
    subplot_ax.axhline(0, c='black', lw=1.2)
    subplot_ax.axhline(-1, c='black', lw=0.75)
    subplot_ax.bar(centers, pulls, width=width, fc='royalblue')
    subplot_ax.set_ylim(-2, 2)
    subplot_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1])
    subplot_ax.set_xlabel(r'Time Walk Corrected $\Delta$TOA [ns]')
    subplot_ax.set_ylabel('Pulls', fontsize=20, loc='center')
    subplot_ax.minorticks_off()

    plt.tight_layout()

    return [out.params['sigma'].value, out.params['sigma'].stderr]

## --------------------------------------
def make_pix_inclusive_plots(
        input_hist: hist.Hist,
        chip_name,
        chip_figname,
        chip_figtitle,
        fig_path,
        save: bool = False,
        show: bool = False,
        tag: str = '',
        title_tag: str = '',
        slide_friendly: bool = False,
    ):

    if not slide_friendly:
        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, CAL{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("CAL")[:].plot1d(ax=ax, lw=2)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_CAL_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, TOT{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOT")[:].plot1d(ax=ax, lw=2)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOT_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, TOA{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOA")[:].plot1d(ax=ax, lw=2)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOA_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,20))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, TOA v TOT{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOA","TOT")[::2j,::2j].plot2d(ax=ax)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOA_TOT_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

    else:
        fig = plt.figure(dpi=100, figsize=(30,13))
        gs = fig.add_gridspec(2,2)

        for i, plot_info in enumerate(gs):
            ax = fig.add_subplot(plot_info)
            hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=20)
            if i == 0:
                ax.set_title(f"{chip_figtitle}, CAL{title_tag}", loc="right", size=15)
                input_hist[chip_name].project("CAL")[:].plot1d(ax=ax, lw=2)
            elif i == 1:
                ax.set_title(f"{chip_figtitle}, TOT{title_tag}", loc="right", size=15)
                input_hist[chip_name].project("TOT")[:].plot1d(ax=ax, lw=2)
            elif i == 2:
                ax.set_title(f"{chip_figtitle}, TOA{title_tag}", loc="right", size=15)
                input_hist[chip_name].project("TOA")[:].plot1d(ax=ax, lw=2)
            elif i == 3:
                ax.set_title(f"{chip_figtitle}, TOA v TOT{title_tag}", loc="right", size=14)
                input_hist[chip_name].project("TOA","TOT")[::2j,::2j].plot2d(ax=ax)

        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/combined_TDC_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

## --------------------------------------
def event_display_withPandas(
        input_df: pd.DataFrame,
    ):
    # Loop over unique evt values
    unique_evts = input_df['evt'].unique()

    for cnt, evt in enumerate(unique_evts):
        if cnt > 15: break

        selected_subset_df = input_df[input_df['evt'] == evt]

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.grid(False)

        # Create a meshgrid for the contourf
        xx, yy = np.meshgrid(np.arange(16), np.arange(16))

        for board_value in [0, 3]:
            board_subset_df = selected_subset_df[selected_subset_df['board'] == board_value]

            # Create a 16x16 binary grid with 1s where "cal" exists, 0s otherwise
            cal_grid = np.zeros((16, 16))
            for _, row in board_subset_df.iterrows():
                cal_grid[row['row'], row['col']] = 1

            # Plot the contourf for this board value
            ax1.contourf(xx, yy, cal_grid, 100, zdir='z', offset=board_value, alpha=0.15, cmap="plasma")

        ax1.set_zlim((0., 3.0))  # Adjust z-axis limit based on your board values
        ax1.set_xlabel('COL', fontsize=15, labelpad=15)
        ax1.set_ylabel('ROW', fontsize=15, labelpad=15)
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ticks = range(0, 16)
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.set_xticks(ticks=range(16), labels=[], minor=True)
        ax1.set_yticks(ticks=range(16), labels=[], minor=True)
        ax1.set_zticks(ticks=[0, 1, 3], labels=["Bottom", "Middle", "Top"])
        ax1.tick_params(axis='x', labelsize=8)  # You can adjust the 'pad' value
        ax1.tick_params(axis='y', labelsize=8)
        ax1.tick_params(axis='z', labelsize=8)
        ax1.grid(visible=False, axis='z')
        ax1.grid(visible=True, which='major', axis='x')
        ax1.grid(visible=True, which='major', axis='y')
        plt.title(f'Event {evt}')

        del xx, yy, fig, ax1  # Clear the figure to avoid overlapping plots

## --------------------------------------
def poly2D(max_order, x, y, *args):
    if max_order < 0:
        raise RuntimeError("The polynomial order must be non-negative")

    ret_val = None

    linear_idx = 0
    for i in range(max_order+1):
        for j in range(max_order - i + 1):
            this_val = args[linear_idx] * x**j * y**i
            linear_idx += 1

            if ret_val is None:
                ret_val = this_val
            else:
                ret_val += this_val

    return ret_val

## --------------------------------------
def poly3D(max_order, x, y, z, *args):
    if max_order < 0:
        raise RuntimeError("The polynomial order must be non-negative")

    ret_val = None

    linear_idx = 0
    for i in range(max_order+1):
        for j in range(max_order - i + 1):
            for k in range(max_order - i - j + 1):
                this_val = args[linear_idx] * x**k * y**j + z**i
                linear_idx += 1

                if ret_val is None:
                    ret_val = this_val
                else:
                    ret_val += this_val

    return ret_val


## --------------------------------------
def making_pivot(
        input_df: pd.DataFrame,
        index: str,
        columns: str,
        drop_columns: tuple,
    ):
        pivot_data_df = input_df.pivot(
        index = index,
        columns = columns,
        values = list(set(input_df.columns) - drop_columns),
        )
        pivot_data_df.columns = ["{}_{}".format(x, y) for x, y in pivot_data_df.columns]

        return pivot_data_df

## --------------------------------------
def making_scatter_with_plotly(
        input_df: pd.DataFrame,
        output_name: str,
    ):
    import plotly.express as px
    fig = px.scatter_matrix(
        input_df,
        dimensions=input_df.columns,
        # labels = labels,
        # color=color_column,
        # title = "Scatter plot comparing variables for each board<br><sup>Run: {}{}</sup>".format(run_name, extra_title),
        opacity = 0.2,
    )

    ## Delete half of un-needed plots
    fig.update_traces(
        diagonal_visible = False,
        showupperhalf=False,
        marker = {'size': 3}
    )

    for k in range(len(fig.data)):
        fig.data[k].update(
            selected = dict(
            marker = dict(
                #opacity = 1,
                #color = 'blue',
                )
            ),
            unselected = dict(
                marker = dict(
                    #opacity = 0.1,
                    color="grey"
                    )
                ),
            )

    fig.write_html(
        f'{output_name}.html',
        full_html = False,
        include_plotlyjs = 'cdn',
    )



## --------------------------------------
# def sort_filter(group):
#     return group.sort_values(by=['board'], ascending=True)

# def distance_filter(group, distance):
#     board0_row = group[(group["board"] == 0)]
#     board3_row = group[(group["board"] == 3)]
#     board0_col = group[(group["board"] == 0)]
#     board3_col = group[(group["board"] == 3)]

#     if not board0_row.empty and not board3_row.empty and not board0_col.empty and not board3_col.empty:
#         row_index_diff = abs(board0_row["row"].values[0] - board3_row["row"].values[0])
#         col_index_diff = abs(board0_col["col"].values[0] - board3_col["col"].values[0])
#         return row_index_diff < distance and col_index_diff < distance
#     else:
#         return False

# tmp_group = selected_subset_df.groupby('evt')
# filtered_simple_group = tmp_group.filter(simple_filter, board=1, row=15, col=6)
# filtered_simple_group.reset_index(inplace=True, drop=True)
# del tmp_group

# grouped = filtered_simple_group.groupby('evt')
# sorted_filtered_simple_group = grouped.apply(sort_filter)
# sorted_filtered_simple_group.reset_index(inplace=True, drop=True)
# sorted_filtered_simple_group
# del grouped

# grouped = sorted_filtered_simple_group.groupby('evt')
# dis_simple_group = grouped.filter(distance_filter, distance=2)
# dis_simple_group

# test_group = dis_simple_group.groupby(['board', 'row', 'col'])
# test = test_group.size().reset_index(name='count')
# test.to_csv('test.csv', index=False)

# del filtered_simple_group,sorted_filtered_simple_group,grouped,dis_simple_group, test_group