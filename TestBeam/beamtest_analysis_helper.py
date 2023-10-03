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

class etroc2_analysis_helper():

    def __init__(
            self,
        ):
        pass

    ## --------------------------------------
    def toSingleDataFrame(
            self,
            files: list,
            do_blockMix: bool = False,
            do_savedf: bool = False,
        ):
        evt = -1
        previous_bcid = -1
        df_count = 0
        d = []

        files = natsorted(files)

        if do_blockMix:
            files = files[1:]

        for ifile in files:
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
                        d.append(
                            {
                            'evt': evt,
                            'board': id,
                            'col': col,
                            'row': row,
                            'toa': toa,
                            'tot': tot,
                            'cal': cal,
                            }
                        )
                    elif line.split(' ')[2] == 'TRAILER':
                        pass

        df = pd.DataFrame(d)

        ## Under develop
        if do_savedf:
            pass

        return df

    ## --------------------------------------
    def toSingleDataFramePerDirectory(
            self,
            root: str,
            path_pattern: str,
        ):

        evt = -1
        previous_bcid = -1
        df_count = 0
        name_pattern = "*translated*.dat"

        dirs = glob(f"{root}/{path_pattern}")
        dirs = natsorted(dirs)
        print(dirs)

        for dir in dirs:
            d = []
            name = dir.split('/')[-1]
            files = glob(f"{dir}/{name_pattern}")

            for ifile in files:
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
                            d.append(
                                {
                                'evt': evt,
                                'board': id,
                                'col': col,
                                'row': row,
                                'toa': toa,
                                'tot': tot,
                                'cal': cal,
                                }
                            )
                        elif line.split(' ')[2] == 'TRAILER':
                            pass

            df = pd.DataFrame(d)
            df.to_parquet(name+'.pqt', index=False)
            del df

    ## --------------------------------------
    def making_heatmap_byPandas(
            self,
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
            self,
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
            self,
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
            self,
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
            self,
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
        self,
        input_df: pd.DataFrame,
        iterative_cnt: int,
        poly_order: int,
    ):

        corr_toas = []
        coeffs = []

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

            corr_b0 = input_df['toa_b0'].values + poly_func_b0(input_df['tot_b0'].values)
            corr_b1 = input_df['toa_b1'].values + poly_func_b1(input_df['tot_b1'].values)
            corr_b3 = input_df['toa_b3'].values + poly_func_b3(input_df['tot_b3'].values)

            del_toa_b0 = (0.5*(corr_b1 + corr_b3) - corr_b0)
            del_toa_b1 = (0.5*(corr_b0 + corr_b3) - corr_b1)
            del_toa_b3 = (0.5*(corr_b0 + corr_b1) - corr_b3)

            if i == iterative_cnt-1:

                coeff_b0 = np.polyfit(input_df['tot_b0'].values, del_toa_b0, poly_order)
                coeff_b1 = np.polyfit(input_df['tot_b1'].values, del_toa_b1, poly_order)
                coeff_b3 = np.polyfit(input_df['tot_b3'].values, del_toa_b3, poly_order)

                coeffs.append(coeff_b0)
                coeffs.append(coeff_b1)
                coeffs.append(coeff_b3)

                corr_toas.append(corr_b0)
                corr_toas.append(corr_b1)
                corr_toas.append(corr_b3)

        return coeffs, corr_toas

    ## --------------------------------------
    def making_3d_heatmap_byPandas(
            self,
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
            self,
            input_df: pd.DataFrame,
            chip_names: list,
            chip_labels: list,
    ):
        h = {chip_names[idx]: hist.Hist(hist.axis.Regular(50, 140, 240, name="CAL", label="CAL [LSB]"),
                    hist.axis.Regular(64, 0, 512,  name="TOT", label="TOT [LSB]"),
                    hist.axis.Regular(64, 0, 1024, name="TOA", label="TOA [LSB]"),
            )
        for idx, boardID in enumerate(chip_labels)}
        for idx, boardID in enumerate(chip_labels):
            tmp_df = input_df[input_df['board'] == int(boardID)]
            h[chip_names[idx]].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values)

        return h
    ## --------------------------------------
    def return_resolution_ps(self, sig_a, err_a, sig_b, err_b, sig_c, err_c):
        res = np.sqrt(0.5)*(np.sqrt(sig_a**2 + sig_b**2 - sig_c**2))
        var_res = (1/4)*(1/res**2)*(((sig_a**2)*(err_a**2))+((sig_b**2)*(err_b**2))+((sig_c**2)*(err_c**2)))
        return res*1e3, np.sqrt(var_res)*1e3

    ## --------------------------------------
    def draw_hist_plot_pull(
        self,
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
    def make_pix_inclusive_plots(
            self,
            input_hist: hist.Hist,
            chip_name,
            chip_figname,
            chip_figtitle,
            fig_path,
            save: bool = False,
            show: bool = False,
            tag: str = '',
            title_tag: str = '',
        ):
        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, CAL{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("CAL")[:].plot1d(ax=ax, lw=2)
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_CAL_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, TOT{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOT")[:].plot1d(ax=ax, lw=2)
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOT_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, TOA{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOA")[:].plot1d(ax=ax, lw=2)
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOA_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,20))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{chip_figtitle}, TOA v TOT{title_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOA","TOT")[::2j,::2j].plot2d(ax=ax)
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOA_TOT_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        if(show): plt.show()
        plt.close()

    ## --------------------------------------
    def event_display_withPandas(
            self,
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
