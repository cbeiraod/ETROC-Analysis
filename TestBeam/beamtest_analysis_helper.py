import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import pandas as pd
from pathlib import Path
from natsort import natsorted
from glob import glob
import scipy.stats as stats
import hist
from hist import Hist
import mplhep as hep
plt.style.use(hep.style.CMS)
import boost_histogram as bh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from scipy.optimize import curve_fit

class etroc2_analysis_helper():

    def __init__(
            self,
            chip_names: list,
            chip_figtitles: list,
            chip_labels: list,
        ):
        self.chip_names = chip_names,
        self.chip_figtitles = chip_figtitles,
        self.chip_labels = chip_labels,

    ## --------------------------------------
    def toPandas(
            self,
            path: str,
            do_blockMix: bool = False,
            do_savedf: bool = False,
        ):
        evt = -1
        previous_bcid = -1
        d = []

        root = '../../ETROC-Data'
        name_pattern = "*translated*.dat"
        files = glob(f"{root}/{path}/{name_pattern}")
        files = natsorted(files)

        if do_blockMix:
            files = files[1:]

        for ifile in files:
            with open(ifile, 'r') as infile:
                for line in infile.readlines():
                    if line.split(' ')[2] == 'HEADER':
                        current_bcid = line.strip().split(' ')[-1]
                        if current_bcid != previous_bcid:
                            evt += 1
                        previous_bcid = current_bcid
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
                    text_color = 'black' if value > (pivot_table.values.max() + pivot_table.values.min()) / 2 else 'white'
                    text = str(value)
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