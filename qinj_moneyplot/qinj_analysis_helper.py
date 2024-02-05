import numpy as np
import datetime
import pandas as pd
from natsort import natsorted
from glob import glob
import hist
import copy
from pathlib import Path
import os
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import mplhep as hep
plt.style.use(hep.style.CMS)


## --------------- Decoding Class -----------------------
## --------------------------------------
class DecodeBinary:
    def __init__(self, firmware_key, board_id: list[int], file_list: list[Path]):
        self.firmware_key            = firmware_key
        self.in_event                = False
        self.eth_words_in_event      = -1
        self.words_in_event          = -1
        self.current_word            = -1
        self.event_number            = -1
        self.enabled_channels        = -1
        self.header_pattern          = 0xc3a3c3a
        self.trailer_pattern         = 0b001011
        self.channel_header_pattern  = 0x3c5c0 >> 2
        self.previous_event          = -1
        self.running_word            = None
        self.position_40bit          = 0
        self.current_channel         = -1
        self.event_counter           = 0
        self.board_ids               = board_id
        self.data                    = {}
        self.files_to_process        = file_list

        self.data_template = {
            'evt_number': [],
            'bcid': [],
            'l1a_counter': [],
            'evt': [],
            'ea': [],
            'board': [],
            'row': [],
            'col': [],
            'toa': [],
            'tot': [],
            'cal': [],
        }

        self.data_to_load = copy.deepcopy(self.data_template)

    def reset_params(self):
        self.in_event                = False
        self.eth_words_in_event      = -1
        self.words_in_event          = -1
        self.current_word            = -1
        self.event_number            = -1
        self.enabled_channels        = -1
        self.running_word            = None
        self.position_40bit          = 0
        self.current_channel         = -1
        self.data = {}

    def div_ceil(self, x,y):
        return -(x//(-y))

    def decode_40bit(self, word):
        # Header
        if word >> 22 == self.channel_header_pattern:
            self.current_channel += 1
            while not ((self.enabled_channels >> self.current_channel) & 0b1):
                self.current_channel += 1
                if self.current_channel > 3:
                    print('Found more headers than number of channels')
                    self.reset_params()
                    return
            self.bcid = (word & 0xfff)
            self.l1acounter = ((word >> 14) & 0xff)
            self.data[self.current_channel] = copy.deepcopy(self.data_template)
        # Data
        elif (word >> 39) == 1:
            self.data[self.current_channel]['evt_number'].append(self.event_number)
            self.data[self.current_channel]['bcid'].append(self.bcid)
            self.data[self.current_channel]['l1a_counter'].append(self.l1acounter)
            self.data[self.current_channel]['evt'].append(self.event_counter)
            self.data[self.current_channel]['ea'].append((word >> 37) & 0b11)
            self.data[self.current_channel]['board'].append(self.current_channel)
            self.data[self.current_channel]['row'].append((word >> 29) & 0b1111)
            self.data[self.current_channel]['col'].append((word >> 33) & 0b1111)
            self.data[self.current_channel]['toa'].append((word >> 19) & 0x3ff)
            self.data[self.current_channel]['tot'].append((word >> 10) & 0x1ff)
            self.data[self.current_channel]['cal'].append((word) & 0x3ff)

        # Trailer
        elif (word >> 22) & 0x3ffff == self.board_ids[self.current_channel]:
            hits = (word >> 8) & 0xff
            if len(self.data[self.current_channel]['evt']) != hits:
                print('Number of hits does not match!')
                self.reset_params()
                return

        # Something else
        else:
            binary = format(word, '040b')
            print(f'Warning! Found 40 bits word which is not matched with the pattern {binary}')
            self.reset_params()
            return

    def decode_files(self):
        df = pd.DataFrame(self.data_template)
        df = df.astype('int')
        decoding = False
        for ifile in self.files_to_process:
            with open(file=ifile, mode='rb') as infile:
                while True:
                    in_data = infile.read(4)
                    # print(in_data)
                    if in_data == b'':
                        break
                    word = int.from_bytes(in_data, byteorder='little')
                    if not decoding and word == 0:
                        continue
                    if not decoding:
                        decoding = True

                    ## Event header
                    if (word >> 4) == 0xc3a3c3a:
                        self.enabled_channels = word & 0b1111
                        self.reset_params()
                        self.in_event = True
                        # print('Event header')
                        continue

                    # Event Header Line Two Found
                    elif(self.in_event and (self.words_in_event == -1) and (word >> 28 == self.firmware_key)):
                        self.current_word       = 0
                        self.event_number       = word >> 12 & 0xffff
                        self.words_in_event     = word >> 2 & 0x3ff
                        self.eth_words_in_event = self.div_ceil(40*self.words_in_event, 32)
                        # print(f"Num Words {self.words_in_event} & Eth Words {self.eth_words_in_event}")
                        # Set valid_data to true once we see fresh data
                        if(self.event_number==1 or self.event_number==0): self.valid_data = True
                        # print('Event Header Line Two Found')
                        # print(self.event_number)
                        continue

                    # Event Header Line Two NOT Found after the Header
                    elif(self.in_event and (self.words_in_event == -1) and (word >> 28 != self.firmware_key)):
                        # print('Event Header Line Two NOT Found after the Header')
                        self.reset_params()
                        continue

                    # Event Trailer NOT Found after the required number of ethernet words was read
                    elif(self.in_event and (self.eth_words_in_event==self.current_word) and (word >> 26 != self.trailer_pattern)):
                        # print('Event Trailer NOT Found after the required number of ethernet words was read')
                        self.reset_params()
                        continue

                    # Event Trailer Found - DO NOT CONTINUE
                    elif(self.in_event and (self.eth_words_in_event==self.current_word) and (word >> 26 == self.trailer_pattern)):
                        for key in self.data_to_load:
                            for board in self.data:
                                self.data_to_load[key] += self.data[board][key]
                        # print(self.event_number)
                        # print(self.data)
                        self.event_counter += 1

                        if len(self.data_to_load['evt']) >= 10000:
                            df = pd.concat([df, pd.DataFrame(self.data_to_load)], ignore_index=True)
                            self.data_to_load = copy.deepcopy(self.data_template)

                    # Event Data Word
                    elif(self.in_event):
                        # print(self.current_word)
                        # print(format(word, '032b'))
                        if self.position_40bit == 4:
                            self.word_40 = self.word_40 | word
                            self.decode_40bit(self.word_40)
                            self.position_40bit = 0
                            self.current_word += 1
                            continue
                        if self.position_40bit >= 1:
                            self.word_40 = self.word_40 | (word >> (8*(4-self.position_40bit)))
                            self.decode_40bit(self.word_40)
                        self.word_40 = (word << ((self.position_40bit + 1)*8)) & 0xffffffffff
                        self.position_40bit += 1
                        self.current_word += 1
                        continue

                    # Reset anyway!
                    self.reset_params()

                if len(self.data_to_load['evt']) > 0:
                    df = pd.concat([df, pd.DataFrame(self.data_to_load)], ignore_index=True)
                    self.data_to_load = copy.deepcopy(self.data_template)
        return df

## --------------- Decoding Class -----------------------


## --------------- Text converting to DataFrame -----------------------
## --------------------------------------
def toSingleDataFrame_newEventModel(
        files: list,
        do_savedf: bool = False,
    ):
    d = {
        # 'board': [],
        'row': [],
        'col': [],
        'toa': [],
        'tot': [],
        'cal': [],
    }

    files = natsorted(files)
    df = pd.DataFrame(d)

    for ifile in files:
        file_d = copy.deepcopy(d)
        with open(ifile, 'r') as infile:
            for line in infile:
                if line.split(' ')[0] == 'EH':
                    pass
                elif line.split(' ')[0] == 'H':
                    pass
                    # bcid = int(line.split(' ')[-1])
                elif line.split(' ')[0] == 'D':
                    # id  = int(line.split(' ')[1])
                    col = int(line.split(' ')[-4])
                    row = int(line.split(' ')[-5])
                    toa = int(line.split(' ')[-3])
                    tot = int(line.split(' ')[-2])
                    cal = int(line.split(' ')[-1])
                    # file_d['board'].append(id)
                    file_d['row'].append(row)
                    file_d['col'].append(col)
                    file_d['toa'].append(toa)
                    file_d['tot'].append(tot)
                    file_d['cal'].append(cal)
                elif line.split(' ')[0] == 'T':
                    pass
                elif line.split(' ')[0] == 'ET':
                    pass
        if len(file_d['evt']) > 0:
            file_df = pd.DataFrame(file_d)
            df = pd.concat((df, file_df), ignore_index=True)
            del file_df
        del file_d

    df = df.astype('int')
    return df

## --------------------------------------
def toSingleDataFramePerDirectory_newEventModel(
        path_to_dir: str,
        dir_name_pattern: str,
        save_to_csv: bool = False,
        debugging: bool = False,
        output_dir: str = "",
        extra_str: str = "",
    ):

    if output_dir != "":
        os.system(f"mkdir -p {output_dir}")
    name_pattern = "*translated*.nem"

    dirs = glob(f"{path_to_dir}/{dir_name_pattern}")
    dirs = natsorted(dirs)
    print(dirs[:3])

    if debugging:
        dirs = dirs[:1]

    d = {
        # 'board': [],
        'row': [],
        'col': [],
        'toa': [],
        'tot': [],
        'cal': [],
    }

    for dir in tqdm(dirs):
        df = pd.DataFrame(d)
        name = dir.split('/')[-1]
        files = glob(f"{dir}/{name_pattern}")

        for ifile in files:
            file_d = copy.deepcopy(d)

            if os.stat(ifile).st_size == 0:
                continue

            with open(ifile, 'r') as infile:
                for line in infile:
                    if line.split(' ')[0] == 'EH':
                        pass
                    elif line.split(' ')[0] == 'H':
                        pass
                        # bcid = int(line.split(' ')[-1])
                    elif line.split(' ')[0] == 'D':
                        # id  = int(line.split(' ')[1])
                        col = int(line.split(' ')[-4])
                        row = int(line.split(' ')[-5])
                        toa = int(line.split(' ')[-3])
                        tot = int(line.split(' ')[-2])
                        cal = int(line.split(' ')[-1])
                        # file_d['evt'].append(evt)
                        # file_d['board'].append(id)
                        file_d['row'].append(row)
                        file_d['col'].append(col)
                        file_d['toa'].append(toa)
                        file_d['tot'].append(tot)
                        file_d['cal'].append(cal)
                    elif line.split(' ')[0] == 'T':
                        pass
                    elif line.split(' ')[0] == 'ET':
                        pass
            # if len(file_d['evt']) > 0:
            file_df = pd.DataFrame(file_d)
            df = pd.concat((df, file_df), ignore_index=True)
            del file_df
            del file_d

        if not df.empty:
            df = df.astype('int')
            if save_to_csv:
                df.to_csv(name+'.csv', index=False)
            else:
                df.to_feather(f"{output_dir}/{name}{extra_str}.feather")
            del df

## --------------- Text converting to DataFrame -----------------------


# ## --------------- Plotting -----------------------
# ## --------------------------------------
# def return_hist(
#         input_df: pd.DataFrame,
#         chipNames: list[str],
#         chipLabels: list[int],
#         hist_bins: list = [50, 64, 64]
# ):
#     h = {chipNames[boardID]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
#                 hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
#                 hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
#         )
#     for boardID in chipLabels}

#     for boardID in chipLabels:
#         tmp_df = input_df.loc[input_df['board'] == boardID]
#         h[chipNames[boardID]].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values)

#     return h

## --------------------------------------
def plot_TDC_summary_table(
        input_df: pd.DataFrame,
        chipLabels: list,
        var: str
    ):

    for idx, id in enumerate(chipLabels):

        if input_df[input_df['board'] == int(id)].empty:
            continue

        sum_group = input_df[input_df['board'] == int(id)].groupby(["col", "row"]).agg({var:['mean','std']})
        sum_group.columns = sum_group.columns.droplevel()
        sum_group.reset_index(inplace=True)

        table_mean = sum_group.pivot_table(index='row', columns='col', values='mean')
        table_mean = table_mean.round(1)

        table_mean = table_mean.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_mean = table_mean.reindex(columns=np.arange(0,16))

        table_std = sum_group.pivot_table(index='row', columns='col', values='std')
        table_std = table_std.round(2)

        table_std = table_std.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        table_std = table_std.reindex(columns=np.arange(0,16))

        plt.rcParams["xtick.major.size"] = 2.5
        plt.rcParams["ytick.major.size"] = 2.5
        plt.rcParams['xtick.minor.visible'] = False
        plt.rcParams['ytick.minor.visible'] = False

        fig, axes = plt.subplots(1, 2, figsize=(20, 20))

        im1 = axes[0].imshow(table_mean, vmin=1)
        im2 = axes[1].imshow(table_std, vmin=1)

        hep.cms.text(loc=0, ax=axes[0], text="Preliminary", fontsize=25)
        hep.cms.text(loc=0, ax=axes[1], text="Preliminary", fontsize=25)

        axes[0].set_title(f'{var.upper()} Mean', loc="right")
        axes[1].set_title(f'{var.upper()} Std', loc="right")

        axes[0].set_xticks(np.arange(0,16))
        axes[0].set_yticks(np.arange(0,16))
        axes[1].set_xticks(np.arange(0,16))
        axes[1].set_yticks(np.arange(0,16))

        axes[0].invert_xaxis()
        axes[0].invert_yaxis()
        axes[1].invert_xaxis()
        axes[1].invert_yaxis()

        # i for col, j for row
        for i in range(16):
            for j in range(16):
                if np.isnan(table_mean.iloc[i,j]):
                    continue
                text_color = 'black' if table_mean.iloc[i,j] > (table_mean.stack().max() + table_mean.stack().min()) / 2 else 'white'
                axes[0].text(j, i, table_mean.iloc[i,j], ha="center", va="center", rotation=45, fontweight="bold", fontsize=12, color=text_color)

        for i in range(16):
            for j in range(16):
                if np.isnan(table_std.iloc[i,j]):
                    continue
                text_color = 'black' if table_std.iloc[i,j] > (table_std.stack().max() + table_std.stack().min()) / 2 else 'white'
                axes[1].text(j, i, table_std.iloc[i,j], ha="center", va="center", rotation=45, color=text_color, fontweight="bold", fontsize=12)

        plt.minorticks_off()
        plt.tight_layout()

## --------------------------------------
def plot_1d_TDC_histograms(
        input_hist: hist.Hist,
        chip_name: str,
        chip_figname: str,
        fig_title: str,
        fig_path: str = './',
        save: bool = False,
        tag: str = '',
        fig_tag: str = '',
        slide_friendly: bool = False,
        do_logy: bool = False,
    ):

    if not slide_friendly:
        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{fig_title}, CAL{fig_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("CAL")[:].plot1d(ax=ax, lw=2)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_CAL_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{fig_title}, TOT{fig_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOT")[:].plot1d(ax=ax, lw=2)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOT_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,10))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{fig_title}, TOA{fig_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOA")[:].plot1d(ax=ax, lw=2)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOA_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        plt.close()

        fig = plt.figure(dpi=50, figsize=(20,20))
        gs = fig.add_gridspec(1,1)
        ax = fig.add_subplot(gs[0,0])
        ax.set_title(f"{fig_title}, TOA v TOT{fig_tag}", loc="right", size=25)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
        input_hist[chip_name].project("TOA","TOT")[::2j,::2j].plot2d(ax=ax)
        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/"+chip_figname+"_TOA_TOT_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        # plt.close()

    else:
        fig = plt.figure(dpi=100, figsize=(30,13))
        gs = fig.add_gridspec(2,2)

        for i, plot_info in enumerate(gs):
            ax = fig.add_subplot(plot_info)
            hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=20)
            if i == 0:
                ax.set_title(f"{fig_title}, CAL{fig_tag}", loc="right", size=15)
                input_hist[chip_name].project("CAL")[:].plot1d(ax=ax, lw=2)
                if do_logy:
                    ax.set_yscale('log')
            elif i == 1:
                ax.set_title(f"{fig_title}, TOA{fig_tag}", loc="right", size=15)
                input_hist[chip_name].project("TOA")[:].plot1d(ax=ax, lw=2)
                if do_logy:
                    ax.set_yscale('log')
            elif i == 2:
                ax.set_title(f"{fig_title}, TOT{fig_tag}", loc="right", size=15)
                input_hist[chip_name].project("TOT")[:].plot1d(ax=ax, lw=2)
                if do_logy:
                    ax.set_yscale('log')
            elif i == 3:
                ax.set_title(f"{fig_title}, TOA v TOT{fig_tag}", loc="right", size=14)
                input_hist[chip_name].project("TOA","TOT")[::2j,::2j].plot2d(ax=ax)

        plt.tight_layout()
        if(save): plt.savefig(fig_path+"/combined_TDC_"+tag+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+".png")
        # plt.close()
## --------------- Plotting -----------------------