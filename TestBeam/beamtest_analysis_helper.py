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
from pathlib import Path

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
        'row': [],
        'col': [],
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
                    col = int(line.split(' ')[8])
                    row = int(line.split(' ')[6])
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
def toSingleDataFrame_newEventModel(
        files: list,
        do_blockMix: bool = False,
        do_savedf: bool = False,
    ):
    evt = -1
    previous_evt = -1
    d = {
        'evt': [],
        'board': [],
        'row': [],
        'col': [],
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
                if line.split(' ')[0] == 'EH':
                    tmp_evt = int(line.split(' ')[2])
                    if previous_evt != tmp_evt:
                        evt += 1
                        previous_evt = tmp_evt
                        pass
                elif line.split(' ')[0] == 'H':
                    pass
                    # bcid = int(line.split(' ')[-1])
                elif line.split(' ')[0] == 'D':
                    id  = int(line.split(' ')[1])
                    col = int(line.split(' ')[-4])
                    row = int(line.split(' ')[-5])
                    toa = int(line.split(' ')[-3])
                    tot = int(line.split(' ')[-2])
                    cal = int(line.split(' ')[-1])
                    file_d['evt'].append(evt)
                    file_d['board'].append(id)
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
    ## Under develop
    if do_savedf:
        pass

    return df


## --------------------------------------
def toSingleDataFramePerDirectory(
        root: str,
        path_pattern: str,
        data_qinj: bool = False,
        save_to_csv: bool = False,
        debugging: bool = False,
    ):

    evt = -1
    previous_bcid = -1
    df_count = 0
    name_pattern = "*translated*.dat"
    if data_qinj:
        name_pattern = "*translated_[1-9]*.dat"

    dirs = glob(f"{root}/{path_pattern}")
    dirs = natsorted(dirs)
    print(dirs[:3])

    if debugging:
        dirs = dirs[:2]

    d = {
        'evt': [],
        'board': [],
        'row': [],
        'col': [],
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

        df = df.astype('int')
        if data_qinj:
            df.drop(columns=['evt', 'board'], inplace=True)
        if save_to_csv:
            df.to_csv(name+'.csv', index=False)
        else:
            df.to_parquet(name+'.pqt', index=False)
        del df

## --------------------------------------
def toSingleDataFramePerDirectory_newEventModel(
        root: str,
        path_pattern: str,
        data_qinj: bool = False,
        save_to_csv: bool = False,
        debugging: bool = False,
    ):

    evt = -1
    previous_evt = -1
    name_pattern = "*translated*.nem"
    if data_qinj:
        name_pattern = "*translated_[1-9]*.nem"

    dirs = glob(f"{root}/{path_pattern}")
    dirs = natsorted(dirs)
    print(dirs[:3])

    if debugging:
        dirs = dirs[:1]

    d = {
        'evt': [],
        'board': [],
        'row': [],
        'col': [],
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
                for line in infile:
                    if line.split(' ')[0] == 'EH':
                        tmp_evt = int(line.split(' ')[2])
                        if previous_evt != tmp_evt:
                            evt += 1
                            previous_evt = tmp_evt
                    elif line.split(' ')[0] == 'H':
                        pass
                        # bcid = int(line.split(' ')[-1])
                    elif line.split(' ')[0] == 'D':
                        id  = int(line.split(' ')[1])
                        col = int(line.split(' ')[-4])
                        row = int(line.split(' ')[-5])
                        toa = int(line.split(' ')[-3])
                        tot = int(line.split(' ')[-2])
                        cal = int(line.split(' ')[-1])
                        file_d['evt'].append(evt)
                        file_d['board'].append(id)
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
        if data_qinj:
            df.drop(columns=['evt', 'board'], inplace=True)
        if save_to_csv:
            df.to_csv(name+'.csv', index=False)
        else:
            df.to_feather(name+'.feather')
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
def make_number_of_fired_board(
        input_df: pd.DataFrame,
    ):

    h = hist.Hist(hist.axis.Regular(5, 0, 5, name="num_board", label="Number of fired board per event"))
    h.fill(input_df.groupby('evt')['board'].nunique())

    fig = plt.figure(dpi=50, figsize=(14,12))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
    h.plot1d(ax=ax, lw=2)
    plt.tight_layout()
    del h

## --------------------------------------
def make_TDC_summary_table(
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
def df_for_efficiency_with_single_board(
        input_df: pd.DataFrame,
        pixel: set = (8, 8), # (row, col)
        board_id: int = 0,
    ):

    df_tmp = input_df.set_index('evt')
    selection = (df_tmp['board'] == board_id) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])
    new_df = input_df.loc[input_df['evt'].isin(df_tmp.loc[selection].index)]

    del df_tmp, selection
    return new_df


## --------------------------------------
def df_for_efficiency_with_two_boards(
        input_df: pd.DataFrame,
        pixel: set = (8, 8), # (row, col)
        board_ids: set = (0, 3), #(board 1, board 2)
    ):

    df_tmp = input_df.set_index('evt')
    selection1 = (df_tmp['board'] == board_ids[0]) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])
    selection2 = (df_tmp['board'] == board_ids[1]) & (df_tmp['row'] == pixel[0]) & (df_tmp['col'] == pixel[1])

    filtered_index = list(set(df_tmp.loc[selection1].index).intersection(df_tmp.loc[selection2].index))
    new_df = input_df.loc[input_df['evt'].isin(filtered_index)]

    del df_tmp, filtered_index, selection1, selection2
    return new_df


## --------------------------------------
def singlehit_event_clear_func(
        input_df: pd.DataFrame
    ):

    ## event has one hit from each board
    event_board_counts = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None
    for board in event_board_counts:
        if event_selection_col is None:
            event_selection_col = (event_board_counts[board] == 1)
        else:
            event_selection_col = event_selection_col & (event_board_counts[board] == 1)
    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = input_df[input_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

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
def tdc_event_selection_pivot(
        input_df: pd.DataFrame,
        tdc_cuts_dict: dict
    ):
    # Create boolean masks for each board's filtering criteria
    masks = {}
    for board, cuts in tdc_cuts_dict.items():
        mask = (
            input_df['cal'][board].between(cuts[0], cuts[1]) &
            input_df['toa'][board].between(cuts[2], cuts[3]) &
            input_df['tot'][board].between(cuts[4], cuts[5])
        )
        masks[board] = mask

    # Combine the masks using logical AND
    combined_mask = pd.concat(masks, axis=1).all(axis=1)
    del masks

    # Apply the combined mask to the DataFrame
    tdc_filtered_df = input_df[combined_mask].reset_index(drop=True)
    del combined_mask
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

    corr_toas = []
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
            corr_toas.append(corr_b0)
            corr_toas.append(corr_b1)
            corr_toas.append(corr_b2)
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
def return_resolution_three_board(
        fit_params: dict,
        var: list,
        board_list:list,
    ):

    results = {
        board_list[0]: np.sqrt((1/2)*(fit_params[var[0]][0]**2 + fit_params[var[1]][0]**2 - fit_params[var[2]][0]**2))*1e3,
        board_list[1]: np.sqrt((1/2)*(fit_params[var[0]][0]**2 + fit_params[var[2]][0]**2 - fit_params[var[1]][0]**2))*1e3,
        board_list[2]: np.sqrt((1/2)*(fit_params[var[1]][0]**2 + fit_params[var[2]][0]**2 - fit_params[var[0]][0]**2))*1e3,
    }

    return results

## --------------------------------------
def return_resolution_four_board(
        fit_params: dict,
        var: list,
    ):
    if len(fit_params) != 6:
        raise ValueError('Need 6 parameters for calculating time resolution')

    resolution = np.sqrt(
        (1/6)*(
            2*fit_params[var[0]][0]**2+
            2*fit_params[var[1]][0]**2+
            2*fit_params[var[2]][0]**2
            -fit_params[var[3]][0]**2
            -fit_params[var[4]][0]**2
            -fit_params[var[5]][0]**2
        )
    )*1e3

    return resolution

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
        no_show_fit: bool,
        no_draw: bool,
        get_chisqure: bool = False,
        skip_reduce: bool = False,
    ):

    from lmfit.models import GaussianModel
    from lmfit.lineshapes import gaussian

    mod = GaussianModel(nan_policy='omit')

    if skip_reduce:
        reduced_data = input_data
    else:
        reduced_data = input_data[(input_data > input_data.mean()-std_range_cut) & (input_data < input_data.mean()+std_range_cut)]


    fit_min = reduced_data.mean()-width_factor*reduced_data.std()
    fit_max = reduced_data.mean()+width_factor*reduced_data.std()
    del reduced_data

    input_hist_peak = input_hist[(fit_min)*1j:(fit_max)*1j]
    centers = input_hist.axes[0].centers
    fit_centers = input_hist_peak.axes[0].centers
    pars = mod.guess(input_hist.values(), x=centers)
    out = mod.fit(input_hist_peak.values(), pars, x=fit_centers, weights=1/np.sqrt(input_hist_peak.values()))

    if not no_draw:

        popt = [par for name, par in out.best_values.items()]
        pcov = out.covar

        if np.isfinite(pcov).all():
            n_samples = 100
            vopts = np.random.multivariate_normal(popt, pcov, n_samples)
            sampled_ydata = np.vstack([gaussian(fit_centers, *vopt).T for vopt in vopts])
            model_uncert = np.nanstd(sampled_ydata, axis=0)
        else:
            model_uncert = np.zeros_like(np.sqrt(input_hist.variances()))

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

        fig = plt.figure()

        if no_show_fit:
            grid = fig.add_gridspec(1, 1, hspace=0)
            main_ax = fig.add_subplot(grid[0])
            hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)
            main_ax.set_title(f'{fig_title}', loc="right", size=25)
            main_ax.errorbar(centers, input_hist.values(), np.sqrt(input_hist.variances()),
                            ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                            ms=6, capsize=1, capthick=2, alpha=0.8)
            main_ax.plot(fit_centers, out.best_fit, color="hotpink", ls="-", lw=2, alpha=0.8,
                        label=fr"$\mu:{out.params['center'].value:.3f}, \sigma: {abs(out.params['sigma'].value):.3f}$")
            main_ax.set_ylabel('Counts')
            main_ax.set_ylim(-20, None)
            main_ax.set_xlabel(r'Time Walk Corrected $\Delta$TOA [ns]')
            plt.tight_layout()

        else:
            grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
            main_ax = fig.add_subplot(grid[0])
            subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
            plt.setp(main_ax.get_xticklabels(), visible=False)
            hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=25)

            main_ax.set_title(f'{fig_title}', loc="right", size=25)
            main_ax.errorbar(centers, input_hist.values(), np.sqrt(input_hist.variances()),
                            ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                            ms=6, capsize=1, capthick=2, alpha=0.8)
            main_ax.plot(fit_centers, out.best_fit, color="hotpink", ls="-", lw=2, alpha=0.8,
                        label=fr"$\mu:{out.params['center'].value:.3f}, \sigma: {abs(out.params['sigma'].value):.3f}$")
            main_ax.set_ylabel('Counts')
            main_ax.set_ylim(-20, None)

            main_ax.fill_between(
                fit_centers,
                out.eval(x=fit_centers) - model_uncert,
                out.eval(x=fit_centers) + model_uncert,
                color="hotpink",
                alpha=0.2,
                label='Uncertainty'
            )
            main_ax.legend(fontsize=20, loc='upper right')

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

    if get_chisqure:
        return [out.params['sigma'].value, out.params['sigma'].stderr, out.chisqr/(out.ndata-1)]
    else:
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
def four_board_single_hit_single_track_time_resolution_by_looping(
        input_df: pd.DataFrame,
        track_df: pd.DataFrame,
        chip_labels: list,
    ):
    from tqdm import tqdm

    output_dict = {
        'row0': [],
        'col0': [],
        'row1': [],
        'col1': [],
        'row2': [],
        'col2': [],
        'row3': [],
        'col3': [],
        'res0': [],
        'res1': [],
        'res2': [],
        'res3': [],
        'chi01': [],
        'chi02': [],
        'chi03': [],
        'chi12': [],
        'chi13': [],
        'chi23': [],
    }

    for i in tqdm(range(len(track_df))):

        pix_dict = {
            # board ID: [row, col]
            0: [track_df.iloc[i]['row_0'], track_df.iloc[i]['col_0']],
            1: [track_df.iloc[i]['row_1'], track_df.iloc[i]['col_1']],
            2: [track_df.iloc[i]['row_2'], track_df.iloc[i]['col_2']],
            3: [track_df.iloc[i]['row_3'], track_df.iloc[i]['col_3']],
        }

        pix_filtered_df = pixel_filter(input_df, pix_dict)

        tdc_cuts = {
            # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
            0: [pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 0]['cal'].std(), 100, 450,    0, 600],
            1: [pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 1]['cal'].std(),   0, 1100,   0, 600],
            2: [pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 2]['cal'].std(),   0, 1100,   0, 600],
            3: [pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].mean()-2*pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].std(), pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].mean()+2*pix_filtered_df.loc[pix_filtered_df['board'] == 3]['cal'].std(),   0, 1100,   0, 600], # pixel ()
        }

        tdc_filtered_df = tdc_event_selection(pix_filtered_df, tdc_cuts)
        tdc_filtered_df = singlehit_event_clear_func(tdc_filtered_df)
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

        corr_toas = four_board_iterative_timewalk_correction(df_in_time, 5, 3)

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
        params = lmfit_gaussfit_with_pulls(diff_b01, dTOA_b01, std_range_cut=0.4, width_factor=1.25, fig_title='Board 0 - Board 1',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=True)
        fit_params_lmfit['01'] = params
        params = lmfit_gaussfit_with_pulls(diff_b02, dTOA_b02, std_range_cut=0.4, width_factor=1.25, fig_title='Board 0 - Board 2',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=True)
        fit_params_lmfit['02'] = params
        params = lmfit_gaussfit_with_pulls(diff_b03, dTOA_b03, std_range_cut=0.4, width_factor=1.25, fig_title='Board 0 - Board 3',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=True)
        fit_params_lmfit['03'] = params
        params = lmfit_gaussfit_with_pulls(diff_b12, dTOA_b12, std_range_cut=0.4, width_factor=1.25, fig_title='Board 1 - Board 2',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=True)
        fit_params_lmfit['12'] = params
        params = lmfit_gaussfit_with_pulls(diff_b13, dTOA_b13, std_range_cut=0.4, width_factor=1.25, fig_title='Board 1 - Board 3',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=True)
        fit_params_lmfit['13'] = params
        params = lmfit_gaussfit_with_pulls(diff_b23, dTOA_b23, std_range_cut=0.4, width_factor=1.25, fig_title='Board 2 - Board 3',
                                                use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=True)
        fit_params_lmfit['23'] = params

        del params
        del dTOA_b01, dTOA_b02, dTOA_b03, dTOA_b12, dTOA_b13, dTOA_b23
        del diff_b01, diff_b02, diff_b03, diff_b12, diff_b13, diff_b23

        res_b0 = return_resolution_four_board(fit_params_lmfit, ['01', '02', '03', '12', '13', '23'])
        res_b1 = return_resolution_four_board(fit_params_lmfit, ['01', '12', '13', '02', '03', '23'])
        res_b2 = return_resolution_four_board(fit_params_lmfit, ['02', '12', '23', '01', '03', '13'])
        res_b3 = return_resolution_four_board(fit_params_lmfit, ['03', '13', '23', '01', '02', '12'])

        output_dict['row0'].append(track_df.iloc[i]['row_0'])
        output_dict['col0'].append(track_df.iloc[i]['col_0'])
        output_dict['row1'].append(track_df.iloc[i]['row_1'])
        output_dict['col1'].append(track_df.iloc[i]['col_1'])
        output_dict['row2'].append(track_df.iloc[i]['row_2'])
        output_dict['col2'].append(track_df.iloc[i]['col_2'])
        output_dict['row3'].append(track_df.iloc[i]['row_3'])
        output_dict['col3'].append(track_df.iloc[i]['col_3'])
        output_dict['res0'].append(res_b0)
        output_dict['res1'].append(res_b1)
        output_dict['res2'].append(res_b2)
        output_dict['res3'].append(res_b3)
        output_dict['chi01'].append(fit_params_lmfit['01'][2])
        output_dict['chi02'].append(fit_params_lmfit['02'][2])
        output_dict['chi03'].append(fit_params_lmfit['03'][2])
        output_dict['chi12'].append(fit_params_lmfit['12'][2])
        output_dict['chi13'].append(fit_params_lmfit['13'][2])
        output_dict['chi23'].append(fit_params_lmfit['23'][2])

        del res_b0, res_b1, res_b2, res_b3, fit_params_lmfit

    summary_df = pd.DataFrame(data=output_dict)
    del output_dict
    return summary_df

## --------------------------------------
def bootstrap_single_track_time_resolution(
        list_of_pivots: list,
        board_to_analyze: list[int],
        iteration: int = 10,
        sampling_fraction: float = 0.75,
    ):
    from tqdm import tqdm

    final_dict = {}

    for idx in board_to_analyze:
        final_dict[f'row{idx}'] = []
        final_dict[f'col{idx}'] = []
        final_dict[f'res{idx}'] = []
        final_dict[f'err{idx}'] = []

    for itable in tqdm(list_of_pivots):

        sum_arr = {}
        sum_square_arr = {}
        counter = 0

        for idx in board_to_analyze:
            sum_arr[idx] = 0
            sum_square_arr[idx] = 0

        for iloop in range(iteration):

            try_df = itable.reset_index()
            tdc_cuts = {}
            for idx in board_to_analyze:
                # board ID: [CAL LB, CAL UB, TOA LB, TOA UB, TOT LB, TOT UB]
                if idx == 0:
                    tdc_cuts[idx] = [try_df['cal'][idx].mean()-5, try_df['cal'][idx].mean()+5,  350, 500, 0, 600]
                else:
                    tdc_cuts[idx] = [try_df['cal'][idx].mean()-5, try_df['cal'][idx].mean()+5,  0, 1100, 0, 600]

            tdc_filtered_df = tdc_event_selection_pivot(try_df, tdc_cuts)
            del try_df, tdc_cuts

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
                d[f'toa_b{str(idx)}'] = 12.5 - tdc_filtered_df['toa'][idx] * bins
                d[f'tot_b{str(idx)}'] = (2*tdc_filtered_df['tot'][idx] - np.floor(tdc_filtered_df['tot'][idx]/32)) * bins

            df_in_time = pd.DataFrame(data=d)
            del d, tdc_filtered_df

            if len(board_to_analyze) == 3:
                corr_toas = three_board_iterative_timewalk_correction(df_in_time, 5, 3, board_list=board_to_analyze)
            else:
                corr_toas = four_board_iterative_timewalk_correction(df_in_time, 5, 3)

            diffs = {}
            for board_a in board_to_analyze:
                for board_b in board_to_analyze:
                    if board_b <= board_a:
                        continue
                    name = f"{board_a}{board_b}"
                    diffs[name] = np.asarray(corr_toas[f'toa_b{board_a}'] - corr_toas[f'toa_b{board_b}'])

            hists = {}
            for key in diffs.keys():
                hists[key] = hist.Hist(hist.axis.Regular(80, -1.2, 1.2, name="TWC_delta_TOA", label=r'Time Walk Corrected $\Delta$TOA [ns]'))
                hists[key].fill(diffs[key])

            try:
                fit_params_lmfit = {}
                for key in hists.keys():
                    params = lmfit_gaussfit_with_pulls(diffs[key], hists[key], std_range_cut=0.4, width_factor=1.25, fig_title='',
                                                        use_pred_uncert=True, no_show_fit=False, no_draw=True, get_chisqure=False)
                    fit_params_lmfit[key] = params
                del params, hists, diffs, corr_toas

                if len(board_to_analyze) == 3:
                    resolutions = return_resolution_three_board(fit_params_lmfit, var=list(fit_params_lmfit.keys()), board_list=board_to_analyze)
                else:
                    print('not support yet')

                if any(np.isnan(val) for key, val in resolutions.items()):
                    print('fit results is not good, skipping this iteration')
                    continue

                for key in resolutions.keys():
                    sum_arr[key] += resolutions[key]
                    sum_square_arr[key] += resolutions[key]**2

                counter += 1

            except:
                print('Failed, skipping')
                del hists, diffs, corr_toas

        if counter != 0:
            for idx in board_to_analyze:
                final_dict[f'row{idx}'].append(itable['row'][idx].unique()[0])
                final_dict[f'col{idx}'].append(itable['col'][idx].unique()[0])

            for key in sum_arr.keys():
                mean = sum_arr[key]/counter
                std = np.sqrt((1/(counter-1))*(sum_square_arr[key]-counter*(mean**2)))
                final_dict[f'res{key}'].append(mean)
                final_dict[f'err{key}'].append(std)
        else:
            print('Track is not validate for bootstrapping')

    return final_dict

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