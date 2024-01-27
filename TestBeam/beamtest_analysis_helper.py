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

from pathlib import Path

import pickle

PeriCfg = {
    0: "PLL Config",
    1: "PLL Config",
    2: "PLL Config",
    3: "VRef & PLL Config",
    4: "TS & PS Config",
    5: "PS Config",
    6: "RefStrSel",
    7: "GRO & CLK40 Config",
    8: "GRO & CLK1280 Config",
    9: "GRO & FC Config",
    10: "BCIDOffset",
    11: "emptySlotBCID & BCIDOffset",
    12: "emptySlotBCID",
    13: "asy* & readoutClockDelayPixel",
    14: "asy* & readoutClockWidthPixel",
    15: "asy* & readoutClockDelayGlobal",
    16: "LTx_AmplSel & readoutClockWidthGlobal",
    17: "RTx_AmplSel & chargeInjectionDelay",
    18: "various",
    19: "various",
    20: "eFuse & trigger config",
    21: "eFuse Config",
    22: "eFuse Prog",
    23: "eFuse Prog",
    24: "eFuse Prog",
    25: "eFuse Prog",
    26: "linkResetFixedPattern",
    27: "linkResetFixedPattern",
    28: "linkResetFixedPattern",
    29: "linkResetFixedPattern",
    30: "ThrCounter Config",
    31: "TDCTest Config & ThrCounter Config",
}

PeriSta = {
    0: "PS_Late & AFCcalCap & AFCBusy",
    1: "fcAlignFinal and controller States",
    2: "FC Status",
    3: "invalidFCCount",
    4: "pllUnlockCount & invalidFCCount",
    5: "pllUnlockCount",
    6: "EFuseQ",
    7: "EFuseQ",
    8: "EFuseQ",
    9: "EFuseQ",
    10: "Unused",
    11: "Unused",
    12: "Unused",
    13: "Unused",
    14: "Unused",
    15: "Unused",
}

PixCfg = {
    0: "RF, IB & CL Selection",
    1: "QInj & QSel",
    2: "PD_DACDiscri & HysSel",
    3: "various",
    4: "DAC",
    5: "TH_Offset & DAC",
    6: "TDC Config",
    7: "various",
    8: "L1Adelay & selfTestOccupancy",
    9: "L1Adelay",
    10: "lowerCal",
    11: "upperCal & lowerCal",
    12: "lowerTOA & upperCal",
    13: "upperTOA & lowerTOA",
    14: "upperTOA",
    15: "lowerTOT",
    16: "upperTOT & lowerTOT",
    17: "lowerCalTrig & upperTOT",
    18: "upperCalTrig & lowerCalTrig",
    19: "lowerTOATrig & upperCalTrig",
    20: "lowerTOATrig",
    21: "upperTOATrig",
    22: "lowerTOTTrig & upperTOATrig",
    23: "upperTOTTrig & lowerTOTTrig",
    24: "upperTOTTrig",
    25: "unused",
    26: "unused",
    27: "unused",
    28: "unused",
    29: "unused",
    30: "unused",
    31: "unused",
}

PixSta = {
    0: "PixelID",
    1: "THState & NW & ScanDone",
    2: "BL",
    3: "TH & BL",
    4: "TH",
    5: "ACC",
    6: "ACC",
    7: "Copy of PixCfg31?",
}


## --------------- Compare Chip Configs -----------------------
## --------------------------------------
def compare_chip_configs(config_file1: Path, config_file2: Path):
    with open(config_file1, 'rb') as f:
        loaded_obj1 = pickle.load(f)
    with open(config_file2, 'rb') as f:
        loaded_obj2 = pickle.load(f)

    if loaded_obj1['chip'] != loaded_obj2['chip']:
        print("The config files are for different chips.")
        print(f"  - config 1: {loaded_obj1['chip']}")
        print(f"  - config 2: {loaded_obj2['chip']}")
        return

    chip1: dict = loaded_obj1['object']
    chip2: dict = loaded_obj2['object']

    common_keys = []
    for key in chip1.keys():
        if key not in chip2:
            print(f"Address Space \"{key}\" in config file 1 and not in config file 2")
        else:
            if key not in common_keys:
                common_keys += [key]
    for key in chip2.keys():
        if key not in chip1:
            print(f"Address Space \"{key}\" in config file 2 and not in config file 1")
        else:
            if key not in common_keys:
                common_keys += [key]

    for address_space_name in common_keys:
        if len(chip1[address_space_name]) != len(chip2[address_space_name]):
            print(f"The length of the \"{address_space_name}\" memory for config file 1 ({len(chip1[address_space_name])}) is not the same as for config file 2 ({len(chip2[address_space_name])})")

        length = min(len(chip1[address_space_name]), len(chip2[address_space_name]))

        for idx in length:
            if idx <= 0x001f:
                register = idx
                reg_info = f" (PeriCfg{register} contains {PeriCfg[register]})"
            elif idx == 0x0020:
                reg_info = " (Magic Number)"
            elif idx < 0x0100:
                reg_info = " (Unnamed Register Blk1)"
            elif idx <= 0x010f:
                register = idx - 0x0100
                reg_info = f" (PeriSta{register} contains {PeriSta[register]})"
            elif idx < 0x0120:
                reg_info = " (Unnamed Register BLk2)"
            elif idx <= 0x0123:
                reg_info = f" (SEUCounter{idx - 0x0120})"
            elif idx < 0x8000:
                reg_info = f" (Unnamed Register Blk3)"
            else:
                space = "Cfg"
                if (idx & 0x4000) != 0:
                    space = "Sta"
                broadcast = ""
                if (idx & 0x2000) != 0:
                    broadcast = " broadcast"
                register = idx & 0x1f
                row = (idx >> 5) & 0xf
                col = (idx >> 9) & 0xf
                contains = ""
                if space == "Cfg":
                    contains = PixCfg[register]
                else:
                    contains = PixSta[register]
                reg_info = f" (Pix{space}{register} - Pixel R{row} C{col}{broadcast} contains {contains})"
            if chip1[address_space_name][idx] != chip2[address_space_name][idx]:
                print(f"The register at address {idx} of {address_space_name}{reg_info} is different between config file 1 and config file 2: {chip1[address_space_name][idx]:#010b} vs {chip2[address_space_name][idx]:#010b}")

    print("Done comparing!")


## --------------- Decoding Class -----------------------
## --------------------------------------
class DecodeBinary:
    def __init__(self, firmware_key, board_id: list[int], file_list: list[Path], save_nem: Path | None = None):
        self.firmware_key            = firmware_key
        self.header_pattern          = 0xc3a3c3a
        self.trailer_pattern         = 0b001011
        self.channel_header_pattern  = 0x3c5c0 >> 2
        self.firmware_filler_pattern = 0x5555
        self.previous_event          = -1
        self.event_counter           = 0
        self.board_ids               = board_id
        self.files_to_process        = file_list
        self.save_nem                = save_nem
        self.nem_file                = None

        self.file_count = 0
        self.line_count = 0
        self.max_file_lines = 5e3

        self.in_event                = False
        self.eth_words_in_event      = -1
        self.words_in_event          = -1
        self.current_word            = -1
        self.event_number            = -1
        self.enabled_channels        = -1
        self.running_word            = None
        self.position_40bit          = 0
        self.current_channel         = -1
        self.in_40bit                = False
        self.data                    = {}
        self.version                 = None
        self.event_type              = None
        self.reset_params()

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
        self.in_40bit                = False
        self.data                    = {}
        self.version                 = None
        self.event_type              = None

    def open_next_file(self):
        if self.save_nem is not None:
            if self.nem_file is not None:
                self.nem_file.close()

            base_dir = self.save_nem.parent
            file_name = self.save_nem.stem
            suffix = self.save_nem.suffix

            file_name = f'{file_name}_{self.file_count}{suffix}'

            self.nem_file = open(base_dir / file_name, "w")
            self.file_count += 1
            self.line_count = 0
        else:
            self.nem_file = None
            self.line_count = 0
            self.file_count = 0

    def close_file(self):
        if self.save_nem is not None:
            if self.nem_file is not None:
                self.nem_file.close()
        self.nem_file = None
        self.line_count = 0
        self.file_count = 0

    def write_to_nem(self, write_str: str):
        if self.nem_file is not None:
            self.write_to_nem(write_str)
            self.line_count += 1

            if self.line_count >= self.max_file_lines:
                self.open_next_file()

    def div_ceil(self, x,y):
        return -(x//(-y))

    def decode_40bit(self, word):
        # Header
        if word >> 22 == self.channel_header_pattern and not self.in_40bit:
            self.current_channel += 1
            while not ((self.enabled_channels >> self.current_channel) & 0b1):
                self.current_channel += 1
                if self.current_channel > 3:
                    print('Found more headers than number of channels')
                    if self.nem_file is not None:
                        self.write_to_nem(f"THIS IS A BROKEN EVENT SINCE MORE HEADERS THAN MASK FOUND\n")
                    self.reset_params()
                    return
            self.bcid = (word & 0xfff)
            self.l1acounter = ((word >> 14) & 0xff)
            self.data[self.current_channel] = copy.deepcopy(self.data_template)
            self.in_40bit = True
            Type = (word >> 12) & 0x3

            if self.nem_file is not None:
                self.write_to_nem(f"H {self.current_channel} {self.l1acounter} 0b{Type:02b} {self.bcid}\n")
        # Data
        elif (word >> 39) == 1 and self.in_40bit:
            EA = (word >> 37) & 0b11
            ROW = (word >> 29) & 0b1111
            COL = (word >> 33) & 0b1111
            TOA = (word >> 19) & 0x3ff
            TOT = (word >> 10) & 0x1ff
            CAL = (word) & 0x3ff
            self.data[self.current_channel]['evt_number'].append(self.event_number)
            self.data[self.current_channel]['bcid'].append(self.bcid)
            self.data[self.current_channel]['l1a_counter'].append(self.l1acounter)
            self.data[self.current_channel]['evt'].append(self.event_counter)
            self.data[self.current_channel]['ea'].append(EA)
            self.data[self.current_channel]['board'].append(self.current_channel)
            self.data[self.current_channel]['row'].append(ROW)
            self.data[self.current_channel]['col'].append(COL)
            self.data[self.current_channel]['toa'].append(TOA)
            self.data[self.current_channel]['tot'].append(TOT)
            self.data[self.current_channel]['cal'].append(CAL)

            if self.nem_file is not None:
                self.write_to_nem(f"D {self.current_channel} 0b{EA:02b} {ROW} {COL} {TOA} {TOT} {CAL}\n")

        # Trailer
        elif (word >> 22) & 0x3ffff == self.board_ids[self.current_channel] and self.in_40bit:
            hits   = (word >> 8) & 0xff
            status = (word >> 16) & 0x3f
            CRC    = (word) & 0xff
            self.in_40bit = False

            if self.nem_file is not None:
                self.write_to_nem(f"T {self.current_channel} {status} {hits} 0b{CRC:08b}\n")

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
        if self.save_nem is not None:
            self.open_next_file()

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
                    if (word >> 4) == self.header_pattern:
                        self.enabled_channels = word & 0b1111
                        self.reset_params()
                        self.in_event = True
                        # print('Event header')
                        continue

                    # Event Header Line Two Found
                    elif(self.in_event and (self.words_in_event == -1) and (word >> 28 == self.firmware_key)):
                        self.current_word       = 0
                        self.event_type         = (word) & 0x3
                        self.event_number       = (word >> 12) & 0xffff
                        self.words_in_event     = (word >> 2) & 0x3ff
                        self.version            = (word >> 28) & 0xf
                        self.eth_words_in_event = self.div_ceil(40*self.words_in_event, 32)
                        # print(f"Num Words {self.words_in_event} & Eth Words {self.eth_words_in_event}")
                        # Set valid_data to true once we see fresh data
                        if(self.event_number==1 or self.event_number==0): self.valid_data = True
                        # print('Event Header Line Two Found')
                        # print(self.event_number)
                        if self.nem_file is not None:
                            self.write_to_nem(f"EH 0b{self.version:04b} {self.event_number} {self.words_in_event} {self.event_type:02b}\n")
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

                        crc            = (word) & 0xff
                        overflow_count = (word >> 11) & 0x7
                        hamming_count  = (word >> 8) & 0x7

                        if self.nem_file is not None:
                            self.write_to_nem(f"ET {self.event_number} {overflow_count} {hamming_count} 0b{crc:08b}\n")

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

                    # If Firmware filler
                    elif (word >> 16) == self.firmware_filler_pattern:
                        if self.nem_file is not None:
                            self.write_to_nem(f"Filler: 0b{word & 0xffff:016b}\n")

                    # Reset anyway!
                    self.reset_params()

                if len(self.data_to_load['evt']) > 0:
                    df = pd.concat([df, pd.DataFrame(self.data_to_load)], ignore_index=True)
                    self.data_to_load = copy.deepcopy(self.data_template)

        self.close_file()
        return df

## --------------- Decoding Class -----------------------



## --------------- Text converting to DataFrame -----------------------
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
        path_to_dir: str,
        dir_name_pattern: str,
        data_qinj: bool = False,
        save_to_csv: bool = False,
        debugging: bool = False,
    ):

    evt = -1
    previous_evt = -1
    name_pattern = "*translated*.nem"

    dirs = glob(f"{path_to_dir}/{dir_name_pattern}")
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

        if not df.empty:
            df = df.astype('int')
            if data_qinj:
                df.drop(columns=['evt', 'board'], inplace=True)
            if save_to_csv:
                df.to_csv(name+'.csv', index=False)
            else:
                df.to_feather(name+'.feather')
            del df

## --------------- Text converting to DataFrame -----------------------


## --------------- Modify DataFrame -----------------------
## --------------------------------------
def efficiency_with_single_board(
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
def efficiency_with_two_boards(
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
def singlehit_event_clear(
        input_df: pd.DataFrame,
        ignore_boards: list[int] = None
    ):

    ana_df = input_df
    if ignore_boards is not None:
        for board in ignore_boards:
            ana_df = ana_df.loc[ana_df['board'] != board].copy()

    ## event has one hit from each board
    event_board_counts = ana_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    event_selection_col = None
    for board in event_board_counts:
        if event_selection_col is None:
            event_selection_col = (event_board_counts[board] == 1)
        else:
            event_selection_col = event_selection_col & (event_board_counts[board] == 1)
    selected_event_numbers = event_board_counts[event_selection_col].index
    selected_subset_df = ana_df[ana_df['evt'].isin(selected_event_numbers)]
    selected_subset_df.reset_index(inplace=True, drop=True)

    del ana_df, event_board_counts, event_selection_col

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

## --------------- Modify DataFrame -----------------------



## --------------- Plotting -----------------------
## --------------------------------------
def return_hist(
        input_df: pd.DataFrame,
        chipNames: list[str],
        chipLabels: list[int],
        hist_bins: list = [50, 64, 64]
):
    h = {chipNames[boardID]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
        )
    for boardID in chipLabels}

    for boardID in chipLabels:
        tmp_df = input_df.loc[input_df['board'] == boardID]
        h[chipNames[boardID]].fill(tmp_df['cal'].values, tmp_df['tot'].values, tmp_df['toa'].values)

    return h

## --------------------------------------
def return_hist_pivot(
        input_df: pd.DataFrame,
        chipNames: list[str],
        chipLabels: list[int],
        hist_bins: list = [50, 64, 64]
):
    h = {chipNames[boardID]: hist.Hist(hist.axis.Regular(hist_bins[0], 140, 240, name="CAL", label="CAL [LSB]"),
                hist.axis.Regular(hist_bins[1], 0, 512,  name="TOT", label="TOT [LSB]"),
                hist.axis.Regular(hist_bins[2], 0, 1024, name="TOA", label="TOA [LSB]"),
        )
    for boardID in chipLabels}

    for boardID in chipLabels:
        h[chipNames[boardID]].fill(input_df['cal'][boardID].values, input_df['tot'][boardID].values, input_df['toa'][boardID].values)

    return h

## --------------------------------------
def plot_number_of_fired_board(
        input_df: pd.DataFrame,
        fig_tag: str = '',
        do_logy: bool = False,
        do_save_fig: bool = False,
        save_fig_path: str = None,
    ):

    h = hist.Hist(hist.axis.Regular(5, 0, 5, name="nBoards", label="nBoards"))
    h.fill(input_df.groupby('evt')['board'].nunique())

    fig = plt.figure(dpi=50, figsize=(14,12))
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
    ax.set_title(f"{fig_tag}", loc="right", size=25)
    h.plot1d(ax=ax, lw=2)
    ax.get_yaxis().get_offset_text().set_position((-0.05, 0))
    if do_logy:
        ax.set_yscale('log')
    plt.tight_layout()

    if do_save_fig:
        if save_fig_path is None:
            print('Please specify a path to save figure. Otherwise this figure will not be saved')
            del h
        else:
            save_path = Path(save_fig_path)
            # save_path = save_path / ''
            # fig.savefig()
    else:
        del h

## --------------------------------------
def plot_number_of_hits_per_event(
        input_df: pd.DataFrame,
        fig_titles: list[str],
        fig_tag: str = '',
        bins: int = 15,
        hist_range: tuple = (0, 15),
        do_logy: bool = False,
    ):
    hit_df = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    hists = {}

    for key in hit_df.columns:
        hists[key] = hist.Hist(hist.axis.Regular(bins, hist_range[0], hist_range[1], name="nHits", label='nHits'))
        hists[key].fill(hit_df[key])

    fig = plt.figure(dpi=100, figsize=(30,13))
    gs = fig.add_gridspec(2,2)

    for i, plot_info in enumerate(gs):
        ax = fig.add_subplot(plot_info)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=20)
        hists[i].plot1d(ax=ax, lw=2)
        ax.set_title(f"{fig_titles[i]} {fig_tag}", loc="right", size=18)
        ax.get_yaxis().get_offset_text().set_position((-0.05, 0))
        if do_logy:
            ax.set_yscale('log')

    plt.tight_layout()
    del hists, hit_df

## --------------------------------------
def plot_2d_nHits_nBoard(
        input_df: pd.DataFrame,
        fig_titles: list[str],
        fig_tag: str = '',
        bins: int = 15,
        hist_range: tuple = (0, 15),

    ):
    nboard_df = input_df.groupby('evt')['board'].nunique()
    hit_df = input_df.groupby(['evt', 'board']).size().unstack(fill_value=0)
    hit_df.dropna(subset=[0], inplace=True)
    hists = {}

    for key in hit_df.columns:
        hists[key] = hist.Hist(
            hist.axis.Regular(bins, hist_range[0], hist_range[1], name="nHits", label='nHits'),
            hist.axis.Regular(5, 0, 5, name="nBoards", label="nBoards")
        )
        hists[key].fill(hit_df[key], nboard_df)

    fig = plt.figure(dpi=100, figsize=(30,13))
    gs = fig.add_gridspec(2,2)

    for i, plot_info in enumerate(gs):
        ax = fig.add_subplot(plot_info)
        hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=20)
        hep.hist2dplot(hists[i], ax=ax, norm=colors.LogNorm())
        ax.set_title(f"{fig_titles[i]} {fig_tag}", loc="right", size=18)

    plt.tight_layout()
    del hists, hit_df, nboard_df

## --------------------------------------
def plot_heatmap_byPandas(
        input_df: pd.DataFrame,
        chipLabels: list[int],
        fig_title: list[str],
        fig_tag: str,
        exclude_noise: bool = False,
    ):

    if exclude_noise:
        ana_df = input_df.loc[input_df['tot'] > 10].copy()
    else:
        ana_df = input_df

    # Group the DataFrame by 'col,' 'row,' and 'board,' and count the number of hits in each group
    hits_count_by_col_row_board = ana_df.groupby(['col', 'row', 'board'])['evt'].count().reset_index()
    del ana_df

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
        ax.set_title(f"{fig_title[idx]}, Occupancy map {fig_tag}", loc="right", size=20)
        ax.tick_params(axis='x', which='both', length=5, labelsize=17)
        ax.tick_params(axis='y', which='both', length=5, labelsize=17)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.minorticks_off()

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

## --------------------------------------
def plot_correlation_of_pixels(
        input_df: pd.DataFrame,
        hit_type: str,
        board_id_to_correlate: int,
        fig_title: str,
        fit_tag: str = '',
    ):

    if board_id_to_correlate == 0:
        print("Self correlation!!")

    xaxis_label = None
    if (board_id_to_correlate == 1):
        xaxis_label = 'DUT 1'
    elif (board_id_to_correlate == 3):
        xaxis_label = 'DUT 2'
    else:
        xaxis_label = 'Reference Board'

    h_row = hist.Hist(
        hist.axis.Regular(16, 0, 16, name='row1', label='Trigger Board Row'),
        hist.axis.Regular(16, 0, 16, name='row2', label=f'{xaxis_label} Row'),
    )
    h_col = hist.Hist(
        hist.axis.Regular(16, 0, 16, name='col1', label='Trigger Board Col'),
        hist.axis.Regular(16, 0, 16, name='col2', label=f'{xaxis_label} Col'),
    )

    if hit_type == "single":
        h_row.fill(input_df.loc[input_df['board'] == 0]['row'], input_df.loc[input_df['board'] == board_id_to_correlate]['row'])
        h_col.fill(input_df.loc[input_df['board'] == 0]['col'], input_df.loc[input_df['board'] == board_id_to_correlate]['col'])

    elif hit_type == "multiple":
        n = int(0.01*input_df.shape[0]) # ~100k events
        indices = np.random.choice(input_df['evt'].unique(), n, replace=False)
        test_df = input_df.loc[input_df['evt'].isin(indices)]

        for name, group in test_df.groupby('evt'):
            cnt = len(group[group['board'] == board_id_to_correlate]['row'])
            broadcasted_trig_row = np.full(cnt, group.loc[group['board'] == 0]['row'].values)
            broadcasted_trig_col = np.full(cnt, group.loc[group['board'] == 0]['col'].values)
            h_row.fill(broadcasted_trig_row, group.loc[group['board'] == board_id_to_correlate]['row'].to_numpy())
            h_col.fill(broadcasted_trig_col, group.loc[group['board'] == board_id_to_correlate]['col'].to_numpy())

    else:
        print('Please specify hit_type. Either single or multiple')
        return

    location = np.arange(0, 16) + 0.5
    tick_labels = np.char.mod('%d', np.arange(0, 16))
    fig, ax = plt.subplots(1, 2, dpi=100, figsize=(23, 11))

    hep.hist2dplot(h_row, ax=ax[0])
    hep.cms.text(loc=0, ax=ax[0], text="Preliminary", fontsize=25)
    ax[0].set_title(f"{fig_title} {fit_tag}", loc="right", size=15)
    ax[0].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[0].xaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[0].xaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[0].yaxis.set_major_formatter(ticker.NullFormatter())
    ax[0].yaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[0].yaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[0].tick_params(axis='both', which='major', length=0)

    hep.hist2dplot(h_col, ax=ax[1])
    hep.cms.text(loc=0, ax=ax[1], text="Preliminary", fontsize=25)
    ax[1].set_title(f"{fig_title} {fit_tag}", loc="right", size=15)
    ax[1].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[1].xaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[1].xaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[1].yaxis.set_major_formatter(ticker.NullFormatter())
    ax[1].yaxis.set_minor_locator(ticker.FixedLocator(location))
    ax[1].yaxis.set_minor_formatter(ticker.FixedFormatter(tick_labels))
    ax[1].tick_params(axis='both', which='major', length=0)

    plt.tight_layout()

## --------------------------------------
def plot_distance(
        input_df: pd.DataFrame,
        hit_type: str,
        board_id_to_correlate: int,
        fig_title: str,
        fig_tag: str = '',
        do_logy: bool = False,
    ):

    xaxis_label = None
    if (board_id_to_correlate == 1):
        xaxis_label = 'DUT 1'
    elif (board_id_to_correlate == 3):
        xaxis_label = 'DUT 2'
    else:
        xaxis_label = 'Reference'

    h_dis = hist.Hist(hist.axis.Regular(32, 0, 32, name='dis', label=f'Distance (Trigger - {xaxis_label})'))

    if hit_type == "single":
        diff_row = (input_df.loc[input_df['board'] == 0]['row'].values - input_df.loc[input_df['board'] == board_id_to_correlate]['row'].values)
        diff_col = (input_df.loc[input_df['board'] == 0]['col'].values - input_df.loc[input_df['board'] == board_id_to_correlate]['col'].values)
        dis = np.sqrt(diff_row**2 + diff_col**2)
        h_dis.fill(dis)
        del diff_row, diff_col, dis

    elif hit_type == "multiple":
        n = int(0.01*input_df.shape[0]) # ~100k events
        indices = np.random.choice(input_df['evt'].unique(), n, replace=False)
        test_df = input_df.loc[input_df['evt'].isin(indices)]

        for name, group in test_df.groupby('evt'):
            cnt = len(group.loc[group['board'] == board_id_to_correlate]['row'])
            broadcasted_trig_row = np.full(cnt, group.loc[group['board'] == 0]['row'].values)
            broadcasted_trig_col = np.full(cnt, group.loc[group['board'] == 0]['col'].values)
            diff_row = (broadcasted_trig_row - group.loc[group['board'] == board_id_to_correlate]['row'].values)
            diff_col = (broadcasted_trig_col - group.loc[group['board'] == board_id_to_correlate]['col'].values)
            dis = np.sqrt(diff_row**2 + diff_col**2)
            h_dis.fill(dis)

        del test_df, indices, n

    else:
        print('Please specify hit_type. Either single or multiple')
        return

    fig, ax = plt.subplots(dpi=100, figsize=(15, 8))
    hep.histplot(h_dis, ax=ax)
    hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
    ax.set_title(f"{fig_title} {fig_tag}", loc="right", size=15)

    if do_logy:
        ax.set_yscale('log')

    return h_dis

## --------------------------------------
def plot_resolution_with_pulls(
        input_df: pd.DataFrame,
        board_ids: list[int],
        fig_title: list[str],
        fig_tag: str = '',
        hist_bins: int = 15,
    ):
    import matplotlib.gridspec as gridspec
    from lmfit.models import GaussianModel
    from lmfit.lineshapes import gaussian

    mod = GaussianModel(nan_policy='omit')

    hists = {}
    fit_params = {}
    pulls_dict = {}

    for key in board_ids:
        hist_x_min = int(input_df[f'res{key}'].min())-5
        hist_x_max = int(input_df[f'res{key}'].max())+5
        hists[key] = hist.Hist(hist.axis.Regular(hist_bins, hist_x_min, hist_x_max, name="time_resolution", label=r'Time Resolution [ps]'))
        hists[key].fill(input_df[f'res{key}'].values)
        centers = hists[key].axes[0].centers
        pars = mod.guess(hists[key].values(), x=centers)
        out = mod.fit(hists[key].values(), pars, x=centers, weights=1/np.sqrt(hists[key].values()))
        fit_params[key] = out

        ### Calculate pull
        pulls = (hists[key].values() - out.eval(x=centers))/np.sqrt(out.eval(x=centers))
        pulls[np.isnan(pulls) | np.isinf(pulls)] = 0
        pulls_dict[key] = pulls

    # Create a figure with a 2x2 grid
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 2)

    for i, plot in enumerate(gs):
        global_ax = fig.add_subplot(plot)
        inner_plot = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=global_ax, hspace=0, height_ratios=[3, 1])
        main_ax = fig.add_subplot(inner_plot[0])
        sub_ax = fig.add_subplot(inner_plot[1], sharex=main_ax)
        global_ax.xaxis.set_visible(False)
        global_ax.yaxis.set_visible(False)
        main_ax.xaxis.set_visible(False)

        if i not in hists:
            global_ax.set_axis_off()
            main_ax.set_axis_off()
            sub_ax.set_axis_off()
            continue

        centers = hists[i].axes[0].centers
        hep.cms.text(loc=0, ax=main_ax, text="Preliminary", fontsize=20)
        main_ax.set_title(f'{fig_title[i]} {fig_tag}', loc="right", size=13)

        main_ax.errorbar(centers, hists[i].values(), np.sqrt(hists[i].variances()),
                        ecolor="steelblue", mfc="steelblue", mec="steelblue", fmt="o",
                        ms=6, capsize=1, capthick=2, alpha=0.8)
        main_ax.set_ylabel('Counts', fontsize=20)
        main_ax.set_ylim(-5, None)
        main_ax.tick_params(axis='x', labelsize=20)
        main_ax.tick_params(axis='y', labelsize=20)

        x_min = centers[0]
        x_max = centers[-1]

        x_range = np.linspace(x_min, x_max, 200)
        popt = [par for name, par in fit_params[i].best_values.items()]
        pcov = fit_params[i].covar

        if np.isfinite(pcov).all():
            n_samples = 100
            vopts = np.random.multivariate_normal(popt, pcov, n_samples)
            sampled_ydata = np.vstack([gaussian(x_range, *vopt).T for vopt in vopts])
            model_uncert = np.nanstd(sampled_ydata, axis=0)
        else:
            model_uncert = np.zeros_like(np.sqrt(hists[i].variances()))

        main_ax.plot(x_range, fit_params[i].eval(x=x_range), color="hotpink", ls="-", lw=2, alpha=0.8,
                    label=fr"$\mu$:{fit_params[i].params['center'].value:.2f} $\pm$ {fit_params[i].params['center'].stderr:.2f}")
        main_ax.plot(np.NaN, np.NaN, color='none',
                     label=fr"$\sigma$: {abs(fit_params[i].params['sigma'].value):.2f} $\pm$ {abs(fit_params[i].params['sigma'].stderr):.2f}")

        main_ax.fill_between(
            x_range,
            fit_params[i].eval(x=x_range) - model_uncert,
            fit_params[i].eval(x=x_range) + model_uncert,
            color="hotpink",
            alpha=0.2,
            label='Uncertainty'
        )
        main_ax.legend(fontsize=20, loc='upper right')

        width = (x_max - x_min) / len(pulls_dict[i])
        sub_ax.axhline(1, c='black', lw=0.75)
        sub_ax.axhline(0, c='black', lw=1.2)
        sub_ax.axhline(-1, c='black', lw=0.75)
        sub_ax.bar(centers, pulls_dict[i], width=width, fc='royalblue')
        sub_ax.set_ylim(-2, 2)
        sub_ax.set_yticks(ticks=np.arange(-1, 2), labels=[-1, 0, 1], fontsize=20)
        sub_ax.set_xlabel(r'Time Resolution [ps]', fontsize=20)
        sub_ax.tick_params(axis='x', which='both', labelsize=20)
        sub_ax.set_ylabel('Pulls', fontsize=20, loc='center')

    del hists, fit_params, pulls_dict, mod

## --------------------------------------
def plot_resolution_table(
        input_df: pd.DataFrame,
        chipLabels: list[int],
        fig_title: list[str],
        fig_tag: str = '',
        slides_friendly: bool = False,
    ):

    from matplotlib import colormaps
    cmap = colormaps['viridis']
    cmap.set_under(color='lightgrey')

    tables = {}
    for board_id in chipLabels:
        board_info = input_df[[f'row{board_id}', f'col{board_id}', f'res{board_id}', f'err{board_id}']]

        res = board_info.groupby([f'row{board_id}', f'col{board_id}']).apply(lambda x: np.average(x[f'res{board_id}'], weights=1/x[f'err{board_id}']**2)).reset_index()
        err = board_info.groupby([f'row{board_id}', f'col{board_id}']).apply(lambda x: np.sqrt(1/(np.sum(1/x[f'err{board_id}']**2)))).reset_index()

        res_table = res.pivot_table(index=f'row{board_id}', columns=f'col{board_id}', values=0, fill_value=-1)
        err_table = err.pivot_table(index=f'row{board_id}', columns=f'col{board_id}', values=0, fill_value=-1)

        res_table = res_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        res_table = res_table.reindex(columns=np.arange(0,16))
        res_table = res_table.fillna(-1)

        err_table = err_table.reindex(pd.Index(np.arange(0,16), name='')).reset_index()
        err_table = err_table.reindex(columns=np.arange(0,16))
        err_table = err_table.fillna(-1)

        tables[board_id] = [res_table, err_table]

    if slides_friendly:
        fig = plt.figure(figsize=(35, 35))
        gs = fig.add_gridspec(2, 2)

        for idx, plot in enumerate(gs):
            ax = fig.add_subplot(plot)

            if idx not in tables:
                ax.set_axis_off()
                continue
            im = ax.imshow(tables[idx][0], cmap=cmap, interpolation="nearest", vmin=30, vmax=85)

            # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Time Resolution', fontsize=20)

            for i in range(16):
                for j in range(16):
                    value = tables[idx][0].iloc[i, j]
                    error = tables[idx][1].iloc[i, j]
                    if value == -1: continue
                    text_color = 'black' if value > (res_table.values.max() + res_table.values.min()) / 2 else 'white'
                    text = str(rf"{value:.1f}""\n"fr"$\pm$ {error:.1f}")
                    plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=20, rotation=45)


            hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
            ax.set_xlabel('Column (col)', fontsize=20)
            ax.set_ylabel('Row (row)', fontsize=20)
            ticks = range(0, 16)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_title(f"{fig_title[idx]}, Resolution map {fig_tag}", loc="right", size=20)
            ax.tick_params(axis='x', which='both', length=5, labelsize=20)
            ax.tick_params(axis='y', which='both', length=5, labelsize=20)
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.minorticks_off()

        plt.tight_layout()
        del tables

    else:
        for idx in tables.keys():
            # Create a heatmap to visualize the count of hits
            fig, ax = plt.subplots(dpi=100, figsize=(20, 20))
            ax.cla()
            im = ax.imshow(tables[idx][0], cmap=cmap, interpolation="nearest", vmin=30, vmax=85)

            # Add color bar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Time Resolution', fontsize=20)

            for i in range(16):
                for j in range(16):
                    value = tables[idx][0].iloc[i, j]
                    error = tables[idx][1].iloc[i, j]
                    if value == -1: continue
                    text_color = 'black' if value > (res_table.values.max() + res_table.values.min()) / 2 else 'white'
                    text = str(rf"{value:.1f}""\n"fr"$\pm$ {error:.1f}")
                    plt.text(j, i, text, va='center', ha='center', color=text_color, fontsize=20, rotation=45)

            hep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
            ax.set_xlabel('Column (col)', fontsize=20)
            ax.set_ylabel('Row (row)', fontsize=20)
            ticks = range(0, 16)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_title(f"{fig_title[board_id]}, Resolution map {fig_tag}", loc="right", size=20)
            ax.tick_params(axis='x', which='both', length=5, labelsize=17)
            ax.tick_params(axis='y', which='both', length=5, labelsize=17)
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.minorticks_off()
            plt.tight_layout()

        del tables

## --------------- Plotting -----------------------



## --------------- Time Walk Correction -----------------------
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
            corr_toas[f'toa_b3'] = corr_b2

    return corr_toas

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

## --------------- Time Walk Correction -----------------------



## --------------- Result -----------------------
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
    ):

    results = {
        0: np.sqrt((1/6)*(2*fit_params['01'][0]**2+2*fit_params['02'][0]**2+2*fit_params['03'][0]**2-fit_params['12'][0]**2-fit_params['13'][0]**2-fit_params['23'][0]**2))*1e3,
        1: np.sqrt((1/6)*(2*fit_params['01'][0]**2+2*fit_params['12'][0]**2+2*fit_params['13'][0]**2-fit_params['02'][0]**2-fit_params['03'][0]**2-fit_params['23'][0]**2))*1e3,
        2: np.sqrt((1/6)*(2*fit_params['02'][0]**2+2*fit_params['12'][0]**2+2*fit_params['23'][0]**2-fit_params['01'][0]**2-fit_params['03'][0]**2-fit_params['13'][0]**2))*1e3,
        3: np.sqrt((1/6)*(2*fit_params['03'][0]**2+2*fit_params['13'][0]**2+2*fit_params['23'][0]**2-fit_params['01'][0]**2-fit_params['02'][0]**2-fit_params['12'][0]**2))*1e3,
    }

    return results

## --------------- Result -----------------------



## --------------- Bootstrap -----------------------
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
        iteration = 100
        sampling_fraction = 0.75
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
                    tdc_cuts[idx] = [try_df['cal'][idx].mode()[0]-3, try_df['cal'][idx].mode()[0]+3,  100, 500, 0, 600]
                else:
                    tdc_cuts[idx] = [try_df['cal'][idx].mode()[0]-3, try_df['cal'][idx].mode()[0]+3,  0, 1100, 0, 600]

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

            corr_toas = three_board_iterative_timewalk_correction(df_in_time, 5, 3, board_list=board_to_analyze)

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

                resolutions = return_resolution_three_board(fit_params_lmfit, var=list(fit_params_lmfit.keys()), board_list=board_to_analyze)

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

## --------------- Bootstrap -----------------------
