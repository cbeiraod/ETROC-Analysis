#############################################################################
# zlib License
#
# (C) 2024 Cristóvão Beirão da Cruz e Silva <cbeiraod@cern.ch>
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#############################################################################

from pathlib import Path
import sqlite3
import pandas
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import mplhep
import pickle
import numpy
import tempfile
import shutil
import subprocess
from math import floor
from math import ceil
import io

run_db = {
    "27Jan2024": {},
    "14Feb2024": {},
}

def plotVRefPower(dataframe: pandas.DataFrame, title: str, save_to: Path, show: bool = False, times_to_plot: dict[str, datetime.datetime] = {}):
    figure, axis = plt.subplots(
                    nrows = 2,
                    ncols = 1,
                    sharex='col',
                    layout='constrained',
                    figsize=(16, 14),
                )

    figure.suptitle(title)
    mplhep.cms.text(loc=0, ax=axis[0], text='Preliminary', fontsize=25)
    mplhep.cms.text(loc=0, ax=axis[1], text='Preliminary', fontsize=25)

    dataframe.plot(
                    x = 'Time',
                    y = 'V',
                    kind = 'scatter',
                    ax=axis[0],
                )

    dataframe.plot(
                    x = 'Time',
                    y = 'I',
                    kind = 'scatter',
                    ax=axis[1],
                )

    axis[0].title.set_text('VRef Voltage over Time')
    axis[1].title.set_text('VRef Current over Time')

    date_form = DateFormatter("%H:%M")
    axis[0].xaxis.set_major_formatter(date_form)
    axis[1].xaxis.set_major_formatter(date_form)

    for key in times_to_plot:
        color = 'c'
        if key == "Run Start":
            color = 'g'
        if key == "Run Stop":
            color = 'r'
        if "Config" in key:
            color = 'y'

        trans = axis[0].get_xaxis_transform()
        axis[0].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[0].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

        trans = axis[1].get_xaxis_transform()
        axis[1].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[1].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

    plt.savefig(fname=save_to/"VRef.pdf")
    if show:
        plt.show()
    else:
        plt.clf()

def plotBoardPower(board: str, channels: dict[str, str], dataframe: pandas.DataFrame, title: str, save_to: Path, show: bool = False, times_to_plot: dict[str, datetime.datetime] = {}):
    figure, axis = plt.subplots(
                    nrows = 2,
                    ncols = 2,
                    sharex='col',
                    layout='constrained',
                    figsize=(32, 14),
                )

    figure.suptitle(title)
    mplhep.cms.text(loc=0, ax=axis[0][0], text='Preliminary', fontsize=25)
    mplhep.cms.text(loc=0, ax=axis[0][1], text='Preliminary', fontsize=25)
    mplhep.cms.text(loc=0, ax=axis[1][0], text='Preliminary', fontsize=25)
    mplhep.cms.text(loc=0, ax=axis[1][1], text='Preliminary', fontsize=25)

    digital_df = dataframe.loc[dataframe['Channel'] == channels["Digital"]].copy()
    analog_df = dataframe.loc[dataframe['Channel'] == channels["Analog"]].copy()

    analog_df.plot(
                    x = 'Time',
                    y = 'V',
                    kind = 'scatter',
                    ax=axis[0][0],
                )
    analog_df.plot(
                    x = 'Time',
                    y = 'I',
                    kind = 'scatter',
                    ax=axis[0][1],
                )
    digital_df.plot(
                    x = 'Time',
                    y = 'V',
                    kind = 'scatter',
                    ax=axis[1][0],
                )
    digital_df.plot(
                    x = 'Time',
                    y = 'I',
                    kind = 'scatter',
                    ax=axis[1][1],
                )

    axis[0][0].yaxis.get_major_formatter().set_useOffset(False)
    axis[0][1].yaxis.get_major_formatter().set_useOffset(False)
    axis[1][0].yaxis.get_major_formatter().set_useOffset(False)
    axis[1][1].yaxis.get_major_formatter().set_useOffset(False)

    axis[0][0].title.set_text(f'{board} Analog Voltage over Time')
    axis[0][1].title.set_text(f'{board} Analog Current over Time')
    axis[1][0].title.set_text(f'{board} Digital Voltage over Time')
    axis[1][1].title.set_text(f'{board} Digital Current over Time')

    date_form = DateFormatter("%H:%M")
    axis[0][0].xaxis.set_major_formatter(date_form)
    axis[0][1].xaxis.set_major_formatter(date_form)
    axis[1][0].xaxis.set_major_formatter(date_form)
    axis[1][1].xaxis.set_major_formatter(date_form)

    for key in times_to_plot:
        color = 'c'
        if key == "Run Start":
            color = 'g'
        if key == "Run Stop":
            color = 'r'
        if "Test" in key:
            color = 'gold'
        if "Extra Run Start" in key:
            color = 'aquamarine'
        if "Extra Run Stop" in key:
            color = 'lightcoral'
        if "Config" in key:
            color = 'y'

        trans = axis[0][0].get_xaxis_transform()
        axis[0][0].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[0][0].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

        trans = axis[0][1].get_xaxis_transform()
        axis[0][1].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[0][1].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

        trans = axis[1][0].get_xaxis_transform()
        axis[1][0].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[1][0].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

        trans = axis[1][1].get_xaxis_transform()
        axis[1][1].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[1][1].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

    board_safe = board.replace(" ", "_")
    plt.savefig(fname=save_to/f"{board_safe}.pdf")
    if show:
        plt.show()
    else:
        plt.clf()

def plotWSPower(dataframe: pandas.DataFrame, title: str, save_to: Path, show: bool = False, times_to_plot: dict[str, datetime.datetime] = {}, filename = "WaveformSampler"):
    figure, axis = plt.subplots(
                    nrows = 2,
                    ncols = 1,
                    sharex='col',
                    layout='constrained',
                    figsize=(16, 14),
                )

    figure.suptitle(title)
    mplhep.cms.text(loc=0, ax=axis[0], text='Preliminary', fontsize=25)
    mplhep.cms.text(loc=0, ax=axis[1], text='Preliminary', fontsize=25)

    dataframe.plot(
                    x = 'Time',
                    y = 'V',
                    kind = 'scatter',
                    ax=axis[0],
                )

    dataframe.plot(
                    x = 'Time',
                    y = 'I',
                    kind = 'scatter',
                    ax=axis[1],
                )

    axis[0].title.set_text('Waveform Sampler Voltage over Time')
    axis[1].title.set_text('Waveform Sampler Current over Time')

    date_form = DateFormatter("%H:%M")
    axis[0].xaxis.set_major_formatter(date_form)
    axis[1].xaxis.set_major_formatter(date_form)

    for key in times_to_plot:
        color = 'c'
        if key == "Run Start":
            color = 'g'
        if key == "Run Stop":
            color = 'r'
        if "Config" in key:
            color = 'y'

        trans = axis[0].get_xaxis_transform()
        axis[0].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[0].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

        trans = axis[1].get_xaxis_transform()
        axis[1].axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis[1].text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

    plt.savefig(fname=save_to/filename+".pdf")
    if show:
        plt.show()
    else:
        plt.clf()

def makePerRunPlots(
        dataframe: pandas.DataFrame,
        run_to_plot_info: dict,
        base_dir: Path,
        power_connections: dict[str, dict[str, str]],
        previous_run_info: dict | None = None,
        all_run_info: list[dict] | None = None,
        extra_run_info: list[dict] | None = None,
        test_run_info: list[dict] | None = None,
                    ):
    run_dir = base_dir/run_to_plot_info["name"]
    run_dir.mkdir(exist_ok=True)

    times_to_plot = {
        "Run Start": run_to_plot_info["start"],
        "Run Stop": run_to_plot_info["stop"],
    }

    if previous_run_info is not None:
        times_to_plot["Previous Run Stop"] = previous_run_info["stop"]
    elif "previous_run" in run_to_plot_info:
        if all_run_info is not None:
            for this_run_info in all_run_info:
                if this_run_info["name"] == run_to_plot_info["previous_run"]:
                    times_to_plot["Previous Run Stop"] = this_run_info["stop"]
        else:
            raise RuntimeError("All run info is not set, however you are trying to look up the previous run information")

    if "extra_run_info" in run_to_plot_info:
        if extra_run_info is not None:
            for extra_run_name in run_to_plot_info["extra_run_info"]:
                for this_run_info in extra_run_info:
                    if this_run_info["name"] == extra_run_name:
                        times_to_plot[f'Extra Run Start - {this_run_info["name"]}'] = this_run_info["start"]
                        times_to_plot[f'Extra Run Stop - {this_run_info["name"]}']  = this_run_info["stop"]
        else:
            raise RuntimeError("Extra run info is not set, however you are trying to look up the extra run information")

    for board_idx in range(len(run_to_plot_info["boards"])):
        if run_to_plot_info["pre_config_times"][board_idx] is not None:
            times_to_plot[f'Save {run_to_plot_info["boards"][board_idx]} Pre-Run Config'] = run_to_plot_info["pre_config_times"][board_idx]
        if run_to_plot_info["post_config_times"][board_idx] is not None:
            times_to_plot[f'Save {run_to_plot_info["boards"][board_idx]} Post-Run Config'] = run_to_plot_info["post_config_times"][board_idx]

    record_start = min(times_to_plot.values())
    record_end = max(times_to_plot.values())

    record_start -= datetime.timedelta(seconds = 40)
    record_end   += datetime.timedelta(seconds = 40)
    if "extra_begin" in run_to_plot_info:
        record_start = min(record_start, run_to_plot_info["extra_begin"])

    if test_run_info is not None:
        for this_test_info in test_run_info:
            test_start = this_test_info["start"]
            if test_start > record_start and test_start < record_end:
                times_to_plot[f'Test {this_test_info["name"]}'] = test_start

    run_df = dataframe.loc[dataframe['Time'] >= record_start]
    run_df = run_df.loc[run_df['Time'] <= record_end].copy()

    plotVRefPower(run_df.loc[run_df['Channel'] == "VRef"], f'{run_to_plot_info["name"]} VRef', run_dir, times_to_plot=times_to_plot)
    plotWSPower(run_df.loc[run_df['Channel'] == "WS"], f'{run_to_plot_info["name"]} Waveform Sampler', run_dir, times_to_plot=times_to_plot)

    for board in run_to_plot_info["boards"]:
        plotBoardPower(board, power_connections[board], run_df, f'{run_to_plot_info["name"]} {board} Power over Time', run_dir, times_to_plot=times_to_plot)

def plotPixelsOverTime(data_df: pandas.DataFrame, var: str, title: str, scan_list: list[tuple[int, int]], save_to: Path, show: bool = False, times_to_plot: dict[str, datetime.datetime] = {}, time_format: str = "%H:%M"):
    figure, axis = plt.subplots(figsize=(16,7), layout='constrained',)
    axis.set_prop_cycle(color=['#e41a1c','#fdbf6f','#d95f02', '#377eb8','#4daf4a','#b2df8a',])

    mplhep.cms.text(loc=0, ax=axis, text="Preliminary", fontsize=25)
    #date_form = DateFormatter("%Y-%m-%d %H:%M")
    date_form = DateFormatter(time_format)
    axis.xaxis.set_major_formatter(date_form)
    #plt.xticks(rotation=60)

    axis.title.set_text(title)

    for row, col in scan_list:
        filtered_df: pandas.DataFrame = data_df.loc[(data_df['row'] == row) & (data_df['col'] == col)]

        axis.plot(filtered_df['Time'], filtered_df[var], '.-', label=f'Row {row}, Col {col}')

    axis.legend(shadow=False, fancybox=True)

    for key in times_to_plot:
        color = 'c'
        if key == "Run Start":
            color = 'g'
        if key == "Run Stop":
            color = 'r'
        if "Config" in key:
            color = 'y'

        trans = axis.get_xaxis_transform()
        axis.axvline(
            x = times_to_plot[key],
            color = color,
        )
        axis.text(times_to_plot[key], .5, key, transform=trans, rotation=90, va='center')

    plt.savefig(fname=save_to)
    if show:
        plt.show()
    else:
        plt.clf()

def diff_chip_configs(config_file1: Path, config_file2: Path):
    with open(config_file1, 'rb') as f:
        loaded_obj1 = pickle.load(f)
    with open(config_file2, 'rb') as f:
        loaded_obj2 = pickle.load(f)

    if loaded_obj1['chip'] != loaded_obj2['chip']:
        raise RuntimeError("The config files are for different chips.")

    chip1: dict = loaded_obj1['object']
    chip2: dict = loaded_obj2['object']

    retVal = {}

    common_keys = []
    for key in chip1.keys():
        if key not in chip2:
            raise RuntimeError(f"Address Space \"{key}\" in config file 1 and not in config file 2")
        else:
            if key not in common_keys:
                common_keys += [key]
    for key in chip2.keys():
        if key not in chip1:
            raise RuntimeError(f"Address Space \"{key}\" in config file 2 and not in config file 1")
        else:
            if key not in common_keys:
                common_keys += [key]

    for address_space_name in common_keys:
        retVal[address_space_name] = {}
        if len(chip1[address_space_name]) != len(chip2[address_space_name]):
            raise RuntimeError(f"The length of the \"{address_space_name}\" memory for config file 1 ({len(chip1[address_space_name])}) is not the same as for config file 2 ({len(chip2[address_space_name])})")

        length = min(len(chip1[address_space_name]), len(chip2[address_space_name]))

        for idx in range(length):
            if chip1[address_space_name][idx] != chip2[address_space_name][idx]:
                retVal[address_space_name][idx] = (chip1[address_space_name][idx], chip2[address_space_name][idx])

    return retVal

def save_changed_config(changed_registers: dict[str, dict[int, tuple[int]]], base_dir: Path, base_name: str, save_broadcast: bool = False, save_extra_pixel_stat: bool = False):
    with open(base_dir/(base_name+".pickle"), "wb") as file:
        pickle.dump(changed_registers, file)

    with open(base_dir/(base_name+".txt"), "w") as file:
        for address_space in changed_registers:
            if address_space == "Waveform Sampler":
                continue
            file.write(f'Address Space - {address_space}:\n')
            for reg in changed_registers[address_space]:
                reg_name = "Unknown"
                broadcast = ""
                if reg < 0x0020:
                    reg_name = f'PeriCfg{reg}'
                elif reg < 0x0022:
                    reg_name = f'Magic Number ({reg:#06x})'
                elif reg < 0x0100:
                    reg_name = f'Unnamed Blk1 ({reg:#06x})'
                elif reg < 0x0110:
                    reg_name = f'PeriStat{reg - 0x0100}'
                elif reg < 0x0120:
                    reg_name = f'Unnamed Blk2 ({reg:#06x})'
                elif reg < 0x0124:
                    reg_name = f'SEU Counter {reg - 0x0120}'
                elif reg < 0x8000:
                    reg_name = f'Unnamed Blk3 ({reg:#06x})'
                else:
                    space = "Cfg"
                    if (reg & 0x4000) != 0:
                        space = "Stat"
                    broadcast = ""
                    if (reg & 0x2000) != 0:
                        broadcast = " broadcast"
                    register = reg & 0x1f
                    row = (reg >> 5) & 0xf
                    col = (reg >> 9) & 0xf
                    reg_name = f'Pix{space}{register} Pixel {row},{col}'

                    if not save_broadcast and broadcast != "":  # Broadcast reading is a copy of non broadcast registers
                        continue
                    if not save_extra_pixel_stat and space == "Stat" and register > 7:  # Stat registers above 7 are a copy of stat registers below
                        continue
                file.write(f'{reg_name}{broadcast} - {changed_registers[address_space][reg][0]:#010b} to {changed_registers[address_space][reg][1]:#010b}\n')

def get_pixel_bitflip_map(changed_registers: dict[str, dict[int, tuple[int]]], count_broadcast: bool = False, count_extra_pixel_stat: bool = False):
    status_map = [[0 for _ in range(16)] for _ in range(16)]
    config_map = [[0 for _ in range(16)] for _ in range(16)]

    for address_space in changed_registers:
        if address_space == "Waveform Sampler":
            continue
        for reg in changed_registers[address_space]:
                if reg < 0x8000:
                    continue
                else:
                    space = "Cfg"
                    if (reg & 0x4000) != 0:
                        space = "Stat"
                    broadcast = ""
                    if (reg & 0x2000) != 0:
                        broadcast = " broadcast"
                    register = reg & 0x1f
                    row = (reg >> 5) & 0xf
                    col = (reg >> 9) & 0xf

                    if not count_broadcast and broadcast != "":  # Broadcast reading is a copy of non broadcast registers
                        continue
                    if not count_extra_pixel_stat and space == "Stat" and register > 7:  # Stat registers above 7 are a copy of stat registers below
                        continue

                    reg_diff = changed_registers[address_space][reg][0] ^ changed_registers[address_space][reg][1]
                    if space == "Stat":
                        status_map[row][col] += reg_diff.bit_count()
                    else:
                        config_map[row][col] += reg_diff.bit_count()

    return status_map, config_map

def min_map(map: list[list[int]]):
    min_val = map[0][0]

    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] < min_val:
                min_val = map[i][j]

    return min_val

def max_map(map: list[list[int]]):
    max_val = map[0][0]

    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] > max_val:
                max_val = map[i][j]

    return max_val

def plot_map(map: list[list[int]], title: str, color_label: str, save_dir: Path | None):
    map_data = numpy.array((map))
    fig, ax = plt.subplots(dpi=100, figsize=(20, 20))
    ax.cla()
    im = ax.imshow(map_data, cmap="viridis", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    min_val = min_map(map)
    max_val = max_map(map)

    for row in range(16):
        for col in range(16):
            value = map[row][col]
            if value == -1: continue
            text_color = 'black' if value > 0.5*(max_val + min_val) else 'white'
            text = str("{:.0f}".format(value))
            plt.text(col, row, text, va='center', ha='center', color=text_color, fontsize=17)

    mplhep.cms.text(loc=0, ax=ax, text="Preliminary", fontsize=25)
    ax.set_xlabel('Column (col)', fontsize=20)
    ax.set_ylabel('Row (row)', fontsize=20)
    ticks = range(0, 16)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title(title, loc="right", size=20)
    ax.tick_params(axis='x', which='both', length=5, labelsize=17)
    ax.tick_params(axis='y', which='both', length=5, labelsize=17)
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.minorticks_off()

    if save_dir is not None:
        fig.savefig(save_dir)
        plt.close(fig)

def fill_bitmap(
        changed_registers: dict[str, dict[int, tuple[int]]],
        bitmap: dict[str, list[int]],
        unused_bitmap: dict[str, list[int]] = {},
        save_broadcast: bool = False,
        save_extra_pixel_stat: bool = False,
        save_seu_counter: bool = False,
                ):
    for reg in changed_registers["ETROC2"]:
        reg_name = None
        broadcast = ""
        if reg < 0x0020:
            reg_name = f'PeriCfg{reg}'
        elif reg < 0x0022:
            reg_name = f'Magic Number ({reg:#06x})'
        elif reg < 0x0100:
            reg_name = f'Unnamed Blk1 ({reg:#06x})'
        elif reg < 0x0110:
            reg_name = f'PeriStat{reg - 0x0100}'
        elif reg < 0x0120:
            reg_name = f'Unnamed Blk2 ({reg:#06x})'
        elif reg < 0x0124:
            reg_name = f'SEU Counter {reg - 0x0120}'
        elif reg < 0x8000:
            reg_name = f'Unnamed Blk3 ({reg:#06x})'
        else:
            space = "Cfg"
            if (reg & 0x4000) != 0:
                space = "Stat"
            if (reg & 0x2000) != 0:
                broadcast = " broadcast"
            register = reg & 0x1f
            row = (reg >> 5) & 0xf
            col = (reg >> 9) & 0xf
            reg_name = f'Pix{space}{register}'

            if not save_broadcast and broadcast != "":
                continue
            if not save_extra_pixel_stat and space == "Stat" and register > 7:
                continue

        if not save_seu_counter and 'SEU Counter' in reg_name:
            continue

        reg_diff = changed_registers["ETROC2"][reg][0] ^ changed_registers["ETROC2"][reg][1]

        if reg_name not in bitmap:
            if reg_name not in unused_bitmap:
                unused_bitmap[reg_name] = [0 for _ in range(8)]

            for bit_index in range(8):
                bit_val = (reg_diff >> bit_index) & 0b1
                if bit_val != 0:
                    unused_bitmap[reg_name][bit_index] += 1
        else:
            for bit_index in range(8):
                bit_val = (reg_diff >> bit_index) & 0b1
                if bit_val != 0:
                    bitmap[reg_name][bit_index] += 1

def write_register_latex_bitmap(register: str, bitmap: list[int], latexfile, header: str | None = None, max_color: int = 10):
    latexfile.write(r"\hline" + '\n')

    if header is not None:
        latexfile.write(r"\multirow{2}{*}{\small " + register + "} & " + header + r" \\" + '\n')
        latexfile.write(r"\hhline{|~|--------|}" + '\n')
    else:
        latexfile.write("\\small " + register)

    for bit_idx in range(8):
        latexfile.write(f' & \\small \\cellcolor{{red!{ceil(min(bitmap[7-bit_idx]/max_color, 1)*100)}}}{bitmap[7-bit_idx]}')
    latexfile.write(r"\\" + '\n')

    latexfile.write(r"\hline" + '\n')

def write_pixstat_latex_bitmap(bitmap: dict[str, list[int]], latexfile, max_color: int = 10):
    latexfile.write(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}" + '\n')
    latexfile.write(r"\tiny" + '\n')
    latexfile.write(r"\begin{tabularx}{700pt}{|r|*{8}{Y|} }" + '\n')
    latexfile.write(r"\hline" + '\n')
    latexfile.write(r" & \textbf{\small Bit 7} & \textbf{\small Bit 6} & \textbf{\small Bit 5} & \textbf{\small Bit 4} & \textbf{\small Bit 3} & \textbf{\small Bit 2} & \textbf{\small Bit 1} & \textbf{\small Bit 0} \\" + '\n')
    latexfile.write(r"\hline" + '\n')

    write_register_latex_bitmap("PixStat0", bitmap["PixStat0"], latexfile, header = r"\multicolumn{4}{c|}{PixelID - Row} & \multicolumn{4}{c|}{PixelID - Column}", max_color = max_color)
    write_register_latex_bitmap("PixStat1", bitmap["PixStat1"], latexfile, header = r"\multicolumn{3}{c|}{THState} & \multicolumn{4}{c|}{NW} & ScanDone", max_color = max_color)
    write_register_latex_bitmap("PixStat2", bitmap["PixStat2"], latexfile, header = r"\multicolumn{8}{c|}{BL[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PixStat3", bitmap["PixStat3"], latexfile, header = r"\multicolumn{2}{c|}{TH[1:0]} & -- & -- & -- & -- & \multicolumn{2}{c|}{BL[9:8]}", max_color = max_color)
    write_register_latex_bitmap("PixStat4", bitmap["PixStat4"], latexfile, header = r"\multicolumn{8}{c|}{TH[9:2]}", max_color = max_color)
    write_register_latex_bitmap("PixStat5", bitmap["PixStat5"], latexfile, header = r"\multicolumn{8}{c|}{ACC[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PixStat6", bitmap["PixStat6"], latexfile, header = r"\multicolumn{8}{c|}{ACC[15:8]}", max_color = max_color)
    write_register_latex_bitmap("PixStat7", bitmap["PixStat7"], latexfile, header = r"\multicolumn{8}{c|}{PixCfg31}", max_color = max_color)

    latexfile.write(r"\end{tabularx}" + '\n')

def write_pixcfg_latex_bitmap(bitmap: dict[str, list[int]], latexfile, max_color: int = 10):
    latexfile.write(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}" + '\n')
    latexfile.write(r"\tiny" + '\n')
    latexfile.write(r"\begin{tabularx}{700pt}{|r|*{8}{Y|} }" + '\n')
    latexfile.write(r"\hline" + '\n')
    latexfile.write(r" & \textbf{\small Bit 7} & \textbf{\small Bit 6} & \textbf{\small Bit 5} & \textbf{\small Bit 4} & \textbf{\small Bit 3} & \textbf{\small Bit 2} & \textbf{\small Bit 1} & \textbf{\small Bit 0} \\" + '\n')
    latexfile.write(r"\hline" + '\n')

    write_register_latex_bitmap( "PixCfg0",  bitmap["PixCfg0"], latexfile, header = r"-- & \multicolumn{2}{c|}{RfSel} & \multicolumn{3}{c|}{IBSel} & \multicolumn{2}{c|}{CLSel}", max_color = max_color)
    write_register_latex_bitmap( "PixCfg1",  bitmap["PixCfg1"], latexfile, header = r"-- & -- & QInjEn & \multicolumn{5}{c|}{QSel}", max_color = max_color)
    write_register_latex_bitmap( "PixCfg2",  bitmap["PixCfg2"], latexfile, header = r"-- & -- & -- & DACDiscri & \multicolumn{4}{c|}{HysSel}", max_color = max_color)
    write_register_latex_bitmap( "PixCfg3",  bitmap["PixCfg3"], latexfile, header = r"-- & -- & -- & ScanStart & CLKEn & Bypass & BufEn & RSTn", max_color = max_color)
    write_register_latex_bitmap( "PixCfg4",  bitmap["PixCfg4"], latexfile, header = r"\multicolumn{8}{c|}{DAC[7:0]}", max_color = max_color)
    write_register_latex_bitmap( "PixCfg5",  bitmap["PixCfg5"], latexfile, header = r"\multicolumn{6}{c|}{TH\_offset} & \multicolumn{2}{c|}{DAC[9:8]}", max_color = max_color)
    write_register_latex_bitmap( "PixCfg6",  bitmap["PixCfg6"], latexfile, header = r"enable\_TDC & resetn\_TDC & autoReset\_TDC & testMode\_TDC & \multicolumn{3}{c|}{level\_TDC} & --", max_color = max_color)
    write_register_latex_bitmap( "PixCfg7",  bitmap["PixCfg7"], latexfile, header = r"-- & -- & -- & \multicolumn{2}{c|}{workMode} & disTrigPath & disDataReadout & addrOffset", max_color = max_color)
    write_register_latex_bitmap( "PixCfg8",  bitmap["PixCfg8"], latexfile, header = r"L1ADelay[0] & \multicolumn{7}{c|}{selfTestOccupancy}", max_color = max_color)
    write_register_latex_bitmap( "PixCfg9",  bitmap["PixCfg9"], latexfile, header = r"\multicolumn{8}{c|}{L1ADelay[8:1]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg10", bitmap["PixCfg10"], latexfile, header = r"\multicolumn{8}{c|}{lowerCal[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg11", bitmap["PixCfg11"], latexfile, header = r"\multicolumn{6}{c|}{upperCal[5:0]} & \multicolumn{2}{c|}{lowerCal[9:8]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg12", bitmap["PixCfg12"], latexfile, header = r"\multicolumn{4}{c|}{lowerTOA[3:0]} & \multicolumn{4}{c|}{upperCal[9:6]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg13", bitmap["PixCfg13"], latexfile, header = r"\multicolumn{2}{c|}{upperTOA[1:0]} & \multicolumn{6}{c|}{lowerTOA[9:4]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg14", bitmap["PixCfg14"], latexfile, header = r"\multicolumn{8}{c|}{upperTOA[9:2]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg15", bitmap["PixCfg15"], latexfile, header = r"\multicolumn{8}{c|}{lowerTOT[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg16", bitmap["PixCfg16"], latexfile, header = r"\multicolumn{7}{c|}{upperTOT[6:0]} & \multicolumn{1}{c|}{lowerTOT[8]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg17", bitmap["PixCfg17"], latexfile, header = r"\multicolumn{6}{c|}{lowerCalTrig[5:0]} & \multicolumn{2}{c|}{upperTOT[8:7]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg18", bitmap["PixCfg18"], latexfile, header = r"\multicolumn{4}{c|}{upperCalTrig[3:0]} & \multicolumn{4}{c|}{lowerCalTrig[9:6]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg19", bitmap["PixCfg19"], latexfile, header = r"\multicolumn{2}{c|}{lowerTOATrig[1:0]} & \multicolumn{6}{c|}{upperCalTrig[9:4]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg20", bitmap["PixCfg20"], latexfile, header = r"\multicolumn{8}{c|}{lowerTOATrig[9:2]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg21", bitmap["PixCfg21"], latexfile, header = r"\multicolumn{8}{c|}{upperTOATrig[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg22", bitmap["PixCfg22"], latexfile, header = r"\multicolumn{6}{c|}{lowerTOTTrig[5:0]} & \multicolumn{2}{c|}{upperTOATrig[9:8]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg23", bitmap["PixCfg23"], latexfile, header = r"\multicolumn{5}{c|}{upperTOTTrig[4:0]} & \multicolumn{3}{c|}{lowerTOTTrig[8:6]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg24", bitmap["PixCfg24"], latexfile, header = r"-- & -- & -- & -- & \multicolumn{4}{c|}{upperTOTTrig[8:5]}", max_color = max_color)
    write_register_latex_bitmap("PixCfg25", bitmap["PixCfg25"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PixCfg26", bitmap["PixCfg26"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PixCfg27", bitmap["PixCfg27"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PixCfg28", bitmap["PixCfg28"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PixCfg29", bitmap["PixCfg29"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PixCfg30", bitmap["PixCfg30"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PixCfg31", bitmap["PixCfg31"], latexfile, max_color = max_color)

    latexfile.write(r"\end{tabularx}" + '\n')

def write_peristat_latex_bitmap(bitmap: dict[str, list[int]], latexfile, max_color: int = 10):
    latexfile.write(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}" + '\n')
    latexfile.write(r"\tiny" + '\n')
    latexfile.write(r"\begin{tabularx}{700pt}{|r|*{8}{Y|} }" + '\n')
    latexfile.write(r"\hline" + '\n')
    latexfile.write(r" & \textbf{\small Bit 7} & \textbf{\small Bit 6} & \textbf{\small Bit 5} & \textbf{\small Bit 4} & \textbf{\small Bit 3} & \textbf{\small Bit 2} & \textbf{\small Bit 1} & \textbf{\small Bit 0} \\" + '\n')
    latexfile.write(r"\hline" + '\n')

    write_register_latex_bitmap( "PeriStat0",  bitmap["PeriStat0"], latexfile, header = r"PS\_Late & \multicolumn{6}{c|}{AFCcalCap} & AFCBusy", max_color = max_color)
    write_register_latex_bitmap( "PeriStat1",  bitmap["PeriStat1"], latexfile, header = r"\multicolumn{4}{c|}{fcAlignFinalState} & \multicolumn{4}{c|}{controlerState}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat2",  bitmap["PeriStat2"], latexfile, header = r"\multicolumn{4}{c|}{fcAlignStatus} & -- & -- & -- & fcBitAlignError", max_color = max_color)
    write_register_latex_bitmap( "PeriStat3",  bitmap["PeriStat3"], latexfile, header = r"\multicolumn{8}{c|}{invalidFCCount[7:0]}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat4",  bitmap["PeriStat4"], latexfile, header = r"\multicolumn{4}{c|}{pllUnlockCount[3:0]} & \multicolumn{4}{c|}{invalidFCCount[11:8]}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat5",  bitmap["PeriStat5"], latexfile, header = r"\multicolumn{8}{c|}{pllUnlockCount[11:4]}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat6",  bitmap["PeriStat6"], latexfile, header = r"\multicolumn{8}{c|}{EFuseQ[7:0]}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat7",  bitmap["PeriStat7"], latexfile, header = r"\multicolumn{8}{c|}{EFuseQ[15:8]}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat8",  bitmap["PeriStat8"], latexfile, header = r"\multicolumn{8}{c|}{EFuseQ[23:16]}", max_color = max_color)
    write_register_latex_bitmap( "PeriStat9",  bitmap["PeriStat9"], latexfile, header = r"\multicolumn{8}{c|}{EFuseQ[31:24]}", max_color = max_color)
    write_register_latex_bitmap("PeriStat10", bitmap["PeriStat10"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PeriStat11", bitmap["PeriStat11"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PeriStat12", bitmap["PeriStat12"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PeriStat13", bitmap["PeriStat13"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PeriStat14", bitmap["PeriStat14"], latexfile, max_color = max_color)
    write_register_latex_bitmap("PeriStat15", bitmap["PeriStat15"], latexfile, max_color = max_color)

    latexfile.write(r"\end{tabularx}" + '\n')

def write_pericfg_latex_bitmap(bitmap: dict[str, list[int]], latexfile, max_color: int = 10):
    latexfile.write(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}" + '\n')
    latexfile.write(r"\tiny" + '\n')
    latexfile.write(r"\begin{tabularx}{700pt}{|r|*{8}{Y|} }" + '\n')
    latexfile.write(r"\hline" + '\n')
    latexfile.write(r" & \textbf{\small Bit 7} & \textbf{\small Bit 6} & \textbf{\small Bit 5} & \textbf{\small Bit 4} & \textbf{\small Bit 3} & \textbf{\small Bit 2} & \textbf{\small Bit 1} & \textbf{\small Bit 0} \\" + '\n')
    latexfile.write(r"\hline" + '\n')

    write_register_latex_bitmap( "PeriCfg0",  bitmap["PeriCfg0"], latexfile, header = r"FBDiv\_skip & clkTreeDisable & CLKSel & disVCO & disSER & disEOM & disDES & disCLK", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg1",  bitmap["PeriCfg1"], latexfile, header = r"\multicolumn{4}{c|}{PLL\_I} & \multicolumn{4}{c|}{PLL\_BiasGen}", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg2",  bitmap["PeriCfg2"], latexfile, header = r"\multicolumn{4}{c|}{PLL\_R} & \multicolumn{4}{c|}{PLL\_P}", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg3",  bitmap["PeriCfg3"], latexfile, header = r"VRefGen & -- & ENABLEPLL & PLL\_vcoRailMode & \multicolumn{4}{c|}{PLL\_vcoDAC}", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg4",  bitmap["PeriCfg4"], latexfile, header = r"TS\_PD & PS\_ForceDown & PS\_Enable & PS\_CapRst & \multicolumn{4}{c|}{PS\_CPCurrent}", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg5",  bitmap["PeriCfg5"], latexfile, header = r"\multicolumn{8}{c|}{PS\_PhaseAdj}", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg6",  bitmap["PeriCfg6"], latexfile, header = r"\multicolumn{8}{c|}{RefStrSel}", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg7",  bitmap["PeriCfg7"], latexfile, header = r"GRO\_TOARST\_N & GRO\_Start & CLK40\_SetCM & CLK40\_InvData & \multicolumn{2}{c|}{CLK40\_Equ} & CLK40\_EnTer & CLK40\_EnRx", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg8",  bitmap["PeriCfg8"], latexfile, header = r"GRO\_TOA\_Latch & GRO\_TOA\_CK & CLK1280\_SetCM & CLK1280\_InvData & \multicolumn{2}{c|}{CLK1280\_Equ} & CLK1280\_EnTer & CLK1280\_EnRx", max_color = max_color)
    write_register_latex_bitmap( "PeriCfg9",  bitmap["PeriCfg9"], latexfile, header = r"GRO\_TOT\_CK\_Latch & GRO\_TOTRST\_N & FC\_SetCM & FC\_InvData & \multicolumn{2}{c|}{FC\_Equ} & FC\_EnTer & FC\_EnRx", max_color = max_color)
    write_register_latex_bitmap("PeriCfg10", bitmap["PeriCfg10"], latexfile, header = r"\multicolumn{8}{c|}{BCIDoffset[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg11", bitmap["PeriCfg11"], latexfile, header = r"\multicolumn{4}{c|}{emptySlotBCID[3:0]} & \multicolumn{4}{c|}{BCIDoffset[11:8]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg12", bitmap["PeriCfg12"], latexfile, header = r"\multicolumn{8}{c|}{emptySlotBCID[11:4]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg13", bitmap["PeriCfg13"], latexfile, header = r"asyPLLReset & asyLinkReset & asyAlignFastcommand & \multicolumn{5}{c|}{readoutClockDelayPixel}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg14", bitmap["PeriCfg14"], latexfile, header = r"asyResetGlobalReadout & asyResetFastcommand & asyResetChargeInj & \multicolumn{5}{c|}{readoutClockWidthPixel}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg15", bitmap["PeriCfg15"], latexfile, header = r"-- & asyStartCalibration & asyResetLockDetect & \multicolumn{5}{c|}{readoutClockDelayGlobal}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg16", bitmap["PeriCfg16"], latexfile, header = r"\multicolumn{3}{c|}{LTx\_AmplSel} & \multicolumn{5}{c|}{readoutClockWidthGlobal}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg17", bitmap["PeriCfg17"], latexfile, header = r"\multicolumn{3}{c|}{RTx\_AmplSel} & \multicolumn{5}{c|}{chargeInjectionDelay}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg18", bitmap["PeriCfg18"], latexfile, header = r"disLTx & \multicolumn{2}{c|}{onChipL1AConf} & fcDataDelayEn & fcClkDelayEn & fcSelfAlignEn & softBoot & disPowerSequence", max_color = max_color)
    write_register_latex_bitmap("PeriCfg19", bitmap["PeriCfg19"], latexfile, header = r"disRTx & singlePort & \multicolumn{2}{c|}{serRateRight} & \multicolumn{2}{c|}{serRateLeft} & linkResetTestPattern & disScrambler", max_color = max_color)
    write_register_latex_bitmap("PeriCfg20", bitmap["PeriCfg20"], latexfile, header = r"\multicolumn{4}{c|}{eFuse\_TCKHP} & \multicolumn{3}{c|}{triggerGranularity} & mergeTriggerData", max_color = max_color)
    write_register_latex_bitmap("PeriCfg21", bitmap["PeriCfg21"], latexfile, header = r"-- & -- & eFuse\_Bypass & eFuse\_Start & eFuse\_Rstn & \multicolumn{2}{c|}{eFuse\_Mode} & eFuse\_EnClk", max_color = max_color)
    write_register_latex_bitmap("PeriCfg22", bitmap["PeriCfg22"], latexfile, header = r"\multicolumn{8}{c|}{eFuse\_Prog[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg23", bitmap["PeriCfg23"], latexfile, header = r"\multicolumn{8}{c|}{eFuse\_Prog[15:8]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg24", bitmap["PeriCfg24"], latexfile, header = r"\multicolumn{8}{c|}{eFuse\_Prog[23:16]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg25", bitmap["PeriCfg25"], latexfile, header = r"\multicolumn{8}{c|}{eFuse\_Prog[31:24]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg26", bitmap["PeriCfg26"], latexfile, header = r"\multicolumn{8}{c|}{linkResetFixedPattern[7:0]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg27", bitmap["PeriCfg27"], latexfile, header = r"\multicolumn{8}{c|}{linkResetFixedPattern[15:8]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg28", bitmap["PeriCfg28"], latexfile, header = r"\multicolumn{8}{c|}{linkResetFixedPattern[23:16]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg29", bitmap["PeriCfg29"], latexfile, header = r"\multicolumn{8}{c|}{linkResetFixedPattern[31:24]}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg30", bitmap["PeriCfg30"], latexfile, header = r"\multicolumn{4}{c|}{lfReLockThrCounter} & \multicolumn{4}{c|}{lfLockThrCounter}", max_color = max_color)
    write_register_latex_bitmap("PeriCfg31", bitmap["PeriCfg31"], latexfile, header = r"-- & -- & TDCStrobeTest & TDCClockTest & \multicolumn{4}{c|}{lfUnLockThrCounter}", max_color = max_color)

    latexfile.write(r"\end{tabularx}" + '\n')

def render_latex_table(work_path: Path, filename: str):
    with open(work_path/"document.tex", "w") as latexfile:
        latexfile.write(r"\documentclass[8pt,border=2pt]{standalone}" + '\n')
        latexfile.write(r"\usepackage{multirow}" + '\n')
        latexfile.write(r"\usepackage{tabularx}" + '\n')
        latexfile.write(r"\usepackage{booktabs}" + '\n')
        latexfile.write(r"\usepackage[table]{xcolor}" + '\n')
        latexfile.write(r"\usepackage{hhline}" + '\n')
        latexfile.write(r"\usepackage{diagbox}" + '\n')
        latexfile.write(r"\begin{document}" + '\n')
        latexfile.write(f"\\input{{{filename}}}" + '\n')
        latexfile.write(r"\end{document}" + '\n')

    #os.system("lualatex document.tex")
    #os.popen("lualatex document.tex").read()
    subprocess.Popen(f"cd {work_path}; lualatex document.tex", shell=True, stdout=subprocess.PIPE).stdout.read()

def save_bitmap_table(bitmap: dict[str, list[int]], save_path: Path, base_name: str, max_color: int = 10):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)

        pixstat_file = tmp_path/f"PixStat.tex"
        pixconf_file = tmp_path/f"PixConf.tex"
        peristat_file = tmp_path/f"PeriStat.tex"
        periconf_file = tmp_path/f"PeriConf.tex"
        with open(pixstat_file, "w") as latexfile:
            write_pixstat_latex_bitmap(bitmap, latexfile, max_color)
        with open(pixconf_file, "w") as latexfile:
            write_pixcfg_latex_bitmap(bitmap, latexfile, max_color)
        with open(peristat_file, "w") as latexfile:
            write_peristat_latex_bitmap(bitmap, latexfile, max_color)
        with open(periconf_file, "w") as latexfile:
            write_pericfg_latex_bitmap(bitmap, latexfile, max_color)

        base_name = base_name.replace(" ", "_")

        render_latex_table(tmp_path, "PixStat")
        shutil.move(tmp_path/"document.pdf", save_path/f"{base_name}_PixStat.pdf")
        #shutil.move(tmp_path/"document.tex", save_path/f"{base_name}_PixStat_gen.tex")
        render_latex_table(tmp_path, "PixConf")
        shutil.move(tmp_path/"document.pdf", save_path/f"{base_name}_PixConf.pdf")
        #shutil.move(tmp_path/"document.tex", save_path/f"{base_name}_PixConf_gen.tex")
        render_latex_table(tmp_path, "PeriStat")
        shutil.move(tmp_path/"document.pdf", save_path/f"{base_name}_PeriStat.pdf")
        #shutil.move(tmp_path/"document.tex", save_path/f"{base_name}_PeriStat_gen.tex")
        render_latex_table(tmp_path, "PeriConf")
        shutil.move(tmp_path/"document.pdf", save_path/f"{base_name}_PeriConf.pdf")
        #shutil.move(tmp_path/"document.tex", save_path/f"{base_name}_PeriConf_gen.tex")

        shutil.move(pixstat_file, save_path/f"{base_name}_PixStat.tex")
        shutil.move(pixconf_file, save_path/f"{base_name}_PixConf.tex")
        shutil.move(peristat_file, save_path/f"{base_name}_PeriStat.tex")
        shutil.move(periconf_file, save_path/f"{base_name}_PeriConf.tex")


### Work in Progress (Also missing equivalent functions for single runs)
def makeSlideHeaderAndTitle(latexfile: io.TextIOWrapper, slide_title: str, slide_short_title: str, slide_subtitle: str, do_toc: bool):
    latexfile.write(r"\documentclass[utf8,aspectratio=1610]{beamer}" + '\n')
    latexfile.write(r"\usetheme{Boadilla}" + '\n')
    latexfile.write(r"\graphicspath{ {figures/} }" + '\n')
    latexfile.write(r"\title[" + slide_short_title + r"]{" + slide_title + r"}" + '\n')
    if slide_subtitle != "":
        latexfile.write(r"\subtitle{" + slide_subtitle + r"}" + '\n')
    latexfile.write(r"\author[Cristóvão~BCS]{Cristóvão~B.~da~Cruz~e~Silva\inst{1}}" + '\n')
    latexfile.write(r"\institute[LIP]{\inst{1}Laboratório de Instrumentação e Física Experimental de Partículas}" + '\n')
    latexfile.write(r"\date[" + datetime.datetime.today().strftime("%Y/%m/%d") + r"]{" + datetime.datetime.today().strftime("%B %d, %Y") +  r"}" + '\n')

    latexfile.write(r"\newcommand*{\includeSinglePdf}[2][1]{\vspace*{-1pt}\makebox[#1\linewidth]{\includegraphics[width=\paperwidth]{#2}}}" + '\n')
    latexfile.write(r"\newcommand*{\includePdfPage}[3][1]{\vspace*{-1pt}\makebox[#1\linewidth]{\includegraphics[page=#3,width=\paperwidth]{#2}}}" + '\n')
    latexfile.write(r"\newcommand*{\includeSinglePdfHeight}[3]{\includegraphics[width=#2,height=#3,keepaspectratio]{#1}}" + '\n')

    latexfile.write(r"\begin{document}" + '\n')

    latexfile.write(r"\frame{\titlepage}" + '\n')

    if do_toc:
        latexfile.write(r"\begin{frame}" + '\n')
        latexfile.write(r"\frametitle{Table of Contents}" + '\n')
        latexfile.write(r"\tableofcontents" + '\n')
        latexfile.write(r"\end{frame}" + '\n')

def makeOverallSummarySlides(
        work_path: Path,
        time_start: datetime.datetime,
        time_end: datetime.datetime,
        chip_names: list[str],
        run_info,
        extra_run_info,
        power_directory: Path,
        baseline_directory:Path,
        scan_list,
        config_compare_dir: Path,
        tdc_plot_dir: Path,
        ps_see_times: list[datetime.datetime] = [],
        slide_title: str = "Overall Summary of SEU Results",
        slide_short_title: str = "Summary SEU Results",
        slide_subtitle: str = "",
        do_toc: bool = True,
                             ):
    with open(work_path/"slides.tex", "w") as latexfile:
        fig_dir = work_path/"figures"
        fig_dir.mkdir(exist_ok=True)

        makeSlideHeaderAndTitle(latexfile, slide_title, slide_short_title, slide_subtitle, do_toc)

        ## Summary Section
        latexfile.write(r"\section{Summary}" + '\n')

        latexfile.write(r"\begin{frame}" + '\n')
        latexfile.write(r"\frametitle{Summary}" + '\n')
        latexfile.write(r"\begin{itemize}" + '\n')
        latexfile.write(r"\item The campaign started at " + time_start.isoformat(sep = ' ') + r" and ended at " + time_end.isoformat(sep = ' ') + '\n')
        latexfile.write(r"\item A total of " + str(len(run_info)) + r" runs were taken" + '\n')
        latexfile.write(r"\item " + str(len(extra_run_info)) + r" additional runs were taken" + '\n')
        latexfile.write(r"\item The following chips were used:" + '\n')
        latexfile.write(r"\begin{itemize}" + '\n')
        for chip_name in chip_names:
            latexfile.write(r"\item " + chip_name.replace("_", r"\_") + '\n')
        latexfile.write(r"\end{itemize}" + '\n')
        if len(ps_see_times) != 0:
            for see_time in ps_see_times:
                if see_time is None:
                    continue
                latexfile.write(r"\item There was a single event upset on the power supply at " + see_time.isoformat(sep=" ") + '\n')
        latexfile.write(r"\end{itemize}" + '\n')
        latexfile.write(r"\end{frame}" + '\n')

        if power_directory.exists() and power_directory.is_dir():
            ## Power Monitoring Section
            latexfile.write(r"\section{Power Monitoring}" + '\n')

            latexfile.write(r"\begin{frame}" + '\n')
            latexfile.write(r"\frametitle{Power Monitoring}" + '\n')
            latexfile.write(r"\begin{itemize}" + '\n')
            latexfile.write(r"\item Voltages and Currents provided by the power supplies were monitored over time" + '\n')
            latexfile.write(r"\item There were individual supplies for:" + '\n')
            latexfile.write(r"\begin{itemize}" + '\n')
            latexfile.write(r"\item VRef - 1.0V reference shared between the chips" + '\n')
            latexfile.write(r"\item WS - 1.2V supply for the waveform sampler, shared between the chips. Required for correct chip operation" + '\n')
            for chip_name in chip_names:
                latexfile.write(r"\item " + chip_name.replace("_", r"\_") + r" Analog - 1.2V analog power supply" + '\n')
                latexfile.write(r"\item " + chip_name.replace("_", r"\_") + r" Digital - 1.2V digital power supply" + '\n')
            latexfile.write(r"\end{itemize}" + '\n')
            #if len(ps_see_times) != 0:
            #    latexfile.write(r"\item One of the power supplies failed during the campaign, see next slide. It supplied VRef and ET2\_Bare\_1" + '\n')
            latexfile.write(r"\end{itemize}" + '\n')
            latexfile.write(r"\end{frame}" + '\n')

            if (power_directory/"VRef.pdf").exists():
                shutil.copy(power_directory/"VRef.pdf", fig_dir/"VRef.pdf")
                latexfile.write(r"\subsection{VRef Monitoring}" + '\n')
                latexfile.write(r"\begin{frame}" + '\n')
                latexfile.write(r"\frametitle{VRef Monitoring}" + '\n')

                latexfile.write(r"\begin{center}" + '\n')
                latexfile.write(r"\includegraphics[width=\linewidth,height=0.5\linewidth,keepaspectratio]{VRef}" + '\n')
                latexfile.write(r"\end{center}" + '\n')

                if len(ps_see_times) != 0:
                    latexfile.write(r"\begin{itemize}" + '\n')
                    latexfile.write(r"\item The power failure is easily observed in this plot, where after " + ps_see_times[0].isoformat(sep=" ") + r" the VRef is no longer being correctly supplied" + '\n')
                    latexfile.write(r"\end{itemize}" + '\n')
                latexfile.write(r"\end{frame}" + '\n')

            if (power_directory/"WaveformSampler.pdf").exists():
                shutil.copy(power_directory/"WaveformSampler.pdf", fig_dir/"WaveformSampler.pdf")
                latexfile.write(r"\subsection{WS Monitoring}" + '\n')
                latexfile.write(r"\begin{frame}" + '\n')
                latexfile.write(r"\frametitle{WS Power Monitoring}" + '\n')

                latexfile.write(r"\begin{center}" + '\n')
                latexfile.write(r"\includegraphics[width=\linewidth,height=0.55\linewidth,keepaspectratio]{WaveformSampler}" + '\n')
                latexfile.write(r"\end{center}" + '\n')

                latexfile.write(r"\end{frame}" + '\n')

            for chip_name in chip_names:
                board_safe = chip_name.replace("_", "-")
                if (power_directory/f"{chip_name}.pdf").exists():
                    shutil.copy(power_directory/f"{chip_name}.pdf", fig_dir/f"{board_safe}-power.pdf")

                    latexfile.write(r"\subsection{" + chip_name.replace("_", " ") + r" Monitoring}" + '\n')
                    latexfile.write(r"\begin{frame}" + '\n')
                    latexfile.write(r"\frametitle{" + chip_name.replace("_", " ") + r" Power Monitoring}" + '\n')

                    latexfile.write(r"\begin{center}" + '\n')
                    latexfile.write(r"\includegraphics[width=\linewidth,height=0.55\linewidth,keepaspectratio]{" + board_safe + r"-power}" + '\n')
                    latexfile.write(r"\end{center}" + '\n')

                    latexfile.write(r"\end{frame}" + '\n')

        if baseline_directory.exists() and baseline_directory.is_dir():
            ## Baseline Monitoring Section
            latexfile.write(r"\section{Baseline Monitoring}" + '\n')

            latexfile.write(r"\begin{frame}" + '\n')
            latexfile.write(r"\frametitle{Baseline Monitoring}" + '\n')
            latexfile.write(r"\begin{itemize}" + '\n')
            latexfile.write(r"\item Baselines and noise widths of the individual pixels were taken over time" + '\n')
            latexfile.write(r"\item Only " + str(len(scan_list)) + r" pixels were enabled for data taking, so only those had baselines taken over time" + '\n')
            latexfile.write(r"\end{itemize}" + '\n')
            latexfile.write(r"\end{frame}" + '\n')

            for chip_name in chip_names:
                if (baseline_directory/f"{chip_name}_Baseline.pdf").exists():
                    shutil.copy(baseline_directory/f"{chip_name}_Baseline.pdf", fig_dir/"{}-Baseline.pdf".format(chip_name.replace("_", "-")))

                    latexfile.write(r"\subsection{" + chip_name.replace("_", " ") + r"}" + '\n')

                    latexfile.write(r"\begin{frame}" + '\n')
                    latexfile.write(r"\frametitle{" + chip_name.replace("_", " ") + r" Baseline Monitoring}" + '\n')
                    latexfile.write(r"\begin{center}" + '\n')
                    latexfile.write(r"\includegraphics[width=\linewidth,height=0.55\linewidth,keepaspectratio]{" + chip_name.replace("_", "-") + r"-Baseline}" + '\n')
                    latexfile.write(r"\end{center}" + '\n')
                    latexfile.write(r"\end{frame}" + '\n')

        if config_compare_dir.exists() and config_compare_dir.is_dir():
            ## I2C Registers Section
            latexfile.write(r"\section{I2C Registers}" + '\n')

            latexfile.write(r"\begin{frame}" + '\n')
            latexfile.write(r"\frametitle{I2C Registers}" + '\n')
            latexfile.write(r"\end{frame}" + '\n')

        if tdc_plot_dir.exists() and tdc_plot_dir.is_dir():
            ## Data Analysis Section
            latexfile.write(r"\section{Data Analysis}" + '\n')

            latexfile.write(r"\begin{frame}" + '\n')
            latexfile.write(r"\frametitle{Data Analysis}" + '\n')
            latexfile.write(r"\end{frame}" + '\n')

        latexfile.write(r"\end{document}" + '\n')

    subprocess.Popen(f"cd {work_path}; lualatex slides.tex; lualatex slides.tex", shell=True, stdout=subprocess.PIPE).stdout.read()
