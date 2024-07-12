import sys, os

path2add = os.path.normpath(os.path.abspath(os.path.join(os.path.curdir, os.path.pardir, 'TestBeam')))
print(path2add)

if (not (path2add in sys.path)) :
    sys.path.append(path2add)

from pathlib import Path
from natsort import natsorted
from beamtest_analysis_helper import toSingleDataFrame_newEventModel_moneyplot
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use('CMS')

def cal_code_filtering(input_df: pd.DataFrame):
    cal_table = input_df.pivot_table(index=["row", "col", "charge", "threshold"], columns=["board"], values=["cal"], aggfunc=lambda x: x.mode().iat[0])
    cal_table = cal_table.reset_index().set_index([('row', ''), ('col', ''), ('charge', ''), ('threshold', '')]).stack().reset_index()
    cal_table.columns = ['row', 'col', 'charge', 'threshold', 'board', 'cal_mode']
    merged_df = pd.merge(input_df, cal_table, on=['board', 'row', 'col', 'charge','threshold'])
    cal_condition = abs(merged_df['cal'] - merged_df['cal_mode']) <= 3
    merged_df = merged_df[cal_condition].drop(columns=['cal_mode'])
    cal_filtered_df = merged_df.reset_index(drop=True)
    cal_filtered_df['board'] = cal_filtered_df['board'].astype(np.uint8)
    del merged_df, cal_condition, cal_table

    return cal_filtered_df

def tot_code_filtering(input_df: pd.DataFrame):
    tot_table = input_df.pivot_table(index=["row", "col", "charge", "threshold"], columns=["board"], values=["tot"], aggfunc=lambda x: x.mode().iat[0])
    tot_table = tot_table.reset_index().set_index([('row', ''), ('col', ''), ('charge', ''), ('threshold', '')]).stack().reset_index()
    tot_table.columns = ['row', 'col', 'charge', 'threshold', 'board', 'tot_mode']
    merged_df = pd.merge(input_df, tot_table, on=['board', 'row', 'col', 'charge','threshold'])
    tot_condition = abs(merged_df['tot'] - merged_df['tot_mode']) <= 100
    merged_df = merged_df[tot_condition].drop(columns=['tot_mode'])
    tot_filtered_df = merged_df.reset_index(drop=True)
    tot_filtered_df['board'] = tot_filtered_df['board'].astype(np.uint8)
    del merged_df, tot_condition, tot_table

    return tot_filtered_df

def make_cal_plots(input_df: pd.DataFrame, board_name: str):
    row_col_combinations = input_df.index.to_frame(index=False)[['row', 'col']].drop_duplicates(subset=['row', 'col'])

    for (row, col) in row_col_combinations.values:
        subset = agg_df.xs((row, col), level=('row', 'col'))
        charges = subset.index.get_level_values('charge').unique()

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        for charge in charges:
            charge_data = subset.xs(charge, level='charge')
            axes[0].plot(charge_data.index.get_level_values('threshold'), charge_data['cal_mean'], '.-', label=f'{charge} fC')
            axes[1].plot(charge_data.index.get_level_values('threshold'), charge_data['cal_std'], '.-', label=f'{charge} fC')

        hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC", fontsize=18)
        hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC", fontsize=18)
        axes[0].set_title(f'Row {row}, Col {col}', loc='right', fontsize=16)
        axes[1].set_title(f'Row {row}, Col {col}', loc='right', fontsize=16)

        axes[0].set_ylim(charge_data['cal_mean'].mean()-5, charge_data['cal_mean'].mean()+5)
        axes[1].set_ylim(-0.03, 1)

        axes[0].set_xlabel('Threshold')
        axes[1].set_xlabel('Threshold')
        axes[0].set_ylabel('CAL Mean')
        axes[1].set_ylabel('CAL Std')
        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(f'{board_name}/{board_name}_Row_{row}_Col_{col}_CAL.png')
        fig.savefig(f'{board_name}/{board_name}_Row_{row}_Col_{col}_CAL.pdf')

def make_toa_plots(input_df: pd.DataFrame, board_name: str):
    row_col_combinations = input_df.index.to_frame(index=False)[['row', 'col']].drop_duplicates(subset=['row', 'col'])

    for (row, col) in row_col_combinations.values:
        subset = agg_df.xs((row, col), level=('row', 'col'))
        charges = subset.index.get_level_values('charge').unique()

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        for charge in charges:
            charge_data = subset.xs(charge, level='charge')
            axes[0].plot(charge_data.index.get_level_values('threshold'), charge_data['toa_mean'], '.-', label=f'{charge} fC')
            axes[1].plot(charge_data.index.get_level_values('threshold'), charge_data['toa_std'], '.-', label=f'{charge} fC')

        hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC", fontsize=18)
        hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC", fontsize=18)
        axes[0].set_title(f'Row {row}, Col {col}', loc='right', fontsize=16)
        axes[1].set_title(f'Row {row}, Col {col}', loc='right', fontsize=16)

        axes[1].set_ylim(-0.03, 4)

        axes[0].set_xlabel('Threshold')
        axes[1].set_xlabel('Threshold')
        axes[0].set_ylabel('TOA Mean')
        axes[1].set_ylabel('TOA Std')
        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(f'{board_name}/{board_name}_Row_{row}_Col_{col}_TOA.png')
        fig.savefig(f'{board_name}/{board_name}_Row_{row}_Col_{col}_TOA.pdf')

def make_tot_plots(input_df: pd.DataFrame, board_name: str):
    row_col_combinations = input_df.index.to_frame(index=False)[['row', 'col']].drop_duplicates(subset=['row', 'col'])

    for (row, col) in row_col_combinations.values:
        subset = agg_df.xs((row, col), level=('row', 'col'))
        charges = subset.index.get_level_values('charge').unique()

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        for charge in charges:
            charge_data = subset.xs(charge, level='charge')
            axes[0].plot(charge_data.index.get_level_values('threshold'), charge_data['tot_mean'], '.-', label=f'{charge} fC')
            axes[1].plot(charge_data.index.get_level_values('threshold'), charge_data['tot_std'], '.-', label=f'{charge} fC')

        hep.cms.text(loc=0, ax=axes[0], text="ETL ETROC", fontsize=18)
        hep.cms.text(loc=0, ax=axes[1], text="ETL ETROC", fontsize=18)
        axes[0].set_title(f'Row {row}, Col {col}', loc='right', fontsize=16)
        axes[1].set_title(f'Row {row}, Col {col}', loc='right', fontsize=16)

        axes[1].set_ylim(-0.03, 4)

        axes[0].set_xlabel('Threshold')
        axes[1].set_xlabel('Threshold')
        axes[0].set_ylabel('TOT Mean')
        axes[1].set_ylabel('TOT Std')
        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(f'{board_name}/{board_name}_Row_{row}_Col_{col}_TOT.png')
        fig.savefig(f'{board_name}/{board_name}_Row_{row}_Col_{col}_TOT.pdf')


#############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='PlaceHolder',
                description='make money plots',
            )

    parser.add_argument(
        '-d',
        '--inputdir',
        metavar = 'DIRNAME',
        type = str,
        help = 'input directory name',
        required = True,
        dest = 'dirname',
    )

    parser.add_argument(
        '-o',
        '--outname',
        metavar = 'OUTNAME',
        type = str,
        help = 'output file name',
        required = True,
        dest = 'outname',
    )

    parser.add_argument(
        '--board_name',
        metavar = 'BOARD NAME',
        type = str,
        help = 'board name',
        required = True,
        dest = 'board_name',
    )

    args = parser.parse_args()

    dirs = natsorted(list(Path(f'{args.dirname}').glob(f'{args.board_name}_VRef_SCurve_TDC_Pixel_*_QInj_*_Threshold_*')))
    print(dirs[:3])
    print(dirs[-3:])

    current_dir = Path('./')
    outdir = current_dir / f'{args.board_name}'
    outdir.mkdir(exist_ok=True)

    if not Path(args.outname).exists():
        df = toSingleDataFrame_newEventModel_moneyplot(directories=dirs)
        df.to_feather(f'{args.outname}_qinj_moneyplot.feather')
    else:
        df = pd.read_feather(args.outname)

    ### Drop unnecessary columns
    df.drop(columns=['ea', 'bcid', 'l1a_counter'], inplace=True)

    cal_filtered_df = cal_code_filtering(df)
    del df

    tot_filtered_df = tot_code_filtering(cal_filtered_df)
    del cal_filtered_df

    ### Calculate mean and std
    grouped = tot_filtered_df.groupby(['row', 'col', 'charge', 'threshold'])
    agg_df = grouped.agg(
        cal_mean=('cal', 'mean'),
        cal_std=('cal', 'std'),
        toa_mean = ('toa','mean'),
        toa_std = ('toa', 'std'),
        tot_mean=('tot', 'mean'),
        tot_std=('tot', 'std'),
    )

    ### Drawing plots
    make_cal_plots(agg_df, args.board_name)
    make_toa_plots(agg_df, args.board_name)
    make_tot_plots(agg_df, args.board_name)