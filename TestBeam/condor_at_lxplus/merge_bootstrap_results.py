from collections import defaultdict
from natsort import natsorted
import argparse
from glob import glob
import pandas as pd
import re
from tqdm import tqdm
from lmfit.models import GaussianModel
import hist
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
            prog='PlaceHolder',
            description='merge individual bootstrap results',
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
    '--outputname',
    metavar = 'OUTNAME',
    type = str,
    help = 'output file name',
    required = True,
    dest = 'outname',
)

args = parser.parse_args()

final_dict = defaultdict(list)
files = natsorted(glob(args.dirname+'/*pkl'))

mod = GaussianModel(nan_policy='omit')


for ifile in tqdm(files):

    # Define the pattern to match "RxCx" part
    pattern = r'R(\d+)C(\d+)'

    # Find all occurrences of the pattern in the string
    matches = re.findall(pattern, ifile)

    df = pd.read_pickle(ifile)
    columns = df.columns

    for idx in range(len(columns)):

        h_temp = hist.Hist(hist.axis.Regular(100, 0, 100, name="time_resolution", label=r'Time Resolution [ps]'))
        h_temp.fill(df[columns[idx]])
        centers = h_temp.axes[0].centers

        fit_constrain = (centers > df[columns[idx]].astype(int).mode()[0]-7) & (centers < df[columns[idx]].astype(int).mode()[0]+7)

        pars = mod.guess(h_temp.values()[fit_constrain], x=centers[fit_constrain])
        out = mod.fit(h_temp.values()[fit_constrain], pars, x=centers[fit_constrain], weights=1/np.sqrt(h_temp.values()[fit_constrain]))

        final_dict[f'row{columns[idx]}'].append(matches[idx][0])
        final_dict[f'col{columns[idx]}'].append(matches[idx][1])
        if abs(out.params['sigma'].value) < 10:
            final_dict[f'res{columns[idx]}'].append(out.params['center'].value)
            final_dict[f'err{columns[idx]}'].append(abs(out.params['sigma'].value))
        else:
            final_dict[f'res{columns[idx]}'].append(np.mean(df[columns[idx]]))
            final_dict[f'err{columns[idx]}'].append(np.std(df[columns[idx]]))


final_df = pd.DataFrame(final_dict)
final_df.to_csv(args.outname+'.csv', index=False)
