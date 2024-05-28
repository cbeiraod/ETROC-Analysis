## Procedure
### Condor submission only available on LXPLUS
### Require python >= 3.9

#### Clone ETROC-Analysis repository
ssh to Lxplus, then move to afs work directory

```cd /afs/cern.ch/work/<first alphabet of the username>/<username>```

```mkdir -p ETROC && cd ETROC```

```git clone https://github.com/CMS-ETROC/ETROC-Analysis.git```

Then go to the directory where control submitting jobs.

```cd ETROC-Analysis/TestBeam/condor_at_lxplus```

Let's load python 3.9 enviornment if you're on the server. (e.g. Lxplus)

```source load_python39.sh```

#### Find track candidates based on pixel ID, and save them into csv format.

```python finding_good_track_candidates.py -p <path> -o <output file name> -i <iteration> -s <sampling> -m <minimum # of tracks> --refID N1 --dutID N2 --ignoreID N3 --four_board```
- `-p`: path to directory which includes feather files.
- `-o`: set output csv file name.
- `-i`: set how many iteration will do to find track candidates
- `-s`: set the fraction of random file selection.
- `-m`: cut for the minimum number of tracks.
- `--refID`: set reference board ID.
- `--dutID`: set Device Under Test (DUT) board ID.
- `--ignoreID`: set the board ID that will be ignored. But with `--four_board` flag, this will be the second reference board.
- `--four_board`: If this flag is given, finding the track based on 4-board combination. Otherwise 3-board combination will be considered.

Example:
```python finding_good_track_candidates.py -p DESYFeb2024_TrigRefOffset15_DUTOffset10_feathers -o DESYFeb2024_TrigRefOffset15_DUTOffset10_good_track_candidates_4boards -i 10 -s 20 -m 1000 --refID 3 --dutID 2 --ignoreID 1 --four_board```

#### Select data based on each track with offline trigger selection

The user will submit jobs on condor. Before proceed, check how many jobs are on queue with `condor_q` command. If there are too many jobs, `myschedd bump` helps to find the work node that has less jobs.

```python submit_track_data_selection.py -d <input directory> -t <track_info.csv> --trigID N1 --refID N2 --dutID N3 --ignoreID N4 --trigTOTLower LB_NUM --trigTOTUpper UB_NUM --dryrun```
- `-d`: path to directory which includes feather files. You must include "*" at the end of the path. See the below example.
- `-t`: csv file including track with pixel ID (row and col).
- `--trigID`: trigger board ID in int.
- `--refID`: reference board ID in int.
- `--dutID`: Device Under Test (DUT) board ID in int.
- `--ignoreID`: The board ID which is not considering during the analysis.
- `--trigTOTLower` and `--trigTOTUpper`: TOT range for offline trigger selection. This cut only applies to the trigger board.
- `--dryrun`: Manual condor submission. Without this option, condor submission will happen automatcially.

Example:
```python submit_track_data_selection.py -d ${dname}_feathers/* -t ${dname}_good_track_candidates_4boards.csv --trigID 0 --refID 3 --dutID 1 --ignoreID 2 --trigTOTLower 100 --trigTOTUpper 200```

```python merge_dataSelectionByTrack_results.py -d <input directory> -o <output directory>```
- `-d`: path to directory with input files (.pickle).
- `-o`: directory name to save output files.

#### Run bootstrap

```python submit_bootstrap.py -d <input directory> -i N1 -s N2 --board_id_for_TOA_cut ID --minimum_nevt NUM --trigTOALower LB_NUM --trigTOAUpper UB_NUM --autoTOTcuts --noTrig --reproducible --dryrun```
- `-d`: path to directory with input files (.pkl).
- `-i`: number of iteration for bootstrap.
- `-s`: fraction of random sampling in int. This number will be converted to float later.
- `--board_id_for_TOA_cut`: board ID to notify which board data will be cut by TOA cut.
- `--minimum_nevt`: minimum number of events to process bootstrap method. If the number of events is less than this cut, bootstrap will not work.
- `--trigTOALower` and `--trigTOAUpper`: Set TOA range.
- `--autoTOTcuts`: If this flag is given, drop the TOT data very low/high (less than 1% and above 99%).
- `--noTrig`: The user must give this flag, when doing analysis without the board ID = 0
- `--reproducible`: Control random seed to select data. Also save the seed at the end, so this enables track down random set when the final results are bad.
- `--dryrun`: Manual condor submission. Without this option, condor submission will happen automatcially.

Example:
```python submit_bootstrap.py -d ${dname}_tracks --trigTOALower 500 --trigTOAUpper 700 --autoTOTcuts --minimum_nevt 700```

```python merge_bootstrap_results.py -d <input directory> -o <output name>```
- `-d`: path to directory with input files (.pkl).
- `-o`: name for the output file.
