## Procedure
### Require python >= 3.9

#### Condor submission only available on LXPLUS

Let's load python 3.9 enviornment if you're on the server. (e.g. Lxplus)
```bash load_python39.sh```

Find track candidates based on pixel ID, and save them into csv format.
```python finding_good_track_candidates.py -p <path> -o <output file name> -i <iteration> -s <sampling> -m <minimum # of tracks> --refID N1 --dutID N2 --ignoreID N3```


```python submit_dataSelectionByTrack.py -d <input directory> -t <track_info.csv> --refID N1 --dutID N2 --ignoreID N3 --dryrun```
- `-d`: path to directory which includes .feather files
- `-t`: csv file including track with pixel ID (row and col)
- `--refID`: reference board ID in int
- `--dutID`: Device Under Test (DUT) board ID in int
- `--ignoreID`: The board ID which is not considering during the analysis
- `--dryrun`: Manual condor submission. Without this option, condor submission will happen automatcially

```python merge_dataSelectionByTrack_results.py -d <input directory> -o <output directory>```
- `-d`: path to directory with input files (.pickle)
- `-o`: directory name to save output files

```python submit_bootstrap.py -d <input directory> -i N1 -s N2 --dryrun```
- `-d`: path to directory with input files (.pkl)
- `-i`: number of iteration for bootstrap
- `-s`: fraction of random sampling in int. This number will be converted to float later.
- `--dryrun`: Manual condor submission. Without this option, condor submission will happen automatcially

```python merge_bootstrap_results.py -d <input directory> -o <output name>```
- `-d`: path to directory with input files (.pickle)
- `-o`: name for the output file