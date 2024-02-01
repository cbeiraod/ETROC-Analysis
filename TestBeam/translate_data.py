from beamtest_analysis_helper import DecodeBinary
from pathlib import Path
import argparse
from natsort import natsorted

def main():
    parser = argparse.ArgumentParser(
                    prog='translate_data.py',
                    description='This script translate data from binary to NEM file format',
                    #epilog='Text at the bottom of help'
                    )

    parser.add_argument(
        '-p',
        '--path',
        metavar = 'PATH',
        type = Path,
        help = 'Path to the data directory containing the loop_* folders',
        dest = 'path',
        required = True,
    )
    parser.add_argument(
        '-o',
        '--outpath',
        metavar = 'PATH',
        type = Path,
        help = 'Path to the output directory to save the translated NEM files',
        dest = 'outpath',
    )
    parser.add_argument(
        '-f',
        '--feather-file',
        metavar = 'PATH',
        type = Path,
        help = 'Path to the output directory to save the translated feather files',
        dest = 'feather_file',
    )
    parser.add_argument(
        '-s',
        '--skip-filler',
        action='store_true',
        help = 'If set, the output NEM file will not contain the firmware filler words',
        dest = 'skip_filler',
    )

    args = parser.parse_args()

    path: Path = args.path
    if not path.exists():
        raise RuntimeError("The path does not exist")
    path = path.absolute()

    outpath: Path = args.outpath
    save_nem = None
    if outpath is not None:
        if not outpath.exists():
            raise RuntimeError("The out path does not exist")
        outpath = outpath.absolute()
        save_nem = outpath / 'translated.nem'

    feather_file: Path = args.feather_file
    if feather_file is not None:
        if not feather_file.exists():
            raise RuntimeError("The feather_file path does not exist")
        feather_file = feather_file.absolute()

    if outpath is None and feather_file is None:
        raise RuntimeError("You must enable at least one output format")



    files = natsorted(list(path.glob('loop_*/*')))
    decoder = DecodeBinary(
                            firmware_key=0b0001,
                            board_id=[0x17f0f, 0x17f0f, 0x17f0f, 0x17f0f],
                            file_list=files,
                            save_nem = save_nem,
                            skip_filler = args.skip_filler,
                           )

    df = decoder.decode_files()
    if feather_file is not None:
        df.to_feather(feather_file / 'translated.feather')

if __name__ == "__main__":
    main()