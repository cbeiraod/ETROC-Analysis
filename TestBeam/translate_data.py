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
        required = True,
    )

    args = parser.parse_args()

    path: Path = args.path
    if not path.exists():
        raise RuntimeError("The path does not exist")
    path = path.absolute()

    outpath: Path = args.outpath
    if not outpath.exists():
        raise RuntimeError("The out path does not exist")
    outpath = outpath.absolute()

    files = natsorted(list(path.glob('loop_*/*')))
    decoder = DecodeBinary(firmware_key=0b0001, board_id=[0x17f0f, 0x17f0f, 0x17f0f, 0x17f0f], file_list=files, save_nem = outpath / 'translated.nem')

    df = decoder.decode_files()

if __name__ == "__main__":
    main()