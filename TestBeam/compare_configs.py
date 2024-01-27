from beamtest_analysis_helper import compare_chip_configs
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
                    prog='compare_configs.py',
                    description='This script compares two config files',
                    #epilog='Text at the bottom of help'
                    )

    parser.add_argument(
        '-a',
        '--config1',
        metavar = 'PATH',
        type = Path,
        help = 'First config file',
        dest = 'config1',
        required = True,
    )
    parser.add_argument(
        '-b',
        '--config2',
        metavar = 'PATH',
        type = Path,
        help = 'Second config file',
        dest = 'config2',
        required = True,
    )

    args = parser.parse_args()

    config1: Path = args.config1
    if not config1.exists():
        raise RuntimeError("The config1 path does not exist")
    config1 = config1.absolute()

    config2: Path = args.config2
    if not config2.exists():
        raise RuntimeError("The config2 path does not exist")
    config2 = config2.absolute()

    compare_chip_configs(config1, config2)

if __name__ == "__main__":
    main()