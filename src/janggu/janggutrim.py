"""janggu-trim bed-file trimming utility."""

import argparse

from janggu.utils import trim_bed


def main():
    """janggu-trim command line tool."""

    parser = argparse.ArgumentParser(
        description='janggu-trim - BED-file trimming tool.\n\n'
                    'The tool trims the interval starts and ends\n'
                    'to be divisible by a specified integer value.\n'
                    'This prevents unwanted rounding effects during training.\n'
                    'janggu (GPL-v3). Copyright (C) 2017-2018 '
                    'Wolfgang Kopp')
    parser.add_argument('inputbed', type=str,
                        help="Input BED-file")
    parser.add_argument('outputbed', type=str,
                        help="Output trimmed BED-file.")

    parser.add_argument('-divby', dest='divby', type=int,
                        default=1,
                        help="Divisibility of region starts and ends by the given factor.")

    args = parser.parse_args()
    trim_bed(args.inputbed, args.outputbed, args.divby)
