
import argparse
from janggu.utils import trim_bed

def main():

    PARSER = argparse.ArgumentParser(description='janggu-trim - BED-file trimming tool.\n\n'
                                                 'The tool trims the interval starts and ends\n'
                                                 'to be divisible by a specified integer value.\n'
                                                 'This prevents unwanted rounding effects during training.\n'
                                                 'janggu (GPL-v3). Copyright (C) 2017-2018 '
                                                 'Wolfgang Kopp')
    PARSER.add_argument('inputbed', type=str,
                        help="Input BED-file")
    PARSER.add_argument('outputbed', type=str,
                        help="Output trimmed BED-file.")
    
    PARSER.add_argument('-divby', dest='divby', type=int,
                        default=1,
                        help="Divisibility of region starts and ends by the given factor.")
    
    ARGS = PARSER.parse_args()
    trim_bed(ARGS.inputbed, ARGS.outputbed, ARGS.divby)


