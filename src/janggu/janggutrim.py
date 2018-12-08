
import argparse
from janggu.utils import _get_genomic_reader
import numpy as np


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


def trim_bed(inputbed, outputbed, divby):
    """Trims starts and ends of intervals."""
    with open(outputbed, 'w') as bed:
        regions = _get_genomic_reader(inputbed)
        for region in regions:
            start = int(np.ceil(region.iv.start / divby)) * divby
            end = (region.iv.end // divby) * divby
            bed.write('{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n'
                .format(chrom=region.iv.chrom,
                        start=start,
                        end=end,
                        name=region.name, 
                        score=region.score if region.score is not None else 0,
                        strand=region.iv.strand))

