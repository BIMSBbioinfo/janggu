
Changelog
=========

0.9.7 (2020-05-07)
------------------

- Performance improvement for loading BED files.
- If no binsize is supplied, the binsize is detected automatically as the longest interval in the roi. Previously automatic detection was only offered if all intervals were of equal length, otherwise, a binsize needed to be specified manually.

0.9.6 (2020-02-26)
------------------

- Retired support for python 2.7
- RandomShift wrapper for data augmentation applied to DNA/Protein sequences was added by (thanks to @remomomo).
- Bedgraph files can be read using Cover.create_from_bed
- Improved efficiency of Cover.export_to_bigwig
- Improved efficiency of Cover.create_from_bed
- Internal refactoring

0.9.5 (2019-10-17)
------------------

- Variant effect prediction: added annotation argument which enables strand-specific variant effect predictions using the strandedness of features in the annotation file.
- Variant effect prediction: added ignore_reference_match argument which enables ignores mismatching nucleotides between the VCF reference base and the reference genome. By default, variant effects are only evaluated if the nucleotides agree in the reference genome and the VCF file.
- Added file validity check
- Added option to control verbosity 
- Improved efficiency for reading BAM and BIGWIG files
- Create a new cachefile with random_state only for not storing the whole genome
- Relaxed constraint for using resolution > 1 with ROI intervals. Still the interval starts have to be divisible by the resolution. Otherwise, weird rounding errors might occur.
- Fixed issue due to different numbers of network output layers.
- Added seperate dataversion to better control when cache files need to be reloaded from scratch.

0.9.4 (2019-07-15)
------------------

- Added SqueezeDim wrapper for compatibility with sklearn
- Added Transpose wrapper, replaces channel_last option of the datasets
- Loading paired-end bam-files with pairedend='5pend' option counts both ends now.
- resolution option added to create_from_array
- Relaxed restriction for sequence feature order
- Cover access via interval now returns nucleotide-resolution data regardless of the store_whole_genome option to ensure consistency.
- Refactoring


0.9.3 (2019-07-08)
------------------

- View mechanism added which allows to reuse the same dataset for different purposes, e.g. training set and test set.
- Added a dataset randomization which allows to internally randomize the data in order to avoid having to use shuffle=True with the fit method. This allows fetch randomized data in coherent chunks from hdf5 format files which improves access time.
- Added lazy loading mechanism for DNA and BED files, which defer the determination of the genome size to the dataset creation phase, but does not perform it when loading cached files to improve reload time.
- Caching logic improved in order to maximize the amount of reusability of dataset. For example, when the whole genome is loaded, the data can later be reloaded with different binsizes.
- Variant effect prediction functionality added.
- Improved efficiency for loading coverage from an array.
- Added axis option to ReduceDim
- Added Track classes to improve flexibility on plotGenomeTrack

0.9.2 (2019-05-04)
------------------

- Bugfix: Bioseq caching mechanism fixed.

0.9.1 (2019-05-03)
------------------

- Removed HTSeq dependence in favour of pybedtools for parsing BED, GFF, etc. This also introduces the requirement to have bedtools installed on the system, but it allows to parse BED-like files faster and more conveniently.
- Internal rearrangements for GenomicArray store_whole_genome=False. Now the data is stored as one array in a dict-like handle with the dummy key 'data' rather than storing the data in a fragmented fashion using as key-values the genomic interval and the respective coverages associated with them. This makes storage and processing more efficient.
- Bugfix: added conditions property to wrapper datasets.

0.9.0 (2019-03-20)
------------------

Added various features and bug fixes:

Changes in janggu.data

- Added new dataset wrapper to remove NaNs: NanToNumConverter
- Added new dataset wrappers for data augmentation: RandomOrientation, RandomSignalScale
- Adapted ReduceDim wrapper: added aggregator argument
- plotGenomeTrack added figsize option
- plotGenomeTrack added other plot types, including heatmap and seqplot.
- plotGenomeTrack refactoring of internal code
- Bioseq bugfix: Fixed issue for reverse complementing N's in the sequence.
- GenomicArray: condition, order, resolution are not read from the cache anymore, but from the arguments to avoid inconsistencies
- Normalization of Cover can handle a list of normalizer callables which are applied in turn
- Normaliation and Transformation: Added PercentileTrimming, RegionLengthNormalization, LogTransform
- ZScore and ZScoreLog do not apply RegionLengthNormalization by default anymore.
- janggu.data version-aware caching of datasets included
- Added copy method for janggu datasets.
- split_train_test refactored
- removed obsolete transformations attribute from the datasets
- Adapted the documentation
- Refactoring according to suggestions from isort and pylint

Changes in janggu

- Added input_attribution via integrated gradients for feature importance assignment
- Performance scoring by name for Janggu.evaluate for a number common metrices, including ROC, PRC, correlation, variance explained, etc.
- training.log is stored by default for each model
- Added model_from_json, model_from_yaml wrappers
- inputlayer decorator only instantiates Input layers if inputs == None, which makes the use of inputlayer less restrictive when using nested functions
- Added create_model method to create a keras model directly
- Adapted the documentation
- Refactoring according to suggestions from isort and pylint


0.8.6 (2019-03-03)
------------------

- Bugfix for ROIs that reach beyond the chromosome when loading Bioseq datasets. Now, zero-padding is performed for intervals that stretch over the sequence ends.

0.8.5 (2019-01-09)
------------------

- Updated abstract, added logo
- Utility: janggutrim command line tool for cutting bed file regions to avoid unwanted rounding effects. If rounding issues are detected an error is raised.
- Caching mechanism revisited. Caching of datasets is based on determining the sha256 hash of the dataset. If the data or some parameters change, the files are automatically reloaded. Consequently, the arguments overwrite and datatags become obsolete and have been marked for deprecation.
- Refactored access of GenomicArray
- Added ReduceDim wrapper to convert a 4D Cover object to a 2D table-like object.

0.8.4 (2018-12-11)
------------------

- Updated installation instructions in the readme

0.8.3 (2018-12-05)
------------------

- Fixed issues for loading SparseGenomicArray
- Made GenomicIndexer.filter_by_region aware of flank
- Fixed BedLoader of partially overlapping ROI and bedfiles issue using filter_by_region.
- Adapted classifier, license and keywords in setup.py
- Fixed hyperlinks

0.8.2 (2018-12-04)
------------------

- Bugfix for zero-padding functionality
- Added ndim for keras compatibility

0.8.1 (2018-12-03)
------------------

- Bugfix in GenomicIndexer.create_from_region

0.8.0 (2018-12-02)
------------------

- Improved test coverage
- Improved linter issues
- Bugs fixed
- Improved documentation for scorers
- Removed kwargs for scorers and exporters
- Adapted exporters to classes


0.7.0 (2018-12-01)
------------------

- First public version
