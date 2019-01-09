
Changelog
=========

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
