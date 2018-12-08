==================
Command line tools
==================

:code:`janggu`
---------------------

The janggu app can be used to browse
through the results of one or more models using 
webbrowser of your choice.

Example usage::

   janggu -path <results-root> -port <PORT>


:code:`janggu-trim`
-------------------

janggu-trim can be used to trim the interval starts
and ends of a given BED/GFF file which is intended
for creating the ROI by a specified factor.

Trimming might circumvent undesired round effects when
using :code:`resolution>1` and :code:`store_whole_genome=True`
to handle covarage data with :code:`Cover`.
Therefore, we suggest to trim the ROIs that are used during
training and evaluation beforehand. For the sake of convenience,
we added the tool janggu-trim to do that.

Example usage::

   janggu-trim input.bed trimmed.bed -divby 50
