"""Genomic track visualization utilities."""
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from janggu.utils import NMAP
from janggu.utils import PMAP
from janggu.utils import _to_list


def plotGenomeTrack(tracks, chrom, start, end, figsize=(10, 5), plottypes=None):

    """plotGenomeTrack shows plots of a specific interval from cover objects data.

    It takes one or more cover objects as well as a genomic interval consisting
    of chromosome name, start and end and creates
    a genome browser-like plot.

    Parameters
    ----------
    tracks : janggu.data.Cover, list(Cover), janggu.data.Track or list(Track)
        One or more track objects.
    chrom : str
        chromosome name.
    start : int
        The start of the required interval.
    end : int
        The end of the required interval.
    figsize : tuple(int, int)
        Figure size passed on to matplotlib.
    plottype : None or list(str)
        Plot type indicates whether to plot coverage tracks as line plots,
        heatmap, or seqplot using 'line' or 'heatmap', respectively.
        By default, all coverage objects are depicted as line plots if plottype=None.
        Otherwise, a list of types must be supplied containing the plot types for each
        coverage object explicitly. For example, ['line', 'heatmap', 'seqplot'].
        While, 'line' and 'heatmap' can be used for any type of coverage data,
        'seqplot' is reserved to plot sequence influence on the output. It is
        intended to be used in conjunction with 'input_attribution' method which
        determines the importance of paricular sequence letters for the output prediction.

    Returns
    -------
    matplotlib Figure
        A matplotlib figure illustrating the genome browser-view of the coverage
        objects for the given interval.
        To depict and save the figure the native matplotlib functions show()
        and savefig() can be used.
    """

    tracks = _to_list(tracks)

    for track in tracks:
        if not isinstance(track, Track):
            warnings.warn('Convert the Dataset object to proper Track objects.'
                          ' In the future, only Track objects will be supported.',
                          FutureWarning)
            if plottypes is None:
                plottypes = ['line'] * len(tracks)

            assert len(plottypes) == len(tracks), \
                "The number of cover objects must be the same as the number of plottyes."
            break

    def _convert_to_track(cover, plottype):
        if plottype == 'heatmap':
            track = HeatTrack(cover)
        elif plottype == 'seqplot':
            track = SeqTrack(cover)
        else:
            track = LineTrack(cover)
        return track

    tracks_ = []
    for itrack, track in enumerate(tracks):
        if isinstance(track, Track):
            tracks_.append(track)
        else:
            warnings.warn('Convert the Dataset object to proper Track objects.'
                          ' In the future, only Track objects will be supported.',
                          FutureWarning)
            tracks_.append(_convert_to_track(track, plottypes[itrack]))

    tracks = tracks_
    headertrack = 2
    trackheights = 0
    for track in tracks:
        trackheights += track.height
    spacer = len(tracks) - 1

    grid = plt.GridSpec(headertrack + trackheights + spacer,
                        10, wspace=0.4, hspace=0.3)
    fig = plt.figure(figsize=figsize)

    # title and reference track
    title = fig.add_subplot(grid[0, 1:])

    title.set_title(chrom)
    plt.xlim([0, end - start])
    title.spines['right'].set_visible(False)
    title.spines['top'].set_visible(False)
    title.spines['left'].set_visible(False)
    plt.xticks([0, end-start], [start, end])
    plt.yticks(())

    y_offset = 1
    for track in tracks:
        y_offset += 1

        track.add_side_bar(fig, grid, y_offset)
        track.plot(fig, grid, y_offset, chrom, start, end)
        y_offset += track.height

    return (fig)


class Track(object):
    """General track

    Parameters
    ----------

    data : Cover object
        Coverage object
    height : int
        Track height.
    """
    def __init__(self, data, height):
        self.height = height
        self.data = data

    @property
    def name(self):
        """Track name"""
        return self.data.name

    def add_side_bar(self, fig, grid, offset):
        """Side-bar"""
        # side bar indicator for current cover
        ax = fig.add_subplot(grid[(offset): (offset + self.height), 0])

        ax.set_xticks(())
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([0.5])
        ax.set_yticklabels([self.name])

    def get_track_axis(self, fig, grid, offset, height):
        """Returns axis object"""
        return fig.add_subplot(grid[offset:(offset + height), 1:])

    def get_data(self, chrom, start, end):
        """Returns data to plot."""
        return self.data[chrom, start, end][0, :, :, :]


class LineTrack(Track):
    """Line track

    Visualizes genomic data as line plot.

    Parameters
    ----------

    data : Cover object
        Coverage object
    height : int
        Track height. Default=3
    linestyle : str
        Linestyle for plot
    marker : str
        Marker code for plot
    color : str
        Color code for plot
    linewidth : float
        Line width.
    """
    def __init__(self, data, height=3, linestyle='-', marker='o', color='b',
                 linewidth=2):
        super(LineTrack, self).__init__(data, height)
        self.height = height * len(data.conditions)
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.marker = marker
        self.color = color

    def plot(self, fig, grid, offset, chrom, start, end):
        """Plot line track."""
        coverage = self.get_data(chrom, start, end)
        offset_ = offset
        trackheight = self.height//len(self.data.conditions)

        def _get_xy(cov):
            xvalue = np.where(np.isfinite(cov))[0]
            yvalue = cov[xvalue]
            return xvalue, yvalue

        for i, condition in enumerate(self.data.conditions):
            ax = self.get_track_axis(fig, grid, offset_, trackheight)
            offset_ += trackheight
            if coverage.shape[1] == 2:
                #both strands are covered separately
                xvalue, yvalue = _get_xy(coverage[:, 0, i])
                ax.plot(xvalue, yvalue,
                        linewidth=self.linewidth,
                        linestyle=self.linestyle,
                        color=self.color, label="+", marker='+')
                xvalue, yvalue = _get_xy(coverage[:, 1, i])
                ax.plot(xvalue, yvalue,
                        linewidth=self.linewidth,
                        linestyle=self.linestyle,
                        color=self.color, label="-", marker=1)
                ax.legend()
            else:
                xvalue, yvalue = _get_xy(coverage[:, 0, i])
                ax.plot(xvalue, yvalue, linewidth=self.linewidth,
                        color=self.color,
                        linestyle=self.linestyle,
                        marker=self.marker)
            ax.set_yticks(())
            ax.set_xticks(())
            ax.set_xlim([0, end-start])
            if len(self.data.conditions) > 1:
                ax.set_ylabel(condition, labelpad=12)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


class SeqTrack(Track):
    """Sequence Track

    Visualizes sequence importance.

    Parameters
    ----------

    data : Cover object
        Coverage object
    height : int
        Track height. Default=3
    """
    def __init__(self, data, height=3):
        super(SeqTrack, self).__init__(data, height)

    def plot(self, fig, grid, offset, chrom, start, end):
        """Plot sequence track"""

        if len(self.data.conditions) % len(NMAP) == 0:
            alphabetsize = len(NMAP)
            MAP = NMAP
        elif len(self.data.conditions) % len(PMAP) == 0:  # pragma: no cover
            alphabetsize = len(PMAP)
            MAP = PMAP
        else:  # pragma: no cover
            raise ValueError(
                "Coverage tracks seems not represent biological sequences. "
                "The last dimension must be divisible by the alphabetsize.")

        for cond in self.data.conditions:
            if cond[0] not in MAP:
                raise ValueError(
                    "Coverage tracks seems not represent biological sequences. "
                    "Condition names must represent the alphabet letters.")

        coverage = self.get_data(chrom, start, end)
        # project higher-order sequence structure onto original sequence.
        coverage = coverage.reshape(coverage.shape[0], -1)
        coverage = coverage.reshape(coverage.shape[:-1] +
                                    (alphabetsize,
                                     int(coverage.shape[-1]/alphabetsize)))
        coverage = coverage.sum(-1)

        ax = self.get_track_axis(fig, grid, offset, self.height)
        x = np.arange(coverage.shape[0])
        y_figure_offset = np.zeros(coverage.shape[0])
        handles = []
        for letter in MAP:
            handles.append(ax.bar(x, coverage[:, MAP[letter]],
                                  bottom=y_figure_offset,
                                  color=sns.color_palette("hls",
                                                          len(MAP))[MAP[letter]],
                                  label=letter))
            y_figure_offset += coverage[:, MAP[letter]]
        ax.legend(handles=handles)
        ax.set_yticklabels(())
        ax.set_yticks(())
        ax.set_xticks(())
        ax.set_xlim([0, end-start])


class HeatTrack(Track):
    """Heatmap Track

    Visualizes genomic data as heatmap.

    Parameters
    ----------

    data : Cover object
        Coverage object
    height : int
        Track height. Default=3
    """
    def __init__(self, data, height=3):
        super(HeatTrack, self).__init__(data, height)

    def plot(self, fig, grid, offset, chrom, start, end):
        """Plot heatmap track."""
        ax = self.get_track_axis(fig, grid, offset, self.height)
        coverage = self.get_data(chrom, start, end)

        im = ax.pcolor(coverage.reshape(coverage.shape[0], -1).T)

        if coverage.shape[-2] == 2:
            ticks = [':'.join([x, y]) for y, x \
                     in product(['+', '-'], self.data.conditions)]
        else:
            ticks = self.data.conditions

        ax.set_yticklabels(ticks)
        ax.set_xticks(())
        ax.set_yticks(np.arange(0, len(ticks) + 1, 1.0))
        ax.set_xlim([0, end-start])
