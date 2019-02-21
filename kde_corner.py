"""kde_corner - Corner plots with proper KDE defined uncertainties.

"""

from __future__ import print_function

# may be needed now. IDK
from matplotlib import use
use("PDF")

# Todo, type hints force python >= 3.5
from typing import Union
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
plt.rcParams["font.family"] = "serif"

__all__ = ['kde_corner', 'run_2D_KDE']

import seaborn as sns

def reflect(samps, othersamps = None, reflect_cut = 0.2):
    the_min = min(samps)
    the_max = max(samps)

    inds = np.where((samps < the_min*(1. - reflect_cut) + the_max*reflect_cut) & (samps > the_min))
    pad_samples = np.concatenate((samps, the_min - (samps[inds] - the_min)))
    if np.all(othersamps != None):
        pad_other = np.concatenate((othersamps, othersamps[inds]))

    inds = np.where((samps > the_min*reflect_cut + the_max*(1. - reflect_cut)) & (samps < the_max))
    pad_samples = np.concatenate((pad_samples, the_max + (the_max - samps[inds])))
    if np.all(othersamps != None):
        pad_other = np.concatenate((pad_other, othersamps[inds]))
        return pad_samples, pad_other

    return pad_samples

def reflect_2D(samps1, samps2, reflect_cut = 0.2):
    pad_samps1, pad_samps2 = reflect(samps1, samps2, reflect_cut = reflect_cut)
    pad_samps2, pad_samps1 = reflect(pad_samps2, pad_samps1, reflect_cut = reflect_cut)

    return pad_samps1, pad_samps2


def every_other_tick(ticks):
    """Matplotlib loves tick labels!"""

    labels = []

    for i in range(len(ticks) - 1):
        if i % 2 == len(ticks) % 2:
            labels.append(ticks[i])
        else:
            labels.append("")
    labels.append("")
    return labels

def run_2D_KDE(samples_i, samples_j, bw_method = 0.1, contours = [0.317311, 0.0455003], steps = 100):

    pad_samples_i, pad_samples_j = reflect_2D(samples_i, samples_j)

    xvals, yvals = np.meshgrid(np.linspace(min(samples_i), max(samples_i), steps),
                               np.linspace(min(samples_j), max(samples_j), steps))

    try:
        kernel = gaussian_kde(np.array([pad_samples_i, pad_samples_j]), bw_method = bw_method)
    except:
        print("Couldn't make KDE!")
        return xvals*0, [0], xvals, yvals, lambda x: 0*x
        

    eval_points = np.array([xvals.reshape(steps**2), yvals.reshape(steps**2)])

    kernel_eval = kernel(eval_points)
    norm_term = kernel_eval.sum()
    kernel_eval /= norm_term
    kernel_sort = np.sort(kernel_eval)
    kernel_eval = np.reshape(kernel_eval, (steps, steps))

    kernel_cum = np.cumsum(kernel_sort)

    levels = [kernel_sort[np.argmin(abs(kernel_cum - item))] for item in contours[::-1]]

    return kernel_eval, levels, xvals, yvals, lambda x: kernel(x)/norm_term

def kde_corner(samples, labels, pltname = None, figsize = None, pad_side = None,
               pad_between = None, label_coord = -0.25, contours = [0.317311, 0.0455003],
               colors = None, bw_method = 0.1, labelfontsize = None):
    # type: (samples, labels, str, figsize, pad_side, pad_between, float, contours, colors, float, labelfontsize) -> Union[None, matplotlib.pyplot.figure]
    """
        labels is a list of length n_var.
        I recommend setting bw_method to 0.1.

        Parameters
        ----------
        samples: array_like of floats
            An array of variables and samples.

        labels: array_like of strings
            This is the the labels that should be added to the  
            Should be one label per variable to be plotted. 

        pltname: str
            If `pltname` is given, resulting figure is saved and `kde_corner` returns `None`.

        figsize: float
            Defaults to ???

        pad_side: float
            Defaults to ???

        pad_between: float
            Defaults to ???

        label_coord: float

        contours: array-like of floats
            0.317311 is 1-sigma, and 0.0455003 is 2-sigma

        colors: array-like

        bw_method: float

        lablefontsize:

        Returns
        -------
        matplotlib.pyplot.figure or None:
            Returns the figure containing the corner plot. Alternatively if `pltname` is used `None`

        Notes
        -----
        It is best to use multiple contours or multiple data sets, but not both.

        """

    try:
        samples[0][0][0]
    except IndexError:
        # there is only 1 data set here
        N_datasets = 1
        
        if len(samples) > len(samples[0]):
            samples = np.transpose(samples)
        else:
            samples = samples
        
        n_var = len(samples)

        alpha = 1
        
        samples = [samples]   # This way we can loop over the set of samples.
    else:
        # There is a collection of data sets here
        N_datasets = len(samples)       

        for i, samp in enumerate(samples):
            if len(samp) > len(samp[0]):
                samples[i] = np.transpose(samp)
            else:
                samples[i] = samp #TODO: fix
        
        n_var = len(samples[0])

        alpha = 0.5

    if figsize == None:
        figsize = [4 + 1.5*n_var]*2

    if pad_between == None:
        pad_between = 0.1/figsize[0]

    if pad_side == None:
        pad_side = pad_between*8.

    if labelfontsize == None:
        labelfontsize = 6 + int(figsize[0])
    print("labelfontsize ", labelfontsize)

    if colors == None:
        # TODO: make this default back to greyscale for 1 dataset
        colors = []
        grayscales = np.linspace(0.8, 0.4, len(contours))
        # colors = [[item]*3 for item in grayscales].
        colors = [sns.color_palette("gray_r", n_colors=len(contours)),
                  sns.color_palette("Blues", n_colors=len(contours)),
                  sns.color_palette("Oranges", n_colors=len(contours))]
                  # sns.color_palette("Reds", n_colors=len(contours)),
                  # sns.color_palette("Purples", n_colors=len(contours))]
        # colors = 
        # for _ in samples:
        #     colors = [[item]*3 for item in grayscales]
        #     colors = cm.ScalarMappable(cmap='Blues').to_rgba(grayscales, alpha=0.5) 
        # colors = [plt.get_cmap('Blues'), plt.get_cmap('Purples') ] # hard code for two samples

    #colors = colors[::-1]

    fig = plt.figure(figsize = figsize)

    plt_size = (1. - pad_side - n_var*pad_between)/n_var
    plt_starts = pad_side + np.arange(float(n_var))*(plt_size + pad_between)
    
    plt_limits = []
    plt_ticks = []

    for i in range(n_var):
        ax = fig.add_axes([plt_starts[i], plt_starts[n_var - 1 - i], plt_size, plt_size])
        #ax.hist(samples[i])

        for samp, color in zip(samples, colors):
            pad_samples = reflect(samp[i])
            try:
                kernel = gaussian_kde(pad_samples, bw_method = bw_method)
            except:
                print("Couldn't run KDE!")
                kernel = lambda x: x*0

            vals = np.linspace(min(samp[i]), max(samp[i]), 1000)

            kernel_eval = kernel(vals)
            kernel_eval /= kernel_eval.sum()
            kernel_sort = np.sort(kernel_eval)
            kernel_cum = np.cumsum(kernel_sort)

            levels = [kernel_sort[np.argmin(abs(kernel_cum - item))] for item in contours[::-1]] + [1.e20]
            print("1D levels ", levels)

            for j in range(len(contours)):
                ax.fill_between(vals, 0, (kernel_eval > levels[j])*(kernel_eval < levels[j+1])*kernel_eval, color = color[j], alpha=alpha)
            ax.plot(vals, kernel_eval, color = 'k')
            ax.set_ylim(0, ax.get_ylim()[1])
            
            # TODO: update so this does not get redone each time. Pull out of loop and do use slicing?
        ax.set_yticks([])
        if i < n_var - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(labels[i], fontsize=labelfontsize)
            plt.xticks(rotation = 45)
            plt.yticks(rotation = 45)

        plt_limits.append(ax.get_xlim())
        plt_ticks.append(ax.get_xticks())

        if plt_ticks[-1][-1] > plt_limits[-1][-1] + 1.e-9:
            print("Weird! Deleting.")
            plt_ticks[-1] = plt_ticks[-1][:-1]
        if plt_ticks[-1][0] < plt_limits[-1][0] - 1.e-9:
            plt_ticks[-1] = plt_ticks[-1][1:]
            print("Weird! Deleting.")

        if i >= n_var - 1:
            #ax.set_xticklabels(every_other_tick(plt_ticks[i]))
            ax.yaxis.set_label_coords(label_coord, 0.5)
            ax.xaxis.set_label_coords(0.5, label_coord)
        

    for i in range(n_var - 1):
        for j in range(i+1, n_var):
            for samp, color in zip(samples, colors):
                kernel_eval, levels, xvals, yvals, kfn = run_2D_KDE(samp[i], samp[j], bw_method = bw_method, contours = contours)

                ax = fig.add_axes([plt_starts[i], plt_starts[n_var - 1 - j], plt_size, plt_size])

                # TODO: only use alpha if multiple samples
                ax.contourf(xvals, yvals, kernel_eval, levels = levels + [1], colors = color, alpha=alpha)
                ax.contour(xvals, yvals, kernel_eval, levels = levels, colors = 'k')

                # TODO: update so this does not get redone each time. Pull out of loop and do use slicing?
            ax.set_xlim(plt_limits[i])
            ax.set_ylim(plt_limits[j])

            ax.set_xticks(plt_ticks[i])
            ax.set_yticks(plt_ticks[j])

            if i > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(labels[j], fontsize=labelfontsize)
                #ax.set_yticklabels(every_other_tick(plt_ticks[j]), rotation = 45)

            if j < (n_var - 1):
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[i], fontsize=labelfontsize)
                #ax.set_xticklabels(every_other_tick(plt_ticks[i]), rotation = 45)
                print("xticks ", labels[i], ax.get_xticks())#, every_other_tick(plt_ticks[i]), plt_limits[i]

            ax.yaxis.set_label_coords(label_coord, 0.5)
            ax.xaxis.set_label_coords(0.5, label_coord)
                


    if pltname == None:
        return fig
    else:
        plt.savefig(pltname, bbox_inches = 'tight')
        plt.close()


def get_color():
    """A generator to cycle through a fix set of color palettes.

    This allows kde_corner to work with arbitrary number of datasets,
    though if you get too long it will not look good since colors will
    start to repeat.
    """
    colors = [sns.color_palette("Blues", n_colors=len(contours)), 
              sns.color_palette("Reds", n_colors=len(contours))]

    yield cycle(colors)
