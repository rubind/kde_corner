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
from scipy.stats import gaussian_kde, scoreatpercentile
try:
    from tqdm import trange
except:
    trange = range
    
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

def latex_1sigma_credible(vals, dataset_labels):
    # vals is 2d: datasets by samples
    
    percentiles = scoreatpercentile(vals, [15.8655, 50., 84.1345], axis = 1)
    
    smallest_unc = (percentiles[1:] - percentiles[:-1]).min()
    decimal_places = int(np.around(0.80102999566 - np.log10(smallest_unc)))

    fmt_txt = "%." + str(decimal_places) + "f"

    latex_txt = ""
    for i in range(len(vals)):
        if dataset_labels[i] != "":
            the_label = dataset_labels[i] + ": "
        else:
            the_label = ""
            
        latex_txt += (the_label + "$" + fmt_txt + "^{+" + fmt_txt + "}_{-" + fmt_txt + "}$\n") % (percentiles[1,i], percentiles[2,i] - percentiles[1,i], percentiles[1,i] - percentiles[0,i])
    #print(latex_txt)
    return latex_txt[:-1] # Leave off last \n
    

def kde_corner(samples, labels, pltname = None, figsize = None, pad_side = None,
               pad_between = None, label_coord = -0.25, contours = [0.317311, 0.0455003],
               colors = None, bw_method = 0.1, labelfontsize = None, dataset_labels = None,
               show_contours = None, ax_limits=[], truths=None, titles=None):
    # type: (samples, labels, str, figsize, pad_side, pad_between, float, contours,
    #        colors, float, labelfontsize) -> Union[None, matplotlib.pyplot.figure]
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

    dataset_labels: Labels for each dataset

    titles:
        15.8655, 50, and 84.1345 percentiles, not gaurenteed to match the KDE CR.

    Returns
    -------
    matplotlib.pyplot.figure or None:
        Returns the figure containing the corner plot. Alternatively if `pltname` is used `None`

    Notes
    -----
    It is best to use multiple contours or multiple data sets, but not both.

    """
    samples = np.array(samples)
    print("samples.shape", samples.shape)
    
    try:
        samples[0][0][0] # Datasets, parameters, samples
    except IndexError:
        # there is only 1 data set here
        N_datasets = 1
        
        if len(samples) > len(samples[0]):
            samples = np.transpose(samples)
        else:
            samples = samples    #todo(this seems unnecessary)
        
        n_var = len(samples)

        alpha = 1
        
        samples = np.expand_dims(samples, axis=0)   # This way we can loop over the set of samples.
    else:
        # There is a collection of data sets here
        N_datasets = len(samples)       

        for i, samp in enumerate(samples):
            if len(samp) > len(samp[0]):
                samples[i] = np.transpose(samp)   # this does not work well with np.arrays.
            else:
                samples[i] = samp #TODO: fix
        
        n_var = len(samples[0])

        # alpha = 0.5
        alpha = 0.4

    # have this after try-except-else block so transpose works.
    samples = np.array(samples)

    if figsize == None:
        figsize = [4 + 1.5*n_var]*2

    if pad_between == None:
        pad_between = 0.1/figsize[0]

    if pad_side == None:
        pad_side = pad_between*8.

    if labelfontsize == None:
        labelfontsize = 6 + int(figsize[0])
    print("labelfontsize ", labelfontsize)

    if dataset_labels == None:
        dataset_labels = [""]*len(samples)

    if show_contours == None:
        show_contours = [1]*len(samples)

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
        #     colors = cm.ScalarMappable(cmap='Blues').to_rgba(grayscales, alpha=0.5) [
        # colors = [plt.get_cmap('Blues'), plt.get_cmap('Purples') ] # hard code for two samples
    truth_color = "#4682b4"
    #colors = colors[::-1]
    print("colors", colors)

    fig = plt.figure(figsize = figsize)

    plt_size = (1. - pad_side - n_var*pad_between)/n_var
    plt_starts = pad_side + np.arange(float(n_var))*(plt_size + pad_between)
    
    plt_limits = []
    plt_ticks = []

    for i in range(n_var):
        ax = fig.add_axes([plt_starts[i], plt_starts[n_var - 1 - i], plt_size, plt_size])
        #ax.hist(samples[i])
        ax.set_title(latex_1sigma_credible(samples[:,i,:], dataset_labels))

        
        for samp, show_contour, color, dataset_label in zip(samples, show_contours, colors, dataset_labels):
            if show_contour:
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
                    if (i == 0) and (j == 0):
                        the_label = dataset_label
                    else:
                        the_label = ""

                    ax.fill_between(vals, 0, (kernel_eval > levels[j])*(kernel_eval < levels[j+1])*kernel_eval, color = color[j], alpha=alpha, label = the_label)
                    if the_label != "":
                        fig.legend(bbox_to_anchor=(1.0, 1.0))#loc = 'best')

                if len(samples) == 1:
                    plot_this_color = 'k'
                else:
                    plot_this_color = color
                    try:
                        plot_this_color[0][0]
                        plot_this_color = plot_this_color[0]
                    except:
                        pass
                print("plot_this_color", plot_this_color)
                ax.plot(vals, kernel_eval, color = plot_this_color)
        ax.set_ylim(0, ax.get_ylim()[1])
            
        # TODO: update so this does not get redone each time. Pull out of loop and do use slicing?
        ax.set_yticks([])
        if i < n_var - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(labels[i], fontsize=labelfontsize)
            plt.xticks(rotation = 45)
            plt.yticks(rotation = 45)


        cur_min = samples[:,i,:].min()
        cur_max = samples[:,i,:].max()

        print("cur_min, cur_max", cur_min, cur_max)
        
        if float((samples[:,i,:] < cur_min*0.98 + cur_max*0.02).sum())/samples[:,i,:].size > 0.004 or float((samples[:,i,:] > cur_max*0.98 + cur_min*0.02).sum())/samples[:,i,:].size > 0.004:
            ax.set_xlim(cur_min, cur_max)
            print("setting to cur_min, cur_max")
            
        #TODO(This is the default operation, but overwriting plt_limits will change the plot limits. I think.)
        if not ax_limits == []:
            ax.set_xlim(ax_limits[i])
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

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)
        

    for i in trange(n_var - 1):
        for j in range(i+1, n_var):
            ax = fig.add_axes([plt_starts[i], plt_starts[n_var - 1 - j], plt_size, plt_size])

            for samp, show_contour, color in zip(samples, show_contours, colors):
                if show_contour:
                    kernel_eval, levels, xvals, yvals, kfn = run_2D_KDE(samp[i], samp[j], bw_method = bw_method, contours = contours)

                    
                    # TODO: only use alpha if multiple samples
                    
                    print("levels", levels + [1])
                    ax.contourf(xvals, yvals, kernel_eval, levels = levels + [1], colors = color + [(1,1,1)], alpha=alpha)
                    ax.contour(xvals, yvals, kernel_eval, levels = levels, colors = color)
                    print("samp.shape", samp.shape)
                else:
                    plt.plot(np.median(samp[i]), np.median(samp[j]), 'o', color = color)

            if truths is not None:
                if truths[j] is not None and truths[i] is not None:
                    ax.plot(truths[i], truths[j], "s", color=truth_color)
                if truths[i] is not None:
                    ax.axvline(truths[i], color=truth_color)
                if truths[j] is not None:
                    ax.axhline(truths[j], color=truth_color)

                # TODO: update so this does not get redone each time. Pull out of loop and do use slicing?
            ax.set_xlim(plt_limits[i])
            ax.set_ylim(plt_limits[j])

            ax.set_xticks(plt_ticks[i])  
            plt.xticks(rotation=45)# updated by benjamin.rose@me.com to always rotate labels.
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
