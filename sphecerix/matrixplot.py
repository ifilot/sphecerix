# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def visualize_matrices(matrices, opnames, labels, numcols = 3,
                       highlight_groups = None, filename = None,
                       figsize=(7,5), xlabelrot = 0):
    """
    Visualize matrix representations of the symmetry operations
    """
    # grab data
    fig, ax = plt.subplots(len(matrices) // numcols, 
                           numcols, dpi=144, figsize=figsize)
    
    for i,(opname,mat) in enumerate(zip(opnames,matrices)):
        axh = ax[i//numcols, i%numcols]
        plot_matrix(axh, mat, labels, title=opname, xlabelrot = xlabelrot)
        
        if highlight_groups:
            plot_highlight_groups(axh, highlight_groups, mat)
    
    plt.tight_layout()
    
    if filename:
        print('Storing: %s' % filename)
        plt.savefig(filename)
        plt.close()

def plot_highlight_groups(axh, blocks, mat):
    # add semitransparent hash
    sumblocks = np.sum(g[0] for g in blocks)
    cum = 0
    for g in blocks:
        rect = patches.Rectangle((cum - 0.5, cum - 0.5), g[0], g[0], 
                                 linewidth=1,
                                 zorder=5,
                                 fill = None,
                                 hatch='///',
                                 alpha=0.5)
        axh.add_patch(rect)
        cum += g[0]
        
    # add red outline
    cum = 0
    for g in blocks:
        rect = patches.Rectangle((cum - 0.5, cum - 0.5), g[0], g[0], 
                                 linewidth=1.5, edgecolor='red',
                                 linestyle='solid',
                                 facecolor='none',
                                 zorder=5,
                                 alpha=1.0)
        axh.add_patch(rect)
        
        axh.text(cum+g[0]/2-0.5, cum+g[0]/2-0.5, '%i' % round(np.trace(mat[cum:cum+g[0],cum:cum+g[0]])),
                 color='red', horizontalalignment='center', verticalalignment='center',
                 bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), ),
                 zorder=6)
        
        k = (cum+1) > sumblocks/2
        
        if k:
            label = r'%i $\times$ %s $\rightarrow$' % (g[1],g[2])
        else:
            label = r'$\leftarrow$ %i $\times$ %s' % (g[1],g[2])
        
        axh.text(cum+(-0.75 if k else g[0] - 0.25), cum+g[0]/2-0.5, 
                 label,
                 fontsize=6,
                 color='red', 
                 horizontalalignment='right' if k else 'left', 
                 verticalalignment='center',
                 bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), ),
                 zorder=6)
        
        cum += g[0]


def plot_matrix(ax, mat, labels, title = None, xlabelrot = 0, 
                highlight_groups = None, **kwargs):
    """
    Produce plot of matrix
    """
    ax.imshow(mat, vmin=-1, vmax=1, cmap='PiYG')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(i, j, '%.2f' % mat[j,i], ha='center', va='center',
                    fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0] - 0.5,
              color='black', linestyle='--', linewidth=1)
    ax.vlines(np.arange(1, mat.shape[0])-0.5, -0.5, mat.shape[0] - 0.5,
              color='black', linestyle='--', linewidth=1)
    
    # add basis functions as axes labels
    ax.set_xticks(np.arange(0, mat.shape[0]))
    ax.set_xticklabels(labels, rotation=xlabelrot)
    ax.set_yticks(np.arange(0, mat.shape[0]))
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # add title if supplied
    if title:
        if 'titlefontsize' in kwargs:
            titlefontsize = kwargs['titlefontsize']
            ax.set_title(title, fontsize=titlefontsize)
        else:
            ax.set_title(title)
        
    if highlight_groups is not None:
        plot_highlight_groups(ax, highlight_groups, mat)
    