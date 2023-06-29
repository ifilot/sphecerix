# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def visualize_matrices(symops, numcols = 3,
                       highlight_groups = None, filename = None,
                       figsize=(7,5), xlabelrot = 0):
    """
    Visualize matrix representations of the symmetry operations
    """
    # grab data
    matrices = symops.operation_matrices
    operations = symops.operations
    bfs = symops.mol.basis
    
    fig, ax = plt.subplots(len(operations) // numcols, 
                           numcols, dpi=144, figsize=figsize)
    
    for i,(op,mat) in enumerate(zip(operations,matrices)):
        axh = ax[i//numcols, i%numcols]
        plot_matrix(axh, mat, bfs, title=op.name, xlabelrot = xlabelrot)
        
        if highlight_groups:
            plot_highlight_groups(axh, highlight_groups, mat)
    
    plt.tight_layout()
    
    if filename:
        print('Storing: %s' % filename)
        plt.savefig(filename)
        plt.close()

def plot_highlight_groups(axh, groups, mat):
    # add semitransparent hash
    cum = 0
    for g in groups:
        rect = patches.Rectangle((cum - 0.5, cum - 0.5), g, g, 
                                 linewidth=1,
                                 zorder=5,
                                 fill = None,
                                 hatch='///',
                                 alpha=0.5)
        axh.add_patch(rect)
        cum += g
        
    # add red outline
    cum = 0
    for g in groups:
        rect = patches.Rectangle((cum - 0.5, cum - 0.5), g, g, 
                                 linewidth=1.5, edgecolor='red',
                                 linestyle='solid',
                                 facecolor='none',
                                 zorder=5,
                                 alpha=1.0)
        axh.add_patch(rect)
        
        axh.text(cum+g/2-0.5, cum+g/2-0.5, '%i' % round(np.trace(mat[cum:cum+g,cum:cum+g])),
                 color='red', horizontalalignment='center', verticalalignment='center',
                 bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), ),
                 zorder=6)
        
        cum += g


def plot_matrix(ax, mat, bfs, title = None, xlabelrot = 0):
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
    labels = [bf.name for bf in bfs]
    ax.set_xticks(np.arange(0, mat.shape[0]))
    ax.set_xticklabels(labels, rotation=xlabelrot)
    ax.set_yticks(np.arange(0, mat.shape[0]))
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # add title if supplied
    if title:
        ax.set_title(title)
    