import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
def plot_data(data, plot_type='histogram', columns=None, figsize=(10, 6), subplot=False, logx=False, logy=False, **kwargs):
    """
    Wrapper function to plot histograms, scatter plots, or bar plots using pandas data frames.

    Parameters:
        - data: pandas DataFrame
            The input data frame containing the data to be plotted.
        - plot_type: str, optional (default: 'histogram')
            The type of plot to be generated. Options: 'histogram', 'scatter', 'bar'.
        - columns: str or list, optional (default: None)
            The column(s) to be plotted. If None, all columns will be plotted.
        - figsize: tuple, optional (default: (10, 6))
            The figure size in inches (width, height).
        - subplot: bool, optional (default: False)
            Whether to plot subplots or individual plots for each column.
        - kwargs: keyword arguments
            Additional keyword arguments to be passed to the corresponding plotting function.

    Returns:
        - None

    Raises:
        - ValueError: If an invalid plot_type is provided.

    """
    params = {'font.family' : 'serif',
                         'font.size' : 14,
                         'errorbar.capsize' : 3,
                         'lines.linewidth'   : 1.0,
                         'xtick.top' : True,
                         'ytick.right' : True,
                         'legend.fancybox' : False,
                         'xtick.major.size' : 4.0,
                         'xtick.minor.size' : 2.0,
                         'ytick.major.size' : 4.0 ,
                         'ytick.minor.size' : 2.0,
                         'xtick.direction' : 'in',
                         'ytick.direction' : 'in',
                         'xtick.color' : 'black',
                         'ytick.color' : 'black',
                         'mathtext.rm' : 'serif',
                         'mathtext.default': 'regular',
                        }
    matplotlib.rcParams.update(params)
    
    if columns is None:
        columns = data.columns.tolist()

    if not isinstance(columns, list):
        columns = [columns]

    for column in columns:
        if column not in data.columns:
            raise ValueError(f"Invalid column name: {column}. Column does not exist in the data frame.")

    if subplot:
        num_plots = len(columns)
        num_cols = int(num_plots ** 0.5) + 1
        num_rows = (num_plots - 1) // num_cols + 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        axs = axs.ravel()
        
    for i, column in enumerate(columns):
        if not subplot:
            plt.figure(figsize=figsize)

        if subplot:
            ax = axs[i]
            
        else:
            ax = plt.gca()

        if plot_type == 'histogram':
            data[column].plot.hist(ax=ax, logx=logx, logy=logy, color="#008B8B",edgecolor='black', **kwargs) #E6E6FA #6495ED
        elif plot_type == 'scatter':
            data.plot.scatter(x=column, y=kwargs.get('y'), ax=ax, **kwargs)
        elif plot_type == 'bar':
            data[column].plot.bar(ax=ax, **kwargs)
        else:
            raise ValueError("Invalid plot_type. Please choose 'histogram', 'scatter', or 'bar'.")

        ax.set_xlabel(column)

        if plot_type != 'scatter':
            ax.set_ylabel("Frequency")

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        if not subplot:
            plt.show()
        
        
    
    if subplot:
        for ax in axs[num_plots:]:
            ax.remove()
        
        plt.tight_layout()
        plt.show()
        
def trueVSpred_scatter(y_test,y_pred,figsize,**kwargs):
    """
    Wrapper function to scatter plots to asses ML model outputs.

    Parameters:
        - y_test: array
            True target
        - y_pred: array
            Predicted target
        - kwargs: keyword arguments
            Additional keyword arguments to be passed to the corresponding plotting function.

    Returns:
        - None
        
    """
    params = {'font.family' : 'serif',
                         'font.size' : 14,
                         'errorbar.capsize' : 3,
                         'lines.linewidth'   : 1.0,
                         'xtick.top' : True,
                         'ytick.right' : True,
                         'legend.fancybox' : False,
                         'xtick.major.size' : 4.0,
                         'xtick.minor.size' : 2.0,
                         'ytick.major.size' : 4.0 ,
                         'ytick.minor.size' : 2.0,
                         'xtick.direction' : 'in',
                         'ytick.direction' : 'in',
                         'xtick.color' : 'black',
                         'ytick.color' : 'black',
                         'mathtext.rm' : 'serif',
                         'mathtext.default': 'regular',
                        }
    matplotlib.rcParams.update(params)
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    sample_size = len(y_test)
    
    ax1.scatter(range(sample_size), y_test, marker='o',facecolor='#008B8B',
                edgecolor='k',alpha=0.7,s=60, label='True win rate')

    ax1.scatter(range(sample_size), y_pred, marker='s',facecolor='k',s=20, 
                edgecolor='k',label='Predicted win rate')

    ax1.plot([range(sample_size), range(sample_size)], [y_test, y_pred],
             color='gray', alpha=0.8, linestyle='-', linewidth=0.5)  # Vertical lines

    ax1.axhline(np.median(y_test), color='gray', linestyle='--',
                lw=2, label='Median Baseline',zorder=0, alpha=0.5)  # Median Baseline

    ax1.set_ylabel('Win Rate [%]')
    ax1.legend(loc=3)
    ax1.xaxis.set_ticklabels([])# Calculate the residuals
    
    
    residuals = y_test - y_pred
    # Scatter plot of residuals  
    ax2.scatter(range(sample_size), residuals, label='true$-$predicted',
               marker='o',facecolor='None',
               edgecolor='k',s=60)
    
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Residual [%]')
    ax2.axhline(0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference
    ax2.legend(loc=3)
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()
