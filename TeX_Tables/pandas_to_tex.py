#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:41:27 2020

@author: s.bykov
"""
import pandas as pd
import numpy as np

def _tex_error(par, Min, Max, funct = lambda x:x, k = 2, ):
    '''
    example: tex_error(10.24,10.19,10.393,k=1,funct=lambda x: 100*x)

    tex_error(np.log10(1e-8),np.log10(0.9e-8),np.log10(1.2e-8),k=2,
    funct=lambda x: 10**x/1e-8)

    '''
    if np.isnan(par) or np.isnan(Min) or np.isnan(Max):
        return None

    par,Min,Max=funct(par),funct(Min),funct(Max)


    low = f'%.{k}f' %(par - Min)
    hi  = f'%.{k}f' %(Max - par)
    parr = f'%.{k}f' %(par)

    if float(Min) == float(Max) and float(Min)==0:
        par_tex = '$' + parr + '$'
        return par_tex
    if hi==low:
        par_tex = '$' + parr + '\pm' + hi + '$'
    else:
        par_tex = '$' + parr + '^{+' + hi + '}_{-' + low + '}$'
    return par_tex


def tex_error(row,par,funct=lambda x:x,k=3):
    '''
    same as above, but applying to pandas ROW

    '''
    err=_tex_error(row[par],row[par+'_lo'],row[par+'_hi'],funct=funct,k=k)
    return err

def latex_format(val,funct=lambda x:x,k=3):
    '''
    used for values for which errors are unknown/not necessary
    '''
    if isinstance(val, str):
        return val
    if np.isnan(val):
        return None
    if type(val) is  str:
        return val
    else:
        val=f'%.{k}f'%funct(val)
        return val

def make_latex_table(DataFrame,
                     free_columns,free_columns_functions,free_columns_formats,
                     err_columns,err_functions,err_formats,
                     ):
    '''
    Make a dataframe with latex strings in cells.

    Parameters
    ----------
    DataFrame :
        dataframe which consists of values with no errors and columns with upper and lower limits.
    free_columns : TYPE
        columns that are to be printed without lower and upper errors
    free_columns_functions : TYPE
        functions to apply to free_columns. For instance, divide exposure values by 1000 to get ksec instead of sec
    free_columns_formats : TYPE
        formats for free columns. E.g. chi squared values with two decimal points
    err_columns : TYPE
        the same as above, but for columns which have 'error' counterpart, e.g. flux, flux_lo, flux_hi.
    err_functions : TYPE
        as above. Applied ti both mean value and lower/upper estimates.
    err_formats : TYPE
        as above.

    Returns
    -------
    df_tex : TYPE
        dataframe with latex strings in cells.

    '''
    df_orig=DataFrame
    df_tex=pd.DataFrame()


    for col,func,k in zip(free_columns,free_columns_functions,free_columns_formats):
        df_tex[col]=df_orig[col].apply(lambda x: latex_format(x,func,k))

    for col,func,k in zip(err_columns,err_functions,err_formats):
        df_tex[col]=df_orig.apply(lambda x: tex_error(x, col,func,k),axis=1)
    return df_tex

def save_latex_table(df_tex,savepath='./log.tex',
                     columns_to_write='DEFAULT',
                     columns_names='DEFAULT',
                     transpose=0):
    '''


    Parameters
    ----------
    df_tex : dataframe from make_latex_table fucntion

    savepath : TYPE, optional
        DESCRIPTION. The default is './log.tex'.
    columns_to_write : TYPE, optional
        DESCRIPTION. In case you want to save only a few columns. Print their names here
        The default is 'DEFAULT'.
    columns_names : TYPE, optional
        DESCRIPTION. In case you want to rename columns which will be saved. Print their new names here,
        The default is 'DEFAULT'.
    transpose : TYPE, optional
        DESCRIPTION. Set 1, If you want to transpose the final table. The default is 0.


    '''
    if not transpose:
        if columns_to_write=='DEFAULT' and columns_names=='DEFAULT':
            columns_to_write=list(df_tex.columns.values)
            columns_names=columns_to_write

        return df_tex.to_latex(buf=savepath,escape = False, index = transpose,
                        columns=columns_to_write,header=columns_names)

    if  transpose:
        '''
        this part is pretty dump, because when there is no transpose operation,
        it is easier to pick up columns to save, but if there is a transpose operation,
        renaming is not as simple as if transpose=0
        '''
        if columns_to_write=='DEFAULT' and columns_names=='DEFAULT':
            columns_to_write=list(df_tex.columns.values)
            columns_names=columns_to_write



        df_tex=df_tex[columns_to_write]
        new_names=dict(zip(columns_to_write,columns_names))
        df_tex=df_tex.rename(columns=new_names)
        df_tex.reset_index(inplace=True,drop=True)
        df_tex=df_tex.transpose()

        return df_tex.to_latex(buf=savepath,escape = False, index = transpose,
                        )
