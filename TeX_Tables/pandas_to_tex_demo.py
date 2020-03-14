#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:41:27 2020

@author: s.bykov
"""
import pandas as pd
import numpy as np
from pandas_to_tex import *

#%% test
df=pd.read_csv('sample_data.txt', delim_whitespace=1)
null=lambda x: x
free_columns=['ObsId','expo','chi']
free_columns_functions=[null,lambda x: x/1000,null]
free_columns_formats=[0,1,2]

err_columns=['nh','PI','norm','par1','par2','par3','flux']
err_functions=[null,null,lambda x: 1000*x,null,null,null,lambda x: 10**x/1e-8]
err_formats=[2,2,2,1,2,2,2]

#%%few pars
transpose=0
df_tex=make_latex_table(df,
                      free_columns=free_columns, free_columns_functions=free_columns_functions, free_columns_formats=free_columns_formats,
                      err_columns=err_columns, err_functions=err_functions, err_formats=err_formats,
                      )

save_latex_table(df_tex, savepath='few_pars.tex',
                 columns_to_write=['ObsId','nh','expo','flux'],
                 columns_names=['ObsID','$N_H, 10^{22} cm^{-2}$','Exposure, ks','$ Flux, 10^{-8} erg s^{-1} cm^{-2}$'],
                 transpose=transpose)

#%%a few pars and transpose
transpose=1
df_tex=make_latex_table(df,
                      free_columns=free_columns, free_columns_functions=free_columns_functions, free_columns_formats=free_columns_formats,
                      err_columns=err_columns, err_functions=err_functions, err_formats=err_formats,
                      )
save_latex_table(df_tex, savepath='few_pars_transpose.tex',
                 columns_to_write=['ObsId','flux'],
                 columns_names=['ObsID','$ Flux, 10^{-8} erg s^{-1} cm^{-2}$'],
                 transpose=transpose)


#%%a all pars and transpose
transpose=1
df_tex=make_latex_table(df,
                      free_columns=free_columns, free_columns_functions=free_columns_functions, free_columns_formats=free_columns_formats,
                      err_columns=err_columns, err_functions=err_functions, err_formats=err_formats,
                      )
save_latex_table(df_tex, savepath='all_pars_transpose.tex',
                 columns_to_write='DEFAULT',
                 columns_names='DEFAULT',
                 transpose=transpose)
