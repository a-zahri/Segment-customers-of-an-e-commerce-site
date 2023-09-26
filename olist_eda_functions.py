#! /usr/bin/env python3
# coding: utf-8

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import datetime


def Get_Data_Path():
    """ Return le chemin des fichiers dans le répértoire Data
    Parameter: sans
    Return: le chemin des fichiers."""
    Dir = os.getcwd()
    parentDir = os.path.dirname(Dir)
    DataDir = os.path.join(parentDir, 'Data')
    return DataDir

def Get_Files(path):
    """ Return the files in the Data directory
     Parameter: File path.
     Return: the files."""
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files
    
def Get_File_names(files):
    """ Return the list of file names
     Parameter: file path
     Return: the list of filenames."""
    file_name_list = []
    for file in files:
        #index = file.index('.')
        # On enleve les parties: "olist_" et "_dataset.csv" du nom de fichier.
        file_name = 'df_'+file[6:-12] 
        file_name_list.append(file_name)    
    return file_name_list
    
def Read_Files(files, path):
    """ it reads files and returns dataFrame
     Parameter: filename, filepath.
     Return: list of dataFrame."""
    df_list = []
    for file in files:
        file_path = os.path.join(path, file)
        df_file_name = pd.read_csv(file_path)
        df_list.append(df_file_name)    
    return df_list

def Check_NaN(dic_df):
    """ Calculates and displays the percentage of nan in each df
     Parameter: dictionary of dataFrame """
    for i in dic_df:
        print(10*"=",i,10*"=")
        #(dic_df[i].isna().sum()/dic_df[i].shape[0]).sort_values(ascending=True)
        print ((dic_df[i].isna().sum()/dic_df[i].shape[0]).sort_values(ascending=False))    
      
def Drop_NaN_columns(df, thres):
    """ Drop columns that contain more than 'thres'% NaN.
     Parameter: DataFrame
     Return: dataFrame without columns that contain more than 'thres'% of nan"""
    df = df[df.columns[df.isna().sum()/df.shape[0] < thres]]
    return df
    
def Display_NaNs(df,name):
    """ displays the nan of all dataFrames.
     Parameter: DataFrame, name of each df."""
    if ((df.isna().sum()).sum()) > 1 :
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        #fig.suptitle('Pourcentage de NaN par variable')
        sns.barplot(ax=axes[0], x=(df.isna().sum()/df.shape[0])*100, y=df.columns)
        axes[0].set_title("The percentage of NaNs per variable for the dataset: "'%s'%(' '.join(name[3:].split("_"))).title(), size=12)
        
        sns.barplot(ax=axes[1], x=df.isna().sum(), y=df.columns)
        axes[1].set_title("Sum of NaNs per variable for the dataset: "'%s'%(' '.join(name[3:].split("_"))).title(),size=12)
    
def Check_Duplicate_Value(dic_df):
    """ Check for duplicates. 
    Parameter: dataFrame dictionary"""
    for i in dic_df:
        if dic_df[i].duplicated().sum()>0:
            print("The number of duplicates for", i ,"is: ", dic_df[i].duplicated().sum())

def Get_Negatif_Value(dic_df):
    """ Look for negative values.
     Parameter: dictionary of dataFrame"""
    for i in dic_df:
        for col in dic_df[i].select_dtypes('float'):
            print(30*"=")
            print(i,".",col,":",dic_df[i][col].min())

def Check_Zero_Value(dic_df):
    """ Checks for zero values.
     Parameter: dictionary of dataFrame """
    for i in dic_df:
        for col in dic_df[i].select_dtypes('float'):
            if dic_df[i][col].min()==0:
                print('The number of zeros in', col, 'is:', dic_df[i][dic_df[i][col]==0].shape[0])
                print("The number of elements in", col, 'is:', dic_df[i].shape[0])

def Drope_Zero_Value(dic_df):
    """ Removes zero values.
     Parameter: dictionary of dataFrame """
    j = 0
    for i in dic_df:
        for col in dic_df[i].select_dtypes('float'):
            if dic_df[i][col].min()==0:
                j += 1
                index_zero = dic_df[i][col].index[dic_df[i][col]==0]
                dic_df[i] = dic_df[i].drop(index=index_zero)
    if j==0:
        print("There is no zero value")
        
def List_Categorial_Variable(dic_df):
    """ Lists categorical variables.
     Parameter: dictionary of dataFrame """
    for i in dic_df:
        print(10*"=",i,10*"=")
        print(dic_df[i].select_dtypes(['object']).nunique().sort_values(ascending=False))

def Categorial_Variable_Elements(dic_df):
    """ List the elements of categorical features.
     Parameter: dictionary of dataFrame """
    for i in dic_df:
        print(10*"=",i,10*"=")
        for col in dic_df[i].select_dtypes('object'):
            print(f'{col :-<40} {dic_df[i][col].unique()}')

def Unique_Variable(dic_df,dtype):
    """ List the elements of the variables.
     Parameter: dictionary of dataFrame, variable type """
    for col in dic_df.select_dtypes(dtype):
        print(f'{col :-<40} {dic_df[col].unique()}')

def Check_NaNs(df):
    """ List nan elements
     Parameter: dataFrame"""
    print((df.isna().sum()/df.shape[0]).sort_values(ascending=True))
    #print (round((df.isna().sum()/df.shape[0]).sort_values(ascending=True)))

def NaN_index(df):
    """ List indexes of nan elements
     Parameter: dataFrame"""
    for i in df.columns:
        print("=======",i,"=========")
        print(i, df[i].index[df[i].isna()])

def Drop_NaNs(df):
    """ Drope nan elements
     Parameter: dataFrame"""
    for i in df.columns:
        index_Nan = df[i].index[df[i].isna()]
        df.drop(index_Nan, inplace=True)
        
def box_plot(df):
    """ plots the distribution of numerical variables.
     Parameter: dataFrame"""
    for col in df.select_dtypes('float'):
        plt.figure(figsize = (8,1))

        sns.boxplot(x=df[col])
        titre = 'Distribution of : ' + col
        plt.title(titre)
        plt.xlabel(col)
        plt.show()        

def handling_outliers_percentile(df, col, mini, maxi):
    """ Computing minith, maxith percentiles and replacing the outliers.
    Parameter: dataFrame, features, minth and maxth percentiles"""
    min_percentile = np.percentile(df[col], mini)
    max_percentile = np.percentile(df[col], maxi)
    df.loc[df[col]>max_percentile, col] = max_percentile
    df.loc[df[col]<min_percentile, col] = min_percentile

def plot_pie(df, cat_var):
    """ plot the categorical variable elements
     Parameter: 
             - DataFrame
             - list categorical variable"""
    for col in df[cat_var]:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(df[col].value_counts(), labels=df[col].unique(), autopct='%1.1f%%', textprops=dict(size=12))
        ax.set_title(' '.join(col.split('_')).title(), size=14)
        plt.show()
        
def plot_bi_var (variable,target,df):
    """ it plots variable according to another variable
     Parameter: Name of two variable of string type """
    plt.figure(figsize=(10,5))
    ax = sns.boxplot(x=variable, y=target, data=df, width=0.5,showfliers=False, showmeans=True)
    
    plt.xlabel(' '.join(variable.split('_')).title(),size=14)
    plt.ylabel(' '.join(target.split('_')).title(),size=14)
    plt.title("\n Distribution of {} virsus {} \n".format(' '.join(target.split('_')).title(), '   '.join(variable.split('_')).title()),size=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()        
        
def plot_bi_var_bar (variable,target,df):
    """ it plots variable according to another variable
     Parameter: Name of two variable of string type """
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x=variable, y=target, data=df)
    
    plt.xlabel(' '.join(variable.split('_')).title(),size=14)
    plt.ylabel(' '.join(target.split('_')).title(),size=14)
    plt.title("\n Distribution of {} virsus {} \n".format(' '.join(target.split('_')).title(), ' '.join(variable.split('_')).title()),size=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    plt.show()        
    

def Rscore(x,p,d):
    """Function to calculate the Quantile scores for Recency.
    Parameter: dataFrame, Recency feature, variable."""
    if x<= d[p][0.25]:
        return 1
    elif x<= d[p][0.50]:
        return 2
    elif x<= d[p][0.75]:
        return 3
    else:
        return 4
# Function to calculate the Quantile scores for Frequency and Monetary.
def FMscore(x,p,d):
    """Function to calculate the Quantile scores for Recency.
    Parameter: dataFrame, FM features, variable."""
    if x<= d[p][0.25]:
        return 4
    elif x<= d[p][0.50]:
        return 3
    elif x<= d[p][0.75]:
        return 2
    else:
        return 1
        
        
        