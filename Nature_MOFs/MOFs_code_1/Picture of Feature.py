import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import multiprocessing
from ctypes import *
from threading import Thread

data_origin=pd.read_excel('CIF_Parameters.xlsx')
Feature_Descriptor=['Materials','Cluster','Linker','Group','Length_a','Length_b','Length_c','Angle_alpha',
                    'Angle_beta','Angle_gamma','Number of Element','Tong','Zn','C','H','Br','Lu','I','N','O',
                    'F','S','Di','Df','Dif','Unitcell_volume','Density','ASA_A^2','ASA_m^2/cm^3','ASA_m^2/g',
                    'NASA_A^2','NASA_m^2/cm^3','NASA_m^2/g','AV_A^3','AV_Volume_fraction','AV_cm^3/g',
                    'NAV_A^3','NAV_Volume_fraction','NAV_cm^3/g',
                    'Energy_C4H4S_298K_10kpa','Energy_C4H4S_363K_10kpa','Energy_C4H4S_363K_100kpa',
                    'Energy_C6H6_298K_10kpa','Energy_C6H6_363K_10kpa','Energy_C6H6_363K_100kpa',
                    'Loading_C4H4S_298K_10kpa','Loading_C4H4S_363K_10kpa','Loading_C4H4S_363K_100kpa',
                    'Loading_C6H6_298K_10kpa','Loading_C6H6_363K_10kpa','Loading_C6H6_363K_100kpa']

data=data_origin.drop(['Materials'],axis=1)
# correlations=data.corr()
# correction=abs(correlations)
#
# fig,ax=plt.subplots(figsize=(20,20))
# sns.heatmap(correlations,vmax=1.0,center=0,fmt='.2f',square=True,linewidths=.5,
#             annot=True,cbar_kws={"shrink":.70})
# plt.show()



# fig=plt.figure()
# ax=fig.add_subplot(figsize=(20,20))
# ax1=sns.heatmap(correction) #相关性热力图
# ax2=sns.pairplot(data,hue='Loading_C6H6_363K_100kpa',markers=["o","s"]) #相关性图，和某一列的关系

# ax3=sns.clustermap(correlations) #分层相关性热力图
# g=sns.PairGrid(data)
# g.map_diag(sns.distplot)
# g.map_upper(plt.scatter)
# g.map_lower(sns.kdeplot)

#单个属性的分布
# ax4=sns.distplot(data['Cluster'])
# ax5=sns.distplot(data['Cluster'],kde=False)
# ax6=sns.countplot(x='Linker',data=data)
# ax7=sns.countplot(x='Group',data=data)
# ax8=sns.rugplot(data['Cluster'])
# ax9=sns.kdeplot(data['Cluster'],shade=True)
#
# #两两属性的相关性图
# ax10=sns.jointplot(x='Cluster',y='Group',data=data)
# ax11=sns.jointplot(x='Cluster',y='Group',data=data,kind='hex')
# ax12=sns.jointplot(x='Cluster',y='Group',data=data,kind='reg')
# ax13=sns.jointplot(x='Cluster',y='Group',data=data,kind='kde')
# ax14=sns.boxplot(x='Cluster',y='Linker',data=data)
# ax15=sns.boxplot(x='Cluster',y='Linker',data=data,hue='Group')
# ax16=sns.violinplot(x='Cluster',y='Linker',data=data)
# ax17=sns.violinplot(x='Cluster',y='Linker',data=data,hue=['Group','Tong'],split=True)
# ax18=sns.stripplot(x='Cluster',y='Linker',data=data)
# ax19=sns.stripplot(x='Cluster',y='Linker',data=data,jitter=True,hue='Group',dodge=True)
# ax20=sns.swarmplot(x='Cluster',y='Linker',data=data)
ax21=sns.factorplot(x='Cluster',y='Linker',data=data)
plt.show()
# cores=multiprocessing.cpu_count()
# print(cores)
# pool=multiprocessing.Pool(processes=cores)
# ax22=sns.regplot(x='Cluster',y='Loading_C6H6_298K_10kpa',data=data) #回归图
# pool.apply_async(ax22)
# plt.show()
