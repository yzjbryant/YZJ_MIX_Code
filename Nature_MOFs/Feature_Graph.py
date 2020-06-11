import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import multiprocessing
from ctypes import *
from threading import Thread

data=pd.read_excel('CIF_Parameters.xlsx',header=0)
'''
Feature_Descriptor=data.columns
print(Feature_Descriptor)
'''
Target=['Pure_C6H6_Heat_298K_9.99kpa',
       'Pure_C6H6_Heat_363K_9.99kpa', 'Pure_C6H6_Heat_363K_99.9kpa',
       'Pure_C6H6_Loading_298K_9.99kpa', 'Pure_C6H6_Loading_363K_9.99kpa',
       'Pure_C6H6_Loading_363K_99.9kpa', 'Pure_C4H4S_Heat_298K_0.01kpa',
       'Pure_C4H4S_Heat_363K_0.01kpa', 'Pure_C4H4S_Heat_363K_0.1kpa',
       'Pure_C4H4S_Loading_298K_0.01kpa', 'Pure_C4H4S_Loading_363K_0.01kpa',
       'Pure_C4H4S_Loading_363K_0.1kpa', 'Mix_Heat_C4H4S_298K_10kpa',
       'Mix_Heat_C4H4S_363K_10kpa', 'Mix_Heat_C4H4S_363K_100kpa',
       'Mix_Heat_C6H6_298K_10kpa', 'Mix_Heat_C6H6_363K_10kpa',
       'Mix_Heat_C6H6_363K_100kpa', 'Mix_Loading_C4H4S_298K_10kpa',
       'Mix_Loading_C4H4S_363K_10kpa', 'Mix_Loading_C4H4S_363K_100kpa',
       'Mix_Loading_C6H6_298K_10kpa', 'Mix_Loading_C6H6_363K_10kpa',
       'Mix_Loading_C6H6_363K_100kpa', 'Selectivity_298K_10kpa',
       'Selectivity_363K_10kpa', 'Selectivity_298K_100kpa',
       'Working_capacity_PSA_363K_100kpa_10kpa',
       'Working_capacity_TSA_10kpa_298K_363K']#Number of Target(i) is 29
Feature_Descriptor_1=['Cluster', 'Linker', 'Group', 'Mass', 'Length_a',
       'Length_b', 'Length_c', 'Angle_alpha', 'Angle_beta', 'Angle_gamma',
       'Number of Element', 'Tong', 'Zn', 'C', 'H', 'Br', 'Lu', 'I', 'N', 'O',
       'F', 'S', 'Di', 'Df', 'Dif', 'Unitcell_volume', 'Density', 'ASA_A^2',
       'ASA_m^2/cm^3', 'ASA_m^2/g', 'NASA_A^2', 'NASA_m^2/cm^3', 'NASA_m^2/g',
       'AV_A^3', 'AV_Volume_fraction', 'AV_cm^3/g', 'NAV_A^3',
       'NAV_Volume_fraction', 'NAV_cm^3/g', 'Pure_C6H6_Heat_298K_9.99kpa',
       'Pure_C6H6_Heat_363K_9.99kpa', 'Pure_C6H6_Heat_363K_99.9kpa',
       'Pure_C6H6_Loading_298K_9.99kpa', 'Pure_C6H6_Loading_363K_9.99kpa',
       'Pure_C6H6_Loading_363K_99.9kpa', 'Pure_C4H4S_Heat_298K_0.01kpa',
       'Pure_C4H4S_Heat_363K_0.01kpa', 'Pure_C4H4S_Heat_363K_0.1kpa',
       'Pure_C4H4S_Loading_298K_0.01kpa', 'Pure_C4H4S_Loading_363K_0.01kpa',
       'Pure_C4H4S_Loading_363K_0.1kpa', 'Mix_Heat_C4H4S_298K_10kpa',
       'Mix_Heat_C4H4S_363K_10kpa', 'Mix_Heat_C4H4S_363K_100kpa',
       'Mix_Heat_C6H6_298K_10kpa', 'Mix_Heat_C6H6_363K_10kpa',
       'Mix_Heat_C6H6_363K_100kpa', 'Mix_Loading_C4H4S_298K_10kpa',
       'Mix_Loading_C4H4S_363K_10kpa', 'Mix_Loading_C4H4S_363K_100kpa',
       'Mix_Loading_C6H6_298K_10kpa', 'Mix_Loading_C6H6_363K_10kpa',
       'Mix_Loading_C6H6_363K_100kpa', 'Selectivity_298K_10kpa',
       'Selectivity_363K_10kpa', 'Selectivity_298K_100kpa',
       'Working_capacity_PSA_363K_100kpa_10kpa',
       'Working_capacity_TSA_10kpa_298K_363K'] #Number of Feature_Descriptor is 69  少了'Materials'
data=data.drop(['Materials'],axis=1)
plt.figure(figsize=(10,10))
correlations=data.corr()
correction=abs(correlations)
fig,ax=plt.subplots(figsize=(20,20))
sns.heatmap(correlations,vmax=1.0,center=0,fmt='.2f',square=True,linewidths=.5,annot=False,cbar_kws={"shrink":.70})
plt.savefig('1.png')


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

# ax10=sns.jointplot(x='Selectivity_363K_10kpa',y= 'Pure_C6H6_Heat_298K_9.99kpa',data=data)
# plt.savefig('1.png')
# ax11=sns.jointplot(x='Cluster',y='Group',data=data,kind='hex')
# ax12=sns.jointplot(x='Cluster',y='Group',data=data,kind='reg')
# ax13=sns.jointplot(x='Cluster',y='Group',data=data,kind='kde')
# ax14=sns.boxplot(x='Selectivity_363K_10kpa',y= 'Pure_C6H6_Heat_298K_9.99kpa',data=data)
# plt.savefig('2.png')
# ax15=sns.boxplot(x='Cluster',y='Linker',data=data,hue='Group')
# ax16=sns.violinplot(x='Selectivity_363K_10kpa',y= 'Pure_C6H6_Heat_298K_9.99kpa',data=data)
# plt.savefig('3.png')
# ax17=sns.violinplot(x='Cluster',y='Linker',data=data,hue=['Group','Tong'],split=True)
# ax18=sns.stripplot(x='Selectivity_363K_10kpa',y= 'Pure_C6H6_Heat_298K_9.99kpa',data=data)
# plt.savefig('4.png')
# ax19=sns.stripplot(x='Cluster',y='Linker',data=data,jitter=True,hue='Group',dodge=True)
# ax20=sns.swarmplot(x='Selectivity_363K_10kpa',y= 'Pure_C6H6_Heat_298K_9.99kpa',data=data)
# plt.savefig('5.png')
# ax21=sns.factorplot(x='Cluster',y='Mass',data=data)
# plt.show()



# cores=multiprocessing.cpu_count()
# print(cores)
# pool=multiprocessing.Pool(processes=cores)
# ax22=sns.regplot(x='Cluster',y='Loading_C6H6_298K_10kpa',data=data) #回归图
# pool.apply_async(ax22)
# plt.show()
