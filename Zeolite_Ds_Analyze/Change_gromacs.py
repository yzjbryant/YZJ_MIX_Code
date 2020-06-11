#!/usr/bin/python
import os
import sys
import re

name=['AEI','AFI','AFT','AFV','AFX','AFY','ATS','AVL','AWW','BEA','BEC','BOG','CFI','CHA','CON','CSV','DDR','DFO','EAB','EEI','EMT','EON','ERI','ETR','EUO','EZT','FAU','FER','GME','GON','IFO','IFR','IFW','IFY','IHW','IMF','IRR','ISV','ITH','ITR','ITT','IWR','IWW','JSR','KFI','LEV','LTA','LTL','MEI','MEL','MER','MFI','MFS','MOR','MOZ','MRE','MSE','MTW','NES','NPT','OFF','OKO','OSI','PAU','POS','RHO','RTE','RWY','SAF','SAT','SBE','SBS','SBT','SEW','SFE','SFF','SFG','SFH','SFN','SFO','SFS','SFV','SFW','SSF','SSY','STF','STI','STO','STT','TER','TSC','UFI','UOV','USI','VFI']
name2=['1','2','3','4']
name3=['AEI_3_3_2.itp','AFI_3_3_5.itp','AFT_4_4_2.itp','AFV_3_3_3.itp','AFX_3_3_2.itp','AFY_3_3_5.itp','ATS_3_2_7.itp','AVL_3_3_2.itp','AWW_2_2_4.itp','BEA_2_2_4.itp','BEC_2_2_2.itp','BOG_2_2_4.itp','CFI_2_5_1.itp','CHA_3_3_3.itp','CON_2_3_3.itp','CSV_3_3_4.itp','DDR_3_3_1.itp','DFO_2_2_2.itp','EAB_3_3_3.itp','EEI_3_1_2.itp','EMT_2_2_1.itp','EON_8_3_2.itp','ERI_3_3_3.itp','ETR_2_2_5.itp','EUO_3_2_2.itp','EZT_4_3_2.itp','FAU_1_1_1.itp','FER_2_3_6.itp','GME_3_3_4.itp','GON_2_2_7.itp','IFO_3_10_2.itp','IFR_2_3_5.itp','IFW_2_2_3.itp','IFY_2_2_3.itp','IHW_4_2_3.itp','IMF_4_1_3.itp','IRR_2_2_3.itp','ISV_2_2_1.itp','ITH_4_4_2.itp','ITR_4_2_2.itp','ITT_2_2_3.itp','IWR_2_3_3.itp','IWW_3_3_3.itp','JSR_2_2_2.itp','KFI_2_2_2.itp','LEV_3_3_2.itp','LTA_3_3_3.itp','LTL_2_2_5.itp','MEI_3_3_3.itp','MEL_2_2_3.itp','MER_2_2_3.itp','MFI_2_2_3.itp','MFS_6_3_2.itp','MOR_2_2_5.itp','MOZ_1_1_4.itp','MRE_5_3_2.itp','MSE_2_2_2.itp','MTW_2_10_4.itp','NES_2_3_2.itp','NPT_2_2_2.itp','OFF_3_3_5.itp','OKO_2_3_4.itp','OSI_2_2_7.itp','PAU_1_1_1.itp','POS_2_2_3.itp','RHO_2_2_2.itp','RTE_2_2_4.itp','RWY_2_2_2.itp','SAF_2_1_3.itp','SAT_5_5_2.itp','SBE_3_3_2.itp','SBS_3_3_2.itp','SBT_2_2_1.itp','SEW_2_4_3.itp','SFE_4_8_3.itp','SFF_4_2_6.itp','SFG_1_2_2.itp','SFH_13_2_3.itp','SFN_1_5_2.itp','SFO_2_3_6.itp','SFS_3_2_3.itp','SFV_5_5_1.itp','SFW_3_3_1.itp','SSF_2_2_3.itp','SSY_8_2_3.itp','STF_3_2_5.itp','STI_3_2_2.itp','STO_2_6_2.itp','STT_3_2_3.itp','TER_4_2_2.itp','TSC_1_1_1.itp','UFI_2_2_1.itp','UOV_3_2_1.itp','USI_2_3_4.itp','VFI_2_2_4.itp']

for i in name:

    # os.chdir("ds_CO2")
    # os.system("mkdir 'C'")
    # print("M")
    os.chdir("ds_CO2/%s"%i)
    # for j in name2:
    #     path=('./%s'%j)
        # os.system("g_msd_mpi -f gromacs_linux.xtc -s gromacs.tpr")
        # os.system("1")
        # os.system("music2gmx -log /filesdata/music_simu/zeolites/data/material_co/material_co_1/%s_co/temperature_298/100/music_gcmc2.txt -xyz /filesdata/music_simu/zeolites/data/material_co/material_co_1/%s_co/temperature_298/100/finalconfig.xyz -itp yes"%i%i)

    print("A")
    for j in name2:
        path=('./%s'%j)
        with open(os.path.join(path,'gromacs.mdp'),'r+') as f:
            all_lines=f.readlines()
            f.seek(0)
            f.truncate()
            for line in all_lines:
                line=re.sub("AEI",'%s'%i,line)
                f.write(line)
            f.close()
        print("B")
        # with open(os.path.join(path,'gromacs.top'),'r+') as f:
        #     all_lines=f.readlines()
        #     f.seek(0)
        #     f.truncate()
        #     for line in all_lines:
        #         line=re.sub("AEI",'%s'%i,line)
        #         f.write(line)
        #     f.close()
        # print("C")
        with open(os.path.join(path,'gibbs_gromacs_fl.pbs'),'r+') as f:
            all_lines=f.readlines()
            f.seek(0)
            f.truncate()
            for line in all_lines:
                line=re.sub("ppn=24",'ppn=1',line)
                f.write(line)
            f.close()
        print("D")
    rootpath = os.path.dirname(sys.path[0])
    os.chdir(rootpath)
# os.chdir("ds_CO2")
# os.system("mkdir 'C'")
# rootpath = os.path.dirname(sys.path[0])
# os.chdir(rootpath)



