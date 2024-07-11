import pandas as pd 
from statistics import mean
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import numpy as np


def subset_of_restructured_df(restructured_levels_per_trial_df, column_selection, selection):
    AP5_restructured_levels_per_trial_df  = restructured_levels_per_trial_df.copy()
    AP5_restructured_levels_per_trial_df = AP5_restructured_levels_per_trial_df.loc[AP5_restructured_levels_per_trial_df[column_selection] == selection]
    AP5_restructured_levels_per_trial_df = AP5_restructured_levels_per_trial_df.reset_index(drop=True)
    return AP5_restructured_levels_per_trial_df

def make_example_infusion_plot(all_TrainingLevels):
    
    trials_per_session= []
    for i in range(len(all_TrainingLevels)):
        trials_per_session = trials_per_session + [len(all_TrainingLevels[i])]
    cum_trials_per_session = np.cumsum(trials_per_session)

    #manually define sessions when infusions were done: 
    AP5_sessions = [5, 10]
    Saline_sessions = [8, 12]

    TrainLevelsAll = sum(all_TrainingLevels, [])
    fnt = 24
    #if CurrentAnimal == 'EJT185':
    fig = plt.figure(figsize=(18, 10))
    ax = plt.subplot2grid((5, 3), (1, 0), rowspan =4,colspan =3)   
    ax.set_xlabel('Trials', fontsize = fnt, color = "black")
    ax.plot(TrainLevelsAll,linewidth = 1,color = 'black')
    ax.set_ylabel('Training level',fontsize =fnt, color = 'black')
    ax.tick_params(axis='x', labelsize=fnt)
    ax.tick_params(axis='y', labelsize=fnt)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(True)
    ax.spines.bottom.set_visible(True)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    #ax.set_title(CurrentAnimal)
    ymax = max(TrainLevelsAll)
    ymin = min(TrainLevelsAll)

    mylightorange = "#BF6D73"# "#fff6f2"#"#fff3ea" #"#f7e3d8" #"#e8ad8b" ##f69d6b" #"#fdbf6f"
    mydarkorange =  "#BF6D73"#"#ffdac4" #"#f9d6c0" #"#f7e3d8" #'#fcd2b8'#"#efc8b2" #"#dd8452" #"#ff7f00" #

    mylightblue = "#364D9C"#"#e4f0f7"
    mydarkblue =  "#364D9C" #"#2b7bba" #"#0000FF" #


    ax.set_xlim(1, 3000)
    ax.set_ylim(0.5, 52)

    ax.axvspan(693, 782, color= mylightorange, alpha=0.3)

    ax.axvspan(1551, 1718, color= mylightorange, alpha=0.3) #subtracted 10 from end

    #Saline
    ax.axvspan(1106, 1341, color= mylightblue, alpha=0.3) #subtracted 10 from end
    ax.axvspan(1946, 2247, color= mylightblue, alpha=0.3)

    ax.set_yticks([1, 10, 20, 30, 40 ,50])
    ax.text(2530, 28, "Test session", color = "black", fontsize = fnt)
    ax.text(2460, 22, "Day after infusion", color = "black", fontsize = fnt)
    #ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.text(0.059, 0.272, "Early", color = "black", transform=plt.gcf().transFigure, fontsize = fnt, rotation = 90)
    ax.add_patch(plt.Rectangle((-275,12), 100, 8,facecolor='gainsboro',clip_on=False,linewidth = 0))
    ax.text(0.059, 0.40, "Middle", color = "black", transform=plt.gcf().transFigure, fontsize = fnt, rotation = 90)
    ax.add_patch(plt.Rectangle((-275,20.5), 100, 14,facecolor='gainsboro',clip_on=False,linewidth = 0))
    ax.text(0.059, 0.59, "Late", color = "black", transform=plt.gcf().transFigure, fontsize = fnt, rotation = 90)
    ax.add_patch(plt.Rectangle((-275,35), 100, 15,facecolor='gainsboro',clip_on=False,linewidth = 0))
    ax.text(0.0305, 0.11, 'Naive', color = "gray",  transform=plt.gcf().transFigure, fontsize = fnt)
    ax.text(0.0305, 0.71, 'Expert', color = "gray",  transform=plt.gcf().transFigure, fontsize = fnt)
    arrow_pos = -0.1
    ax.annotate('', xy = (arrow_pos, 0.97), xycoords='axes fraction', xytext=(arrow_pos, 0.04), 
            arrowprops=dict(arrowstyle="->, head_width=0.3, head_length =0.7", color='gray', linewidth = 2))

    for session in Saline_sessions:
        #if CurrentAnimal == 'EJT185':
        infusion_trial = cum_trials_per_session[session - 1]
        infusion_level = all_TrainingLevels[session - 1][-1]
        plt.arrow(infusion_trial, infusion_level + 8, 0, -7.8, length_includes_head = True, width = 13, head_width=40, 
                  head_length = 1.1, color = "#1f78b4") #"#2b7bba")
        ax.text(infusion_trial - 80, infusion_level + 8.5, 'Saline', color = "#1f78b4", fontsize = fnt)


    for infusion_number, session in enumerate(AP5_sessions):
    #         if CurrentAnimal == 'EJT185':
        infusion_trial = cum_trials_per_session[session-1] #i.e. the cumulative number of trials just before the infusion
        infusion_level = all_TrainingLevels[session -1][-1]
        plt.arrow(infusion_trial, infusion_level +8, 0, -7.8, length_includes_head = True, width = 13, head_width=40, 
                  head_length = 1.1, color = "#dd8452")# "darkorange")
        ax.text(infusion_trial - 60, infusion_level + 8.5, 'AP5', color = "#dd8452", fontsize = fnt)
                
    return