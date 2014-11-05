# file="painSAX.py" Instituition="University of Florida, Intelligent Health Lab"
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

__author__ = "Paul Nickerson, Parisa Rashidi, Patrick Tighe"
__copyright__ = "Copyright 2014, The Pain SAX Project"
__credits__ = ["Eamonn Keogh group (University of California - Riverside)"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Paul Nickerson"
__email__ = "pvnick@ufl.edu"
__status__ = "Experimental"

import subprocess
import shutil
import sys
import locale
import scipy as sc
import numpy as np
import pandas as pd
import os
import math
import string
import re
from pylab import plt
from matplotlib.artist import setp
from matplotlib import gridspec, patches, transforms, ticker
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.diagnostic import kstest_normal as lilliefors

"""
See the PainDataUtils for examples on how to use.
Expected input format (generally a csv file):
Patient ID | Record Time | Value | Other Column(s)
1          | 1           | 4     | Val 1
1          | 75          | 9     | Val 1
1          | 100         | 7     | Val 1
2          | 8           | 1     | Val 2
2          | 19          | 6     | Val 2
2          | 50          | 3     | Val 2
That is, input contains multiple time series, each belonging to an "owner" (in this case, Patient ID).
Time series are defined as a sequence of values given for each record time. Other column(s) are expected 
to be static and are used for stratification, which is specified using a callback function (see the 
Stratification class for examples). All relevant settings are contained within opts, defined below.

Example:
#load csv into a pandas DataFrame
data = load_data() 
#create an instance of the analysis driver, passing the data and specifying a stratification callback function 
driver = Driver(data, stratification_func) 
#run the analysis, generate a set of icons, and render them to an output file
driver.run()
"""

opts = {
    #the following options are general to any data set
    "driver.mathtext.fontstyle": "it",
    "driver.rng_seed": 10, #ensure we get consistent replicatable results
    "driver.bootstrapping.num_samples": 1000, #number of bootstrapped normalization resampling replications
    "driver.bootstrapping.min_stratum_size": 50, #strata with fewer items than this will be discarded
    "driver.sax.alphabet_size": 5, #"beta" from the manuscript, relates to the SAX resolution and is used for icon height/width
    "driver.sax.paa_frame_length": 5 * 60, #5 hours
    "driver.icons.icons_per_row": 3, 
    "driver.icons.icon_set_figure_size": (23, 20), #(width, height) tuple
    "driver.icons.output_file": "icons.pdf",

    #if true, will sort displayed icons in descending cosine similarity
    #otherwise, will sort by stratum name
    "driver.icons.sort_by_similarity": True, 

    #if true, each icon (besides the first one) will be rendered with its similarity
    #to the first displayed icon
    "driver.icons.show_icon_similarities": True, 

    #id for the stratum whose icon will be displayed first, and all other
    #icons will be shown in descending cosine similarity (only applicable
    #when driver.icons.sort_by_similarity is True). This value should be one
    #of the ids returned from the 
    "driver.icons.first_displayed_stratum": "Male, age 21-39", 

    #each unique value for this column is associated with its own time series in the dataset
    "driver.time_series.owner_id_label": "PatientID", 

    "driver.time_series.x_label": "NRSTimeFromEndSurgery_mins", #x axis for the time series
    "driver.time_series.y_label": "PainScoreQuantMissing", #y axis for the time series
    "driver.time_series.interpolation_interval": 10, #step size for time series linear interpolation
    #the following options are specific to the pain data set used to develop the procedure
    "pain_data.data_file_path": os.path.join("E:\\", "parisa2_pain_encrypted_short.csv"),
    "pain_data.output_directory": os.path.join("E:\\", "tempos"),
    "pain_data.tmp_directory": os.path.join("E:\\", "tmp"),
    "pain_data.start_recording_time_minutes_cutoff": 120,
    "pain_data.table1_out_path": "table1.csv",
    "pain_data.figures.start_times_lte_cutoff": "pain_score_start_times_lte_2hours.pdf",
    "pain_data.figures.start_times": "pain_score_start_times.pdf",
    "pain_data.figures.raw_scores_histogram": "raw_scores_histogram.pdf",
    "pain_data.figures.interpolation_demo.interpolation": "interpolation_demo.pdf",
    "pain_data.figures.interpolation_demo.paa": "paa_demo.pdf",
    "pain_data.figures.interpolation_demo.sax": "sax_demo.pdf",
    "pain_data.figures.paa_frame_length_demo": "paa_frame_length_demo.pdf",
}

def get_opt_val(option, default = ""):
    if option not in opts:
        raise Exception("Option '" + option + "' not found")
    return opts[option]

def main(argv):
    np.random.seed(get_opt_val("driver.rng_seed"))
    plt.rcParams['mathtext.default'] = get_opt_val("driver.mathtext.fontstyle")
    utils = PainDataUtils()
    utils.run()

class Driver:
    def __init__(self, ts_df, stratification_callback):
        self.ts_df = ts_df
        self.all_motifs = {}
        self.saxes = []
        self.ts_df_interpolated = None
        self.owner_motif_matrices = []
        self.icons = None
        self.breakpoints = []
        self.stratification_callback = stratification_callback
    
    def run(self):
        print("Interpolating time series to fixed intervals of %d time units" % (get_opt_val("driver.time_series.interpolation_interval")))
        self.interpolate_timeseries()
        print("Calculating quantile breakpoints based on all time series")
        self.define_breakpoints()
        print("Calculating sax words for each owned time series")
        self.define_sax_words()
        print("Defining a motif matrix for each time series")
        self.define_motif_matrices()
        print("Normalizing")
        self.normalize()
        print("Calculating strata cosine similarities")
        self.calculate_strata_similarities()
        print("Saving icon set to " + get_opt_val("driver.icons.output_file"))
        self.save_icons()
        print("Done")
        
    def define_motif_matrices(self): 
        self.owner_motif_matrices = self.saxes.apply(self.get_raw_sequence_motif_vector)

    def define_breakpoints(self):
        def owner_paa(owner_series):
            frame_length = float(get_opt_val("driver.sax.paa_frame_length"))
            interval_minutes = float(get_opt_val("driver.time_series.interpolation_interval"))
            frame_points = frame_length / interval_minutes
            paa_size = np.ceil(owner_series.count() / frame_points) 
            return self.piecewise_aggregate_approximation(owner_series, paa_size)

        ts_y_label = get_opt_val("driver.time_series.y_label")
        ts_owner_label = get_opt_val("driver.time_series.owner_id_label")
        groupedby_owner = self.ts_df_interpolated[ts_y_label].groupby(level = ts_owner_label)
        #get PAAs for all owned time series and concatenate them
        all_paa = groupedby_owner.apply(owner_paa)
        all_paa_series = pd.Series(all_paa.values)
        #use the concatenated PAAs to generate quantile-based breakpoints that are applied to each owned time series
        self.breakpoints = self.get_quantile_breakpoints(all_paa_series)
    
    def define_sax_words(self):
        def owner_sax(owner_series, breakpoints):
            frame_length = float(get_opt_val("driver.sax.paa_frame_length"))
            interval_minutes = float(get_opt_val("driver.time_series.interpolation_interval"))
            frame_points = frame_length / interval_minutes
            paa_size = np.ceil(owner_series.count() / frame_points) 
            return self.sax(pd.Series(owner_series.values), paa_size, breakpoints)

        ts_y_label = get_opt_val("driver.time_series.y_label")
        ts_owner_label = get_opt_val("driver.time_series.owner_id_label")
        groupedby_owner = self.ts_df_interpolated[ts_y_label].groupby(level = ts_owner_label)
        self.saxes = groupedby_owner.apply(owner_sax, breakpoints = self.breakpoints)

    #The following function accepts a pandas dataframe like the following:
    # Patient ID | Record Time | Value | Other Column(s)
    # 1          | 1           | 4     | Val 1
    # 1          | 75          | 9     | Val 1
    # 1          | 100         | 7     | Val 1
    # 2          | 8           | 1     | Val 2
    # 2          | 19          | 6     | Val 2
    # 2          | 50          | 3     | Val 2
    #And returns the following:
    # Patient ID | Record Time | Value | Other Column(s)
    # 1          | 4           | 5.00  | Val 1
    # 1          | 14          | 5.56  | Val 1
    # 1          | 24          | 6.12  | Val 1
    # ...
    # 1          | 84          | 8.28  | Val 1
    # 1          | 94          | 7.48  | Val 1
    # 2          | 8           | 1.00  | Val 2
    # ...
    # 2          | 48          | 3.19  | Val 2
    def interpolate_timeseries(self):
        def interpolate_owner_ts(dataframe_multiindexed):
            interval_minutes = get_opt_val("driver.time_series.interpolation_interval")
            #pandas automatically re-adds this in post-processing. remove it so we dont have a duplicate
            dataframe_multiindexed.index = dataframe_multiindexed.index.droplevel(0)
            if dataframe_multiindexed[ts_y_label].count() <= 1:
                #interpolation requires at least two points
                return dataframe_multiindexed
            dataframe_multiindexed.dropna(subset=[ts_y_label], inplace=True)
            dataframe_unindexed = dataframe_multiindexed.reset_index()
            dataframe_time_indexed = dataframe_unindexed.set_index(ts_x_label).sort_index()
            
            #expand the original data structure by forward-filling feature values through the interpolated values
            first_time_value = dataframe_unindexed[ts_x_label].iloc[0]
            last_time_value = dataframe_unindexed[ts_x_label].iloc[-1]
            constant_time_interval_points = np.arange(first_time_value, last_time_value, interval_minutes)
            total_indexed_by_time = dataframe_unindexed.set_index(ts_x_label)
            total_indexed_by_time_expanded = total_indexed_by_time.reindex(constant_time_interval_points, method='ffill')
            
            #use linear interpolation from scipy
            interpolation_func = interp1d(x=dataframe_unindexed[ts_x_label], 
                                          y=dataframe_unindexed[ts_y_label])
            interpolated_vals = interpolation_func(constant_time_interval_points)
            total_indexed_by_time_expanded[ts_y_label] = [interpolated_val if interpolated_val > 0 else 0 for interpolated_val in interpolated_vals]
            return total_indexed_by_time_expanded

        ts_owner_label = get_opt_val("driver.time_series.owner_id_label")
        ts_x_label = get_opt_val("driver.time_series.x_label")
        ts_y_label = get_opt_val("driver.time_series.y_label")
        ts_indexed = self.ts_df.set_index([ts_owner_label, ts_x_label])
        grouped_by_owner = ts_indexed.groupby(level=0)
        ts_interpolated = grouped_by_owner.apply(interpolate_owner_ts)
        ts_interpolated.dropna(subset=[ts_y_label], inplace=True)
        
        #dont include owners with null time series after interpolation
        owner_ts_lengths = ts_interpolated[ts_y_label].groupby(level=0).count()
        blank_series_owners = owner_ts_lengths[owner_ts_lengths == 0]
        ts_interpolated.drop(blank_series_owners.index, level=0, inplace=True)
        self.ts_df_interpolated = ts_interpolated
    
    def get_raw_sequence_motif_vector(self, sequence):
        l = 2 #motif length, hardcoded for reasons outlined in paper
        a = get_opt_val("driver.sax.alphabet_size")
        n = a ** l #possible motif combinations
        Mp = np.zeros(n) #motif vector
        for motif_pos in range(len(sequence) - l + 1):
            motif = sequence[motif_pos:motif_pos + l] #current motif
            i = 0
            for j in np.arange(l):
                letter = motif[j]
                kj = string.lowercase.index(letter)
                i += kj * a ** (l - j - 1)
            self.all_motifs[i] = motif
            Mp[i] += 1
        return pd.Series(Mp)

    def get_stratified_motif_matrices(self):
        ts_owner_label = get_opt_val("driver.time_series.owner_id_label")
        #stratification_callback assumes static feature values per owner, so just get the owners associated with each stratum
        groupedby_owner = self \
            .ts_df_interpolated \
            .reset_index() \
            .groupby(ts_owner_label, sort = False, as_index=False)
        owner_features = groupedby_owner.head(1).set_index(ts_owner_label, drop = False)
        #strata_designations is pd.Series, [{owner1: stratum_id}, {owner2: stratum_id}, etc]
        strata_designations = owner_features.apply(self.stratification_callback, axis=1)
        strata_joined_with_motifs_matrices = pd.concat([strata_designations.to_frame(name="stratum"), 
                                                        self.owner_motif_matrices], 
                                                       axis=1)
        #motrif_matrices_by_strata is pd.DataFrame, {stratum1: {owner1: motif_matrix1, owner2: motif_matrix2}, stratum2: etc}
        motrif_matrices_by_strata = strata_joined_with_motifs_matrices \
            .reset_index() \
            .dropna() \
            .set_index(["stratum", ts_owner_label]) \
            .sort_index()
        return motrif_matrices_by_strata
    
    def sample_motif_importances(self, sample_size):
        sample_indices = np.random.choice(self.owner_motif_matrices.index, size=sample_size, replace=True)
        sample = self.owner_motif_matrices.ix[sample_indices]
        individual_motif_counts = sample.sum()
        total_motif_quantities = individual_motif_counts.sum(axis=1)
        x = individual_motif_counts / total_motif_quantities
        return x
    
    def get_quantile_breakpoints(self, paa_series):
        alphabet_size = get_opt_val("driver.sax.alphabet_size")
        i_vals = np.arange(0, alphabet_size)
        breakpoints = [
            paa_series.quantile(i / float(alphabet_size))
            for i in i_vals
        ] + [np.inf]
        return breakpoints
    
    def get_cuts(self, paa_series, breakpoints):
        cuts = paa_series.apply(lambda paa_element: np.sum(breakpoints <= paa_element))
        return cuts
    
    def convert_cuts_to_sax_word(self, cuts):
        sax_series = cuts.apply(lambda cut: string.ascii_lowercase[cut - 1])
        sax_str = "".join(sax_series.values)
        return sax_str
    
    def piecewise_aggregate_approximation(self, series, paa_size):
        series = pd.Series(series.values) #disregard indices
        paa_size = int(paa_size)
        series_length = int(series.count())
        #take care of the special case where there is no dimensionality reduction
        if series_length == paa_size:
            paa = series
        else:
            if series_length % paa_size == 0:
                paa = series.reshape([series_length / paa_size, paa_size], order="F").mean(axis=0)
            else:
                temp = pd.DataFrame(series).T.reindex(index = xrange(paa_size), method="ffill")
                expanded_sub_section = temp.values.reshape(paa_size * series_length, order="F")
                paa = expanded_sub_section.reshape([series_length, paa_size], order="F").mean(axis=0)
        return pd.Series(paa)
    
    def sax(self, time_series, paa_length, breakpoints):
        paa_series = self.piecewise_aggregate_approximation(time_series, paa_length)
        cuts = self.get_cuts(paa_series, breakpoints)
        word = self.convert_cuts_to_sax_word(cuts)
        return word
    
    def normalize(self):
        stratified_motif_matrices = self.get_stratified_motif_matrices()
        stratified_motif_matrices_grouped = stratified_motif_matrices.groupby(level=0)
        strata_sample_sizes = stratified_motif_matrices_grouped.count()
        individual_motif_counts_by_stratum = stratified_motif_matrices_grouped.sum()
        total_motif_quantities_by_stratum = individual_motif_counts_by_stratum.sum(axis=1)
        motif_proportions_strata = individual_motif_counts_by_stratum.apply(lambda val: val / total_motif_quantities_by_stratum)
        stratification_criteria = motif_proportions_strata.index
        icons = pd.DataFrame(columns = self.owner_motif_matrices.columns, index = pd.Index(data=[], name="stratum_id"))
        for criterion in stratification_criteria:
            motif_proportions_stratum = motif_proportions_strata.ix[criterion]
            stratum_size = strata_sample_sizes.ix[criterion][0]
            min_stratum_size = get_opt_val("driver.bootstrapping.min_stratum_size")
            if stratum_size < min_stratum_size:
                print("Skipping stratum '%s' (size=%d which is less than the minimum, %d)" % (criterion, stratum_size, min_stratum_size))
            else:
                print("Generating Xci for stratum '%s' (size=%d)" % (criterion, stratum_size))
                num_samples = get_opt_val("driver.bootstrapping.num_samples")
                sample_accumulator = []
                for i in range(num_samples):
                    sample_accumulator.append(self.sample_motif_importances(stratum_size))
                motif_importance_sample = pd.DataFrame(sample_accumulator)
                numerators = motif_importance_sample.apply(lambda row: row <= motif_proportions_stratum, axis=1).sum()
                denominators = motif_importance_sample.count()
                CDF = numerators / denominators
                icon = CDF
                icons.ix[str(criterion)] = icon
        self.icons = icons
        
    def calculate_strata_similarities(self):
        def get_similarities(icon, all_icons):
            similarities = cosine_similarity(X=icon, Y=all_icons)[0]
            return pd.DataFrame(data=similarities, 
                                index=pd.Index(all_icons.index, name="other_item"),
                                columns=pd.Index(["similarity"]))

        self.strata_similarities = self.icons.groupby(level=0).apply(get_similarities, all_icons=self.icons)
        
    def save_icons(self):
        first_listed_stratum_id = get_opt_val("driver.icons.first_displayed_stratum")
        sort_by_similarity = get_opt_val("driver.icons.sort_by_similarity")        
        num_cols = get_opt_val("driver.icons.icons_per_row")
        figsize = get_opt_val("driver.icons.icon_set_figure_size")
        show_icon_similarities = get_opt_val("driver.icons.show_icon_similarities")
        figure_path = os.path.join(get_opt_val("pain_data.output_directory"), 
                                   get_opt_val("driver.icons.output_file"))

        stratum_similarities = self.strata_similarities.ix[first_listed_stratum_id]
        if sort_by_similarity:
            strata_sorted = stratum_similarities.sort(columns="similarity", ascending=False).index
        else:
            #sort lexicographically by stratum id
            strata_sorted = stratum_similarities.sort_index().index
        num_icons = float(len(self.icons.index))
        fig, axes = plt.subplots(ncols=num_cols,
                                 nrows=np.ceil(float(num_icons) / num_cols).astype(int), 
                                 figsize=figsize)
        axes_flattened = axes.flatten()
        plt.hold(True)
        axis_index = 0
        for stratum_id in strata_sorted:
            ax = axes_flattened[axis_index]
            icon = self.icons.ix[stratum_id]
            icon_side = np.sqrt(icon.count()).astype(int)
            #hide distracting axis ticks
            ax.tick_params(labelbottom='off',
                           labelleft='off')
            ax.set_title(label="$" + string.replace(stratum_id, ' ', '\ ') + "$", fontsize=38, y=1.04)
            similarity = stratum_similarities.ix[stratum_id].similarity
            if stratum_id != first_listed_stratum_id and show_icon_similarities:
                ax.set_xlabel(xlabel=r"$Similarity = " + ('%.3f' % similarity) + "$", \
                              fontsize=38, labelpad = 20)
            else:
                #placeholder label so icon sets can be neatly shown next to each other
                ax.set_xlabel(xlabel=r"a", color="white", fontsize=38, labelpad = 20)
            pixel_data = icon.reshape([icon_side, icon_side]).astype(float)
            heatmap = ax.pcolor(pixel_data, cmap = plt.cool())
            ax.invert_yaxis()
            cbar = fig.colorbar(heatmap, ax=ax)
            cbar.ax.tick_params(labelsize=32)
            cbar.set_ticks(np.linspace(0, 1, 5))
            axis_index = axis_index + 1
        for remaining_axis_index in range(axis_index, len(axes_flattened)):
            ax = axes_flattened[remaining_axis_index]
            fig.delaxes(ax)
        plt.tight_layout(pad=1.6)
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)


class Stratification:
    #owners/patients with the same stratification id will be assigned to the same stratum
    #this class contains sample methods which are used to generate stratification ids
    #the method used (designated during Driver initialization) takes 1 row for each timeseries owner (eg Patient)
    #and generates a stratification id based on its feature values
        
    @staticmethod
    def agegroup_gender(row):
        #stratify by a combination of gender and age group
        gender_feature_val = row["Gender"]
        age_group_feature_val = row["Age_Group"]
        if (gender_feature_val is None or age_group_feature_val is None):
            #missing feature value - indicate that this row is to be discarded
            return None
        gender_str = "Male" if gender_feature_val == "MALE" else "Female"
        ret_val = "%s, age %s" % (gender_str, age_group_feature_val)
        return ret_val

class PainDataUtils:
    def __init__(self):
        self.output_directory = get_opt_val("pain_data.output_directory")
        self.tmp_directory = get_opt_val("pain_data.tmp_directory")
        
    def load_data(self):
        def get_mins_from_hhmm(hhmm):
            parts = hhmm.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        self.all_pain_data = pd.read_csv(get_opt_val("pain_data.data_file_path"), true_values=["Yes", "True"], false_values=["No", "False"])
        pain_data_columns = self.all_pain_data.columns.values
        #rename the first column ("id") to something more meaningful
        pain_data_columns[0] = "PatientID"
        self.all_pain_data.columns = pain_data_columns
        #provided timestamp is hh:mm. convert to scalar
        self.all_pain_data["NRSTimeFromEndSurgery_mins"] = self.all_pain_data.NRSTimeFromEndSurgery.apply(get_mins_from_hhmm)
        
    def run(self):
        print("Executing routines associated with the pain score data set")
        self.load_data()
        self.generate_start_time_figures()
        self.pain_data_start_cutoff = self.generate_cutoff_start_time_figures()
        self.normality_check()
        print("Running icon driver")
        self.driver = Driver(self.pain_data_start_cutoff, Stratification.agegroup_gender)
        self.driver.run()
        print("Running auxillary stuff")
        self.gen_table1()
        self.pain_score_histogram()
        self.demonstrate_interpolation_procedure()
        self.demonstrate_paa_frame_length_determination()
        print("Done")

    def normality_check(self):
        print("Lilliefors test for normality")
        #all scores
        normality_test = lilliefors(self.pain_data_start_cutoff.PainScoreQuantMissing.dropna().values)
        normality_test_pval = normality_test[1]
        print("Null hypothesis is that all pain scores follow a Gaussian distribution")
        print("P-value for rejecting the null hypothesis = " + str(normality_test_pval))
        #only scores greater than 0
        normality_test = lilliefors(self.pain_data_start_cutoff[self.pain_data_start_cutoff.PainScoreQuantMissing > 0].PainScoreQuantMissing.dropna().values)
        normality_test_pval = normality_test[1]
        print("Null hypothesis is that pain scores greater than 0 follow a Gaussian distribution")
        print("P-value for rejecting the null hypothesis = " + str(normality_test_pval))
    
    #Generate histograms of patient score record start times (normal and log scale)
    def generate_start_time_figures(self):            
        figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.start_times"))
        print("Generating histograms for record start times and saving to " + figure_path)
        recording_time_grouped_by_patient = self.all_pain_data[["PatientID", "NRSTimeFromEndSurgery_mins"]].groupby("PatientID")
        recording_start_minutes = recording_time_grouped_by_patient.min()

        fig1 = "fig1.pdf"
        fig2 = "fig2.pdf"

        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_minutes.values, bins=20, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig1), bbox_inches="tight")
        plt.close()

        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times, log scale", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_minutes.values, bins=20, log=True, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig2), bbox_inches="tight")
        plt.close()

        #save the figures in panel format
        f = open(os.path.join(self.tmp_directory, "tmp.tex"), 'w')
        f.write(r"""
            \documentclass[%
            ,float=false % this is the new default and can be left away.
            ,preview=true
            ,class=scrartcl
            ,fontsize=20pt
            ]{standalone}
            \usepackage[active,tightpage]{preview}
            \usepackage{varwidth}
            \usepackage{graphicx}
            \usepackage[justification=centering]{caption}
            \usepackage{subcaption}
            \usepackage[caption=false,font=footnotesize]{subfig}
            \renewcommand{\thesubfigure}{\Alph{subfigure}}
            \begin{document}
            \begin{preview}
            \begin{figure}[h]
                \begin{subfigure}{0.5\textwidth}
                        \includegraphics[width=\textwidth]{""" + fig1 + r"""}
                        \caption{Normal scale}
                \end{subfigure}\begin{subfigure}{0.5\textwidth}
                        \includegraphics[width=\textwidth]{""" + fig2 + r"""}
                        \caption{Log scale}
                \end{subfigure}
            \end{figure}
            \end{preview}
            \end{document}
        """)
        f.close()
        subprocess.call(["pdflatex", 
                            "-halt-on-error", 
                            "tmp.tex"],
                        cwd = self.tmp_directory)
        shutil.move(os.path.join(self.tmp_directory, "tmp.pdf"), figure_path)

    #Generate histograms of patient score record start times that are 
    #less than pain_data.start_recording_time_minutes_cutoff (normal and log scale)
    def generate_cutoff_start_time_figures(self):
        start_recording_time_minutes_cutoff = get_opt_val("pain_data.start_recording_time_minutes_cutoff")
        figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.start_times_lte_cutoff"))
        print("Generating histograms for record start times which are less than the cutoff, %s, and saving to %s" %
              (start_recording_time_minutes_cutoff, figure_path))
        recording_time_grouped_by_patient = self.all_pain_data[["PatientID", "NRSTimeFromEndSurgery_mins"]].groupby("PatientID")
        recording_start_minutes = recording_time_grouped_by_patient.min()
        recording_start_cutoff_patientid_minutes = recording_start_minutes[recording_start_minutes <= start_recording_time_minutes_cutoff].dropna()

        fig1 = "fig1.pdf"
        fig2 = "fig2.pdf"

        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times less than " + str(start_recording_time_minutes_cutoff) + "\nminutes postoperation", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_cutoff_patientid_minutes.values, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig1), bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times less than " + str(start_recording_time_minutes_cutoff) + "\nminutes postoperation, log scale", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_cutoff_patientid_minutes.values, log=True, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig2), bbox_inches="tight")
        plt.close()
        
        cutoff_patientids = recording_start_cutoff_patientid_minutes.index
        start_cutoff_pain_data = self.all_pain_data.set_index("PatientID").ix[cutoff_patientids].reset_index()

        #save the figures in panel format

        f = open(os.path.join(self.tmp_directory, "tmp.tex"), 'w')
        f.write(r"""
            \documentclass[%
            ,float=false % this is the new default and can be left away.
            ,preview=true
            ,class=scrartcl
            ,fontsize=20pt
            ]{standalone}
            \usepackage[active,tightpage]{preview}
            \usepackage{varwidth}
            \usepackage{graphicx}
            \usepackage[justification=centering]{caption}
            \usepackage{subcaption}
            \usepackage[caption=false,font=footnotesize]{subfig}
            \renewcommand{\thesubfigure}{\Alph{subfigure}}
            \begin{document}
            \begin{preview}
            \begin{figure}[h]
                \begin{subfigure}{0.5\textwidth}
                        \includegraphics[width=\textwidth]{""" + fig1 + r"""}
                        \caption{Normal scale}
                \end{subfigure}\begin{subfigure}{0.5\textwidth}
                        \includegraphics[width=\textwidth]{""" + fig2 + r"""}
                        \caption{Log scale}
                \end{subfigure}
            \end{figure}
            \end{preview}
            \end{document}
        """)
        f.close()
        subprocess.call(["pdflatex", 
                            "-halt-on-error", 
                            "tmp.tex"],
                        cwd = self.tmp_directory)
        shutil.move(os.path.join(self.tmp_directory, "tmp.pdf"), figure_path)
        return start_cutoff_pain_data
    
    def pain_score_histogram(self):
        figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.raw_scores_histogram"))
        print("Generating pain score histogram and saving to " + figure_path)
        fig = plt.figure(figsize=(7,4))
        self.pain_data_start_cutoff.reset_index().PainScoreQuantMissing.hist(bins=11, color="0.5")
        plt.xlabel("Reported pain score")
        plt.ylabel("Occurences")
        plt.title("Occurences of reported pain scores in data file", y=1.03)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig(figure_path)
        plt.close()
     
    def gen_table1(self):
        table1_path = os.path.join(self.output_directory, get_opt_val("pain_data.table1_out_path"))
        print("Generating table1 and saving to " + table1_path)
        
        interpolated_by_patient = self.driver.ts_df_interpolated.reset_index().set_index("PatientID")
        patients_with_key = self.driver.ts_df[["PatientID", "Gender", "Age_Group"]].drop_duplicates()
        counts = patients_with_key.groupby(["Gender", "Age_Group"]).count()["PatientID"]
        table1 = pd.DataFrame(columns=["Variable", "Level1", "Level2", "Value", "Average Pain Score"]).set_index(["Variable", "Level1", "Level2"])

        def comma(integer):
            return "{:,}".format(integer)

        def pct(count, total):
            return str(round(100 * float(count) / total, 1))

        def truncflt(flt):
            return "%.2f" % flt
        
        total_patients = self.all_pain_data.PatientID.drop_duplicates().count()
        print("Number of patients", comma(total_patients))

        total_nonsleep_pain_scores = self.all_pain_data.PainScoreQuantMissing.dropna().count()
        print("Total nonsleep pain scores - ", comma(total_nonsleep_pain_scores))

        patients_before_2_hours = self.driver.ts_df.PatientID.drop_duplicates()
        print("Patients with scores before 2 hours - ", comma(patients_before_2_hours.count()))

        pain_scores_for_patients_before_2_hours = self.driver.ts_df.PainScoreQuantMissing.dropna().count()
        print("Scores for patients scores before 2 hours - ", comma(pain_scores_for_patients_before_2_hours))

        interpolated_pain_scores = self.driver.ts_df_interpolated.PainScoreQuantMissing.dropna().count()
        print("Interpolated scores - ", comma(interpolated_pain_scores))


        #-------- Age group -------------
        key = "Age_Group"
        variable = "Age"
        patient_with_age = self.driver.ts_df[["PatientID", "Age"]].drop_duplicates()
        age_mean = patient_with_age.Age.mean().round(1)
        age_stddev = patient_with_age.Age.std().round(1)

        table1.loc[(variable, "", ""), "Value"] = str(age_mean) + "+/-" + str(age_stddev) + r" (mean +/- SD)"
        table1.loc[(variable, "", ""), "Average Pain Score"] = ""

        patients_with_key = self.driver.ts_df[["PatientID", key]].drop_duplicates()
        level_counts = patients_with_key.groupby(key).count()["PatientID"]
        for level, count in level_counts.to_dict().iteritems():
            table1.loc[(variable, level, ""), "Value"] = comma(count) + " (" + pct(count, level_counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[patients_with_key[key] == level].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, level, ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- Age group coclustered with opioid -------------
        variable = "Age co-clustered with Opioid"
        patients_with_key = self.driver.ts_df[["PatientID", "Age_Group", "Opioid"]].drop_duplicates()
        counts = patients_with_key.groupby(["Age_Group", "Opioid"]).count()["PatientID"]
        for levels, count in counts.to_dict().iteritems():
            age_group = levels[0]
            opioid = ("Taking" if levels[1] else "Not taking") + " Opioid"
            table1.loc[(variable, age_group, opioid), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[(patients_with_key.Age_Group == age_group) & (patients_with_key.Opioid == levels[1])].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, age_group, opioid), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- Gender -------------
        variable = "Gender"
        key = "Gender"
        patients_with_key = self.driver.ts_df[["PatientID", key]].drop_duplicates()
        level_counts = patients_with_key.groupby(key).count()["PatientID"]
        for level, count in level_counts.to_dict().iteritems():
            table1.loc[(variable, level, ""), "Value"] = comma(count) + " (" + pct(count, level_counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[patients_with_key[key] == level].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, level, ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- Gender co-clustered with age group -------------
        variable = "Gender co-clustered with Age"
        patients_with_key = self.driver.ts_df[["PatientID", "Gender", "Age_Group"]].drop_duplicates()
        counts = patients_with_key.groupby(["Gender", "Age_Group"]).count()["PatientID"]
        for levels, count in counts.to_dict().iteritems():
            gender = levels[0]
            age_group = levels[1]
            table1.loc[(variable, gender, age_group), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[(patients_with_key.Gender == gender) & (patients_with_key.Age_Group == age_group)].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, gender, age_group), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- Gender co-clustered with SSRI -------------
        variable = "Gender co-clustered with SSRI"
        patients_with_key = self.driver.ts_df[["PatientID", "Gender", "SSRI"]].drop_duplicates()
        counts = patients_with_key.groupby(["Gender", "SSRI"]).count()["PatientID"]
        count = counts.ix["FEMALE", True]
        table1.loc[(variable, "Female", "Taking SSRI"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "FEMALE") & (patients_with_key.SSRI == True)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Female", "Taking SSRI"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix["MALE", True]
        table1.loc[(variable, "Male", "Taking SSRI"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "MALE") & (patients_with_key.SSRI == True)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Male", "Taking SSRI"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix["FEMALE", False]
        table1.loc[(variable, "Female", "Not taking SSRI"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "FEMALE") & (patients_with_key.SSRI == False)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Female", "Not taking SSRI"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix["MALE", False]
        table1.loc[(variable, "Male", "Not taking SSRI"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "MALE") & (patients_with_key.SSRI == False)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Male", "Not taking SSRI"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)


        #-------- Gender co-clustered with Opioid -------------
        variable = "Gender co-clustered with Opioid"
        patients_with_key = self.driver.ts_df[["PatientID", "Gender", "Opioid"]].drop_duplicates()
        counts = patients_with_key.groupby(["Gender", "Opioid"]).count()["PatientID"]
        count = counts.ix["FEMALE", True]
        table1.loc[(variable, "Female", "Taking Opioid"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "FEMALE") & (patients_with_key.Opioid == True)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Female", "Taking Opioid"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix["MALE", True]
        table1.loc[(variable, "Male", "Taking Opioid"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "MALE") & (patients_with_key.Opioid == True)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Male", "Taking Opioid"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix["FEMALE", False]
        table1.loc[(variable, "Female", "Not taking Opioid"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "FEMALE") & (patients_with_key.Opioid == False)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Female", "Not taking Opioid"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix["MALE", False]
        table1.loc[(variable, "Male", "Not taking Opioid"), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[(patients_with_key.Gender == "MALE") & (patients_with_key.Opioid == False)].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Male", "Not taking Opioid"), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- Opioid -------------
        variable = "Opioid"
        patients_with_key = self.driver.ts_df[["PatientID", "Opioid"]].drop_duplicates()
        counts = patients_with_key.groupby(["Opioid"]).count()["PatientID"]
        count = counts.ix[True]
        table1.loc[(variable, "Taking Opioid", ""), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[patients_with_key.Opioid == True].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Taking Opioid", ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        count = counts.ix[False]
        table1.loc[(variable, "Not taking Opioid", ""), "Value"] = comma(count) + " (" + pct(count, counts.sum()) + "%)"
        patients_in_cluster = patients_with_key[patients_with_key.Opioid == False].PatientID.values
        pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
        pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
        table1.loc[(variable, "Not taking Opioid", ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- surgery type -------------
        variable = "CPT anatomic groupings"
        key = "PrimaryCPTCodeCategory2"
        patients_with_key = self.driver.ts_df[["PatientID", key]].drop_duplicates()
        level_counts = patients_with_key.groupby(key).count()["PatientID"]
        for level, count in level_counts.to_dict().iteritems():
            table1.loc[(variable, level, ""), "Value"] = comma(count) + " (" + pct(count, level_counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[patients_with_key[key] == level].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, level, ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- comobidities -------------
        variable = "Total number of ICD9 coded comorbidities"
        key = "ICD9_Comorbidity_Count_Group"
        patients_with_key = self.driver.ts_df[["PatientID", key]].drop_duplicates()
        level_counts = patients_with_key.groupby(key).count()["PatientID"]
        for level, count in level_counts.to_dict().iteritems():
            table1.loc[(variable, level, ""), "Value"] = comma(count) + " (" + pct(count, level_counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[patients_with_key[key] == level].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, level, ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        #-------- Procedure count -------------
        variable = "Total number of CPT coded procedures"
        key = "Total_CPT_Count_Group"
        patients_with_key = self.driver.ts_df[["PatientID", key]].drop_duplicates()
        level_counts = patients_with_key.groupby(key).count()["PatientID"]
        for level, count in level_counts.to_dict().iteritems():
            table1.loc[(variable, level, ""), "Value"] = comma(count) + " (" + pct(count, level_counts.sum()) + "%)"
            patients_in_cluster = patients_with_key[patients_with_key[key] == level].PatientID.values
            pain_avg = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.mean()
            pain_sd = interpolated_by_patient.ix[patients_in_cluster].PainScoreQuantMissing.std()
            table1.loc[(variable, level, ""), "Average Pain Score"] = truncflt(pain_avg) + "+/-" + truncflt(pain_sd)

        table1.to_csv("/tmp/table1.csv")
        
    def demonstrate_interpolation_procedure(self):
        interp_figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.interpolation_demo.interpolation"))
        paa_figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.interpolation_demo.paa"))
        sax_figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.interpolation_demo.sax"))
        sample_data = [
            [1, 3, 6],
            [1, 24, 8],
            [1, 55, 9],
            [1, 78, 5],
            [1, 115, 4],
            [1, 135, 1],
            [1, 160, 2],
            [1, 194, 1],
            [1, 215, 6],
            [1, 255, 9]
        ]
        sample_data_df = pd.DataFrame.from_records(sample_data, 
                                                   columns = ["PatientID", 
                                                              "NRSTimeFromEndSurgery_mins", 
                                                              "PainScoreQuantMissing"]
                                                   )
        tmp_driver = Driver(sample_data_df, lambda r: "dummy")
        tmp_driver.interpolate_timeseries()

        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_axes([0.15, 0.1, 0.78, 0.8])
        plt.hold(True)
        
        print("Generating interpolation demonstration figure and saving to " + interp_figure_path)
        constant_time_interval_points = np.arange(sample_data_df.NRSTimeFromEndSurgery_mins.min(), 
                                                  sample_data_df.NRSTimeFromEndSurgery_mins.max(), 
                                                  get_opt_val("driver.time_series.interpolation_interval"))
        all_relevant_points = np.union1d(sample_data_df.NRSTimeFromEndSurgery_mins, constant_time_interval_points)
        p1, = ax.plot(pd.Series(sample_data_df.NRSTimeFromEndSurgery_mins) - sample_data_df.NRSTimeFromEndSurgery_mins[0], 
                      sample_data_df.PainScoreQuantMissing.values,
                      'ko', 
                      figure = fig, 
                      label = 'Recorded')
        p2, = ax.plot(constant_time_interval_points - sample_data_df.NRSTimeFromEndSurgery_mins[0], 
                      tmp_driver.ts_df_interpolated.values, 
                      "kx", 
                      figure = fig, 
                      label = 'Interpolated')
        interp_func = interp1d(sample_data_df.NRSTimeFromEndSurgery_mins, sample_data_df.PainScoreQuantMissing)
        p3, = ax.plot(all_relevant_points - sample_data_df.NRSTimeFromEndSurgery_mins[0], 
                      interp_func(all_relevant_points), 
                      "k:", 
                      figure = fig, 
                      label = 'interpolation_line')
        p4, = ax.plot([], [], "k-", figure=fig, label='paa_lines')
        p5, = ax.plot([], [], "k--", figure=fig, label='sax_lines')
        ax.legend([p1, p2], ["Recorded", "Interpolated"], loc='upper center', fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_xlim(0, 260)
        ax.set_title("Sample Pain Scores", fontsize=18, y=1.03)
        ax.set_xlabel("$t$", fontsize=18)
        ax.set_ylabel("$pain\ score$", fontsize=18)
        fig.savefig(interp_figure_path, bbox_inches="tight")

        print("Generating PAA demonstration figure and saving to " + paa_figure_path)
        paa_vals = tmp_driver.piecewise_aggregate_approximation(tmp_driver.ts_df_interpolated.PainScoreQuantMissing, 5)
        ax.hlines(paa_vals[0], 0, 50, linewidth=2)
        ax.hlines(paa_vals[1], 50, 100, linewidth=2)
        ax.hlines(paa_vals[2], 100, 150, linewidth=2)
        ax.hlines(paa_vals[3], 150, 200, linewidth=2)
        ax.hlines(paa_vals[4], 200, 250, linewidth=2)
        ax.legend([p1, p2, p4], ["Recorded", "Interpolated", "PAA"], loc='upper center', fontsize=12)
        fig.savefig(paa_figure_path, bbox_inches="tight")
        print("PAA=" + str(paa_vals.values))

        print("Generating SAX demonstration figure and saving to " + sax_figure_path)
        #the following are quantile breakpoints generated from the pain data set
        beta_3_breakpoints = [0.0, 2.0, 5.09, np.inf]
        ax.hlines(beta_3_breakpoints[1], 0, 260, linewidth=1, linestyles="--")
        ax.hlines(beta_3_breakpoints[2], 0, 260, linewidth=1, linestyles="--")
        ax.text(-0.17, (beta_3_breakpoints[2] + 10) / 20 - 0.03, "c", transform=ax.transAxes, fontsize=24,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), zorder=100)
        ax.text(-0.17, (beta_3_breakpoints[1] + beta_3_breakpoints[2]) / 20 - 0.03, "b", transform=ax.transAxes, fontsize=24,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), zorder=100)
        ax.text(-0.17, beta_3_breakpoints[1] / 20 - 0.03, "a", transform=ax.transAxes, fontsize=24,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), zorder=100)
        #hack to avoid cliping the left margin
        ax.text(-0.2, (1./3. / 2) - 0.03, "a", transform=ax.transAxes, color="white", zorder=1)
        ax.legend([p1, p2, p4, p5], ["Recorded", "Interpolated", "PAA", "Breakpoint"], loc='upper center', fontsize=12)
        fig.savefig(sax_figure_path, bbox_inches="tight")
        plt.hold(False)
        plt.close(fig)
        
    def demonstrate_paa_frame_length_determination(self):
        figure_path = os.path.join(self.output_directory, get_opt_val("pain_data.figures.paa_frame_length_demo"))
        print("Generating figure demonstrating PAA frame length calculation and saving to " + figure_path)
        datapoint_counts = self.driver.ts_df_interpolated \
                    .reset_index()[["PatientID", "NRSTimeFromEndSurgery_mins"]] \
                    .set_index("PatientID") \
                    .groupby(level=0) \
                    .count()
        datapoint_counts.columns = ["number_of_points"]
        short_recovery_patients = datapoint_counts[datapoint_counts.number_of_points <= 30].index.values
        datapoint_counts.drop(short_recovery_patients, inplace=True)
        d = datapoint_counts.number_of_points - 1
        w = np.ceil(d / 30.)
        window_length = d / w  * 10. / 60.

        fig = plt.figure(figsize = (10, 4))
        plt.plot(datapoint_counts.values, window_length.values, 'k.')
        plt.ylabel("Observed frame duration (hours)", fontsize = 14)
        plt.xlabel("$d$", fontsize = 18)
        plt.title("Observed PAA frame duration versus interpolated score dimensions", fontsize = 14, y = 1.03) 
        plt.ylim((0, 5))
        plt.savefig(figure_path, bbox_inches="tight")

if __name__ == "__main__":
    main(sys.argv)