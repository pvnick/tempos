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

opts = {
    "pain_data.data_file_path": os.path.join("E:\\", "parisa2_pain_encrypted.txt"),
    "pain_data.output_directory": os.path.join("E:\\", "tempos", "figures"),
    "pain_data.tmp_directory": os.path.join("E:\\", "tmp"),
    "pain_data.start_recording_time_minutes_cutoff": 120,
}

def get_opt_val(option, default = ""):
    return opts.get(option, default)

plt.rcParams['mathtext.default'] = 'it' #'regular'
#locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def main(argv):
    utils = PainDataUtils()
    utils.run()

class PainDataUtils:
    def __init__(self):
        self.output_directory = get_opt_val("pain_data.output_directory")
        self.tmp_directory = get_opt_val("pain_data.tmp_directory")
        print(get_opt_val("pain_data.data_file_path"))
        self.pain_data = pd.read_csv(get_opt_val("pain_data.data_file_path"), true_values=["Yes", "True"], false_values=["No", "False"])
        pain_data_columns = self.pain_data.columns.values
        #rename the first column ("id") to something more meaningful
        pain_data_columns[0] = "PatientID"
        self.pain_data.columns = pain_data_columns
        self.pain_data['NRSTimeFromEndSurgery_mins'] = self.pain_data.NRSTimeFromEndSurgery.apply(self.get_mins_from_hhmm)

    @staticmethod
    def get_mins_from_hhmm(hhmm):
        parts = hhmm.split(':')
        return int(parts[0]) * 60 + int(parts[1])

    def run(self):
        self.generate_start_time_figures()
        self.generate_cutoff_start_time_figures()

    #Generate histograms of patient score record start times (normal and log scale)
    def generate_start_time_figures(self):
        recording_time_grouped_by_patient = self.pain_data[["PatientID", "NRSTimeFromEndSurgery_mins"]].groupby("PatientID")
        recording_start_minutes = recording_time_grouped_by_patient.min()

        fig1 = "fig1.pdf"
        fig2 = "fig2.pdf"

        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_minutes.values, bins=20, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig1), bbox_inches="tight")

        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times, log scale", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_minutes.values, bins=20, log=True, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig2), bbox_inches="tight")

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
                            "-output-directory", 
                            self.tmp_directory, 
                            os.path.join(self.tmp_directory, "tmp.tex")])
        shutil.move(os.path.join(self.tmp_directory, "tmp.pdf"), 
                    os.path.join(self.output_directory, "pain_score_start_times.pdf"))

    #Generate histograms of patient score record start times that are 
    #less than pain_data.start_recording_time_minutes_cutoff (normal and log scale)
    def generate_cutoff_start_time_figures(self):
        start_recording_time_minutes_cutoff = get_opt_val("pain_data.start_recording_time_minutes_cutoff", 120)
        recording_time_grouped_by_patient = self.pain_data[["PatientID", "NRSTimeFromEndSurgery_mins"]].groupby("PatientID")
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

        plt.figure(figsize=[8,4])
        plt.title("Pain score recording start times less than " + str(start_recording_time_minutes_cutoff) + "\nminutes postoperation, log scale", fontsize=14).set_y(1.05) 
        plt.ylabel("Occurrences", fontsize=14)
        plt.xlabel("Recording Start Time (minutes)", fontsize=14)
        plt.hist(recording_start_cutoff_patientid_minutes.values, log=True, color="0.5")
        plt.savefig(os.path.join(self.tmp_directory, fig2), bbox_inches="tight")

        cutoff_patientids = recording_start_cutoff_patientid_minutes.index
        start_cutoff_pain_data = self.pain_data.set_index("PatientID").ix[cutoff_patientids].reset_index()

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
                            "-output-directory", 
                            self.tmp_directory, 
                            os.path.join(self.tmp_directory, "tmp.tex")])
        shutil.move(os.path.join(self.tmp_directory, "tmp.pdf"), 
                    os.path.join(self.output_directory, "pain_score_start_times_lte_2hours.pdf"))

if __name__ == "__main__":
    main(sys.argv)