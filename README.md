# nighttime_place_recognition
It's a midterm project of CS545 conducted by Mingjie Zeng, Shuwen Liu, and Yang Wu.

The project contains three folders:
- analysis pics : contains all data analysis pics in the experiments
- match compare pics : contains the matching result pics for choosing a proper pipeline
- nighttime place recognition : contains all the .py files and the dataset

In the folder "nighttime place recognition":
- feature_analysis.py : this is the program is for experiment 1, analyzing which feature detector can obtain proper and enough features
- test_matching_1.py : this is a test program for the pipeline "GFTTDetector_create + SIFT + BF"
- test_matching_2.py : this is a test program for the pipeline "goodFeaturesToTrack + SIFT + BF"
- test_matrix.py : this is the program for testing the recognition results using confusion matrix
- train.py : this is the program to get the threshold to achieve the night-time place recognition goal
