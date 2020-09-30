# Loc1D
Analysis of Bluetooth Received Signal Strength at the inputs of four anchors placed along a single dimension to 
obtain device location.

Contains:
* Rough measured data for 1D positioning in the 'data' subfolder
* Rough measured data for the 2D positioning in the 'data_new', 'SE7.105', 'SE7.106' subfolders
* Preprocessed and clean data in the 'data_csv' subfolder
* Python scripts implementing location classifiers in the root.

In order to train and evaluate a model, run
* modelXd_CLF.py
such that 
* X = 1 or 2, depending on whether localization in one dimension or in a two-dimensional space is desired
* CLF is one of knn (k Nearest Neighbors), svm (Support Vector Machine), mlp (multi layer perceptron), rf (random forest)

The modell will call functions from the modelXd_preprocessing and load csv measurement data from the 'data_csv' subfolder.

At the output, the script will save:
* modelXd_CLF_trained.joblib - a re-usable trained model obained with 5-fold cross validation
* modelXd_CLF_predictions.csv - real and predicted values in the test set
* modelXd_CLF_log.txt - text output to document the training process

Auxiliary scrips are available for visualizations and error calculation:
* errors.py
* error2d.py
* visualization.py
* visualization2d.py
