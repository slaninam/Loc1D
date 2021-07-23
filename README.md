# Loc1D
Analysis of Bluetooth Received Signal Strength at the inputs of four anchors placed along a single dimension to 
obtain device location.

**Measurement setup**

The portable device (tag) periodically transmits Bluetooth Low Energy (BLE) advertisement packets at BLE channels 37, 38, 39 with fixed transmit power. Four anchors are located in the experimental area, each equipped with four BLE receivers. The measurement focuses on Received Signal Strength, i.e., the received signal level (in dBm) of the advertisement packet at the anchor receivers. It can be assumed that greater distance between the tag and the anchor involves larger signal path attenuation and thus lower RSS. This dependency, however, is not straightforward in real environment.

**Contains**

* Rough measured data for 1D positioning in the 'data' subfolder
* Rough measured data for the 2D positioning in the 'data_new', 'SE7.105', 'SE7.106' subfolders
* Preprocessed and clean data in the 'data_csv' subfolder. For preprocessing, the data_exploration.py script is used.
* Python scripts implementing location classifiers in the root.

**Usage**

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

**Citation**

If your use of these files leads to a scientific publication, please, cite our work:

@article{Polak_Sensors_2021,\
&nbsp;&nbsp;     doi = {10.3390/s21134605},\
&nbsp;&nbsp;     url = {https://doi.org/10.3390/s21134605}, \
&nbsp;&nbsp;     year = {2021},\
&nbsp;&nbsp;     month = {July},\
&nbsp;&nbsp;     publisher = {MDPI},\
&nbsp;&nbsp;     volume = {21},\
&nbsp;&nbsp;     number = {13},\
&nbsp;&nbsp;     pages = {1--25},\
&nbsp;&nbsp;     author = {Ladislav Polak and Stanislav Rozum and Martin Slanina and Tomas Bravenec and Tomas Fryza and Aggelos Pikrakis},\
&nbsp;&nbsp;     title = {Received Signal Strength Fingerprinting-Based Indoor Location Estimation Employing Machine Learning},\
&nbsp;&nbsp;     journal = {Sensors}\
}

