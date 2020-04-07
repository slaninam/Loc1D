The data provided in this folder correspond to the measurement results in paper:
"Improved Fingerprinting for Indoor Location Estimation Employing Machine Learning"
(arXiv preprint to be added)

There are two types of data files:
* _rough - no pre-filtering has been applied
* _clean - filtering performed: reaplacing -110 dBm values with NaN, merging two consecuting lines within a given spatial coordinate to decrease the probability of a misssing values, remove all the remaining samples in which values are missing.

Radio channel mapping:
* channel 0 = BLE channel 37
* channel 1 = BLE channel 38
* channel 2 = BLE channel 39
