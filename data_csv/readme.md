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
