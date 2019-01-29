# GEPMTR: Performing Multi-Target Regression via GEP-based Ensemble Models

-- *Here include the abstract* --

In this repository we provide the code of GEPMTR base method, as well as of three ensemble-based methods EGEPMTR-B, EGEPMTR-OTA, and EGEPMTR-S. GEPMTR has been implemented using Mulan [[Tso11]](#Tso11), and Weka [[Hal09]](#Hal09) libraries.

The distribution in packages of the sorce code is done as follows:
* ensemble: ensemble methods based on GEPMTR.
* experiments: classes to execute the different methods.
* gep: classes for the base GEPMTR method.
* utils: classes with many utilities.

More information about these algorithms can be find in the following article:
> J. M. Moyano, H. M. Fardoun, and O. Reyes. "Performing Multi-Target Regression via GEP-based Ensemble Models". Submitted to XXX. (2019).


### References

<a name="Hal09"></a>**[Hal09]** M. Hall, E. Frank, G. Holmes, B. Pfahringer, P. Reutemann, and I. H. Witten. (2009). The WEKA data mining software: an update. ACM SIGKDD explorations newsletter, 11(1), 10-18.

<a name="Tso11"></a>**[Tso11]** G. Tsoumakas, E. Spyromitros-Xioufis, J. Vilcek, and I. Vlahavas. (2011). Mulan: A java library for multi-label learning. Journal of Machine Learning Research, 12, 2411-2414.