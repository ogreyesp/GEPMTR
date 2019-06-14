# GEPMTR: Performing Multi-Target Regression via GEP-based Ensemble Models

Multi-target regression problem comprises the prediction of multiple continuous variables given a common set of input features. In this work, a gene expression programming method for multi-target regression is proposed. It follows the symbolic regression approach, and evolves a population of individuals, where each one represents a complete solution to the problem. Also, three ensemble-based methods are developed to better exploit the inter-target and input-output relationships. The experimental study showed that the proposed approach is effective and attains competitive results.

In this repository we provide the code of GEPMTR method, as well as of three ensemble-based methods EGEPMTR-B, EGEPMTR-OTA, and EGEPMTR-S. GEPMTR has been implemented using Mulan [[Tso11]](#Tso11), and Weka [[Hal09]](#Hal09) libraries.

The distribution in packages of the source code is as follows:
* ensemble: ensemble methods comprising members based on GEPMTR.
* experiments: main classes to execute the different methods.
* gep: code base for the GEPMTR method.
* utils: classes with many utilities.

More information about these algorithms can be find in the following article:
> J. M. Moyano, O. Reyes, H. M. Fardoun, and S. Ventura. "Performing Multi-Target Regression via GEP-based Ensemble Models".


### References

<a name="Hal09"></a>**[Hal09]** M. Hall, E. Frank, G. Holmes, B. Pfahringer, P. Reutemann, and I. H. Witten. (2009). The WEKA data mining software: an update. ACM SIGKDD explorations newsletter, 11(1), 10-18.

<a name="Tso11"></a>**[Tso11]** G. Tsoumakas, E. Spyromitros-Xioufis, J. Vilcek, and I. Vlahavas. (2011). Mulan: A java library for multi-label learning. Journal of Machine Learning Research, 12, 2411-2414.
