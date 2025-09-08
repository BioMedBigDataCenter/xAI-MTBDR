# xAI-MTBDR: An explainable artificial intelligence framework reveals mutations associated with drug resistance in *Mycobacterium tuberculosis*

Understanding the mechanisms of drug resistance in Mycobacterium tuberculosis (MTB) is essential for the rapid detection of resistance and for guiding effective treatment, ultimately contributing to reducing the global burden of tuberculosis (TB). Under anti-TB drugs pressure, MTB continues to accumulate resistance loci. The current repertoire of known resistance-associated mutations requires further refinement, necessitating efficient methods for the timely identification of potential resistance sites. Here, we introduce xAI-MTBDR, an explainable artificial intelligence framework designed to identify potential resistance-associated mutations and predict drug resistance in MTB. It outperforms state-of-the-art methods in predicting drug resistance for all first-line drugs, and scoring each mutation’s contribution to resistance. By leveraging public whole-genome sequencing data from nearly 40,000 MTB isolates, the framework identified 788 candidate resistance-related mutations and revealed 27 potential resistance markers, several of which are positioned closer to their respective drugs in protein structures than known resistance mutations, suggesting a potentially more direct role in mediating resistance. Furthermore, these scores enabled the framework to efficiently subgroup isolates with different resistance mechanisms and reflect varying levels of resistance. The framework serves as a valuable tool for accurate detection of drug-resistant MTB and offers new insights into its underlying mechanisms.

## Hardware
A linux (CentOS) server with a Intel Xeon CPU.

## Dependencies
+ scikit-learn==0.24.2
+ numpy==1.19.5
+ pandas==0.22.0
+ xgboost==1.5.2
+ shap==0.41.0

## File Description
+ To run the code, clone this repository and add the folder to your python path.
+ Leave-One-Out (LOO) strategy is employed to train and test the ensemble model for each drug. When training the models, each dataset was iteratively selected as the test dataset, while the other two datasets were used as the training datasets. For instance, in one iteration, the 'GenTB' and 'Walker et al.' datasets served as the training set, while the 'CRyPTIC' dataset was used as the test set. Therefore, three datasets are provided in the `data` directory.
  
`ensemble_model.py` predicts drug resistance for 11 drugs using the leave-one-out (LOO) strategy
<br>
`explanation.py` explains the model using rifampicin as an example



## About Us
Bio-Med Big Data Center, CAS Key Laboratory of Computational Biology, Shanghai Institute of Nutrition and Health, Chinese Academy of Sciences.

