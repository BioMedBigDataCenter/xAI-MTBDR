# xAI-MTBDR: An explainable artificial intelligence framework to predict drug resistance of *Mycobacterium tuberculosis*

Rapid and accurate detection of *Mycobacterium tuberculosis* (MTB) drug resistance and identification of key mutations are essential for controlling transmission and advancing our understanding of resistance mechanisms. While machine learning models offer promising solutions for predicting drug resistance, performance gaps remain for certain drugs, and resistance mechanisms at the level of individual isolates have not yet been considered. Here, we introduce xAI-MTBDR, an explainable artificial intelligence framework designed to deliver accurate and interpretable predictions of drug resistance in MTB. By leveraging whole-genome sequencing data from 39,145 MTB isolates, xAI-MTBDR outperforms state-of-the-art methods for all first-line drugs. Beyond population-level insights, the framework identifies key mutations in individual isolates, discovering 30 potential resistance markers. Additionally, xAI-MTBDR efficiently subgroups drug-resistant isolates with different resistance mechanisms and reflects varying resistance levels. The framework serves as a valuable tool for accurate detection of drug-resistant MTB and offers new insights into its underlying mechanisms.

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

