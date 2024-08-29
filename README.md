# xAI-MTBDR: An explainable artificial intelligence framework to predict drug resistance of *Mycobacterium tuberculosis*

Rapid determination of *Mycobacterium tuberculosis* (MTB) drug resistance, and identification of key resistance-contributing mutations are crucial for preventing ongoing transmission and deepening our understanding of resistance mechanisms. Machine learning models provide promising solutions for fast prediction and detection of mutations associated with drug resistance. However, the prediction performance for some drugs still needs improvement, and resistance mechanisms at the level of individual isolates have not yet been considered. xAI-MTBDR introduces an explainable artificial intelligence framework for fast and accurate prediction of MTB drug resistance, identifying resistance-influencing mutations at both the population and individual levels. 

## Dependencies
+ scikit-learn==0.24.2
+ numpy==1.19.5
+ pandas==0.22.0
+ xgboost==1.5.2
+ shap==0.41.0

## File Description
+ To run the code, clone this repository and add the folder to your python path.
+ Leave-One-Out (LOO) strategy is employed to train and test the ensemble model for each drug. When training the models, each dataset was iteratively selected as the test dataset, while the other two datasets were used as the training datasets. For instance, in one iteration, the Gröschel_2021 and Walker_2022 datasets served as the training set, while the CRyPTIC_2022 dataset was used as the test set. Therefore, three datasets are provided in the `data` directory.
  
`ensemble_model.py` predicts drug resistance for 11 drugs using the leave-one-out (LOO) strategy
<br>
`explanation.py` explains the model using rifampicin as an example



## About Us
Bio-Med Big Data Center, CAS Key Laboratory of Computational Biology, Shanghai Institute of Nutrition and Health, Chinese Academy of Sciences.

