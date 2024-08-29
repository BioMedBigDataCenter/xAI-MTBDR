import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

start_time = time.time()

# Definitions
drugs = ["Rifampicin", "Isoniazid", "Ethambutol", "Pyrazinamide", "Amikacin", "Capreomycin", "Ethionamide", "Kanamycin", "Levofloxacin", "Moxifloxacin", "Streptomycin"]

column_names = ['Algorithm', 'Drug', 'AUC']

results = pd.DataFrame(columns=column_names)
results_index = 0

data_dir = './data/'
out_dir = './output/'

# Load training data
X = np.loadtxt(data_dir + 'Gröschel-Walker_train_data_new.csv', delimiter=',')
y = np.loadtxt(data_dir + 'Gröschel-Walker_train_data_pheno_new.csv', delimiter=',')

# Load test data
X_test = np.loadtxt(data_dir + 'CRyPTIC_test_data_new.csv', delimiter=',')
y_test = np.loadtxt(data_dir + 'CRyPTIC_test_data_pheno_new.csv', delimiter=',')


for i, drug in enumerate(drugs):
    # Remove rows that do not contain resistance information
    y_drug = y[:, i]
    y_train = y_drug[y_drug != -1]
    X_train = X[y_drug != -1]

    y_drug_test = y_test[:, i]
    y_val = y_drug_test[y_drug_test != -1]
    X_val = X_test[y_drug_test != -1]

    # Define three basic models for ensemble
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=9, reg_lambda=0, reg_alpha=0.1, n_estimators=750, random_state=0)
    lm = LogisticRegression(penalty='l1', C=2.15443469e-02, solver='liblinear', random_state=0)
    rf = RandomForestClassifier(n_estimators=1000, max_features='auto', min_samples_leaf=0.002, random_state=0)

    # Combine the three basic  models into a voting ensemble and training
    model = VotingClassifier(estimators=[('xgb', xgb_model), ('lm', lm), ('rf', rf)], voting='soft', weights=[2, 1.5, 1]).fit(X_train, y_train)

    # Get the probabilities and predicted labels
    proba_val = model.predict_proba(X_val)
    predict_val = model.predict(X_val)

    # Calculate the AUC for the current drug
    auc = roc_auc_score(y_val, proba_val[:, 1])

    # Store the results for the ensemble model and the current drug
    results.loc[results_index] = ['Ensemble model', drug, auc]
    results_index += 1

# Save the results DataFrame to a CSV file
results.to_csv(out_dir + 'results_ensemble_voting_for_CRyPTIC_predict.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print("Code runtime is: ", elapsed_time, "s")
