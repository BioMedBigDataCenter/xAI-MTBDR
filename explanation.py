import time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

start_time = time.time()

data_dir = './data/'
out_dir = './output/'

# Load training data
X = np.loadtxt(data_dir + 'Gröschel-Walker_train_data_new.csv', delimiter=',')
X_have_col = pd.read_csv(data_dir + 'Gröschel-Walker_train_data.csv')
y = np.loadtxt(data_dir + 'Gröschel-Walker_train_data_pheno_Rifampicin_label_reverse_new.csv', delimiter=',')

# Load test data
X_test = np.loadtxt(data_dir + 'CRyPTIC_test_data_new.csv', delimiter=',')
y_test = np.loadtxt(data_dir + 'CRyPTIC_test_data_pheno_Rifampicin_label_reverse_new.csv', delimiter=',')


# Remove rows that do not contain resistance information
y_drug = y
y_train = y_drug[y_drug != -1]
X_train = X[y_drug != -1]

y_drug_test = y_test
y_val = y_drug_test[y_drug_test != -1]
X_val = X_test[y_drug_test != -1]

# Train and predict on XGBoost
xgb_model = xgb.XGBClassifier(objective='binary:logistic',learning_rate=0.1, max_depth=9, min_child_weight=1, n_estimators=750, random_state=0).fit(X_train, y_train)

# Get the probabilities and predicted labels
proba_val = xgb_model.predict_proba(X_val)
predict_val = xgb_model.predict(X_val)

# SHAP value explanation
tree_explainer = shap.TreeExplainer(xgb_model)
tree_shap_values = tree_explainer.shap_values(X_val)

# Create a DataFrame to hold SHAP values and additional information
column = list(X_have_col.columns)
shap_values_result = pd.DataFrame(tree_shap_values, columns=column)
new_column_y_test = pd.Series(y_val)
shap_values_result.insert(0, 'label', new_column_y_test)

new_column_predict = pd.Series(predict_val)
shap_values_result.insert(1, 'predict_result', new_column_predict)

new_column_pred_xgb = pd.Series(proba_val[:, 1])
shap_values_result.insert(2, 'probability_result', new_column_pred_xgb)

# Save the SHAP values DataFrame to a CSV file
shap_values_result.to_csv(out_dir + 'shap_values_CRyPTIC_Rifampicin_all.csv', index=True, index_label='shap_sample_index')

# Create a summary plot of SHAP values
plt.figure(figsize=(12, 7))
shap.summary_plot(tree_shap_values, X_val, show=False, plot_size=None, max_display=10, plot_type=None, feature_names=column)
plt.savefig(out_dir + 'shap_beeswarm_CRyPTIC_Rifampicin.pdf', dpi=1200)

# Generate individual waterfall plots for each sample
for i in range(len(y_val)):
    plt.figure(figsize=(12, 7))
    shap.plots._waterfall.waterfall_legacy(
        tree_explainer.expected_value,
        tree_shap_values[i],
        show=False, max_display=10, feature_names=column)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir + 'shap_waterfall_CRyPTIC_' + str(i) + '_.pdf', dpi = 1200, bbox_inches = 'tight')

end_time = time.time()
elapsed_time = end_time - start_time
print("Code runtime is: ", elapsed_time, "s")
