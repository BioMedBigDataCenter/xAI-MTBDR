#how well the model distinguishes mono-resistant strains from MDR or XDR cases?
#aims: confusion matrix of INH mono-resistant and MDR/XDR.

import pandas as pd
import sys

def main():
    if len(sys.argv) != 6:
        print("Usage: python script.py <label_file> <rif_file> <inh_file> <emb_file> <pyz_file>")
        sys.exit(1)
    
    label_file = sys.argv[1]
    rif_file = sys.argv[2]
    inh_file = sys.argv[3]
    emb_file = sys.argv[4]
    pyz_file = sys.argv[5]
    
    df_label = pd.read_csv(label_file, header=0)  
    
    df_rif = pd.read_csv(rif_file)
    df_inh = pd.read_csv(inh_file)
    df_emb = pd.read_csv(emb_file)
    df_pyz = pd.read_csv(pyz_file)
    
    df_label = df_label.iloc[:, :4]
    
    df_label.columns = ['label_rif', 'label_inh', 'label_emb', 'label_pyz']
    
    df_merged = pd.DataFrame(columns=[
        'label_rif', 'label_inh', 'label_emb', 'label_pyz',
        'pred_rif', 'pred_inh', 'pred_emb', 'pred_pyz'
    ])
    
    idx_rif, idx_inh, idx_emb, idx_pyz = 0, 0, 0, 0
    
    for i in range(len(df_label)):
        df_merged.loc[i, 'label_rif'] = df_label.loc[i, 'label_rif']
        df_merged.loc[i, 'label_inh'] = df_label.loc[i, 'label_inh']
        df_merged.loc[i, 'label_emb'] = df_label.loc[i, 'label_emb']
        df_merged.loc[i, 'label_pyz'] = df_label.loc[i, 'label_pyz']
        
        if df_label.loc[i, 'label_rif'] == -1:
            df_merged.loc[i, 'pred_rif'] = -1
        else:
            pred_prob = df_rif.loc[idx_rif, 'Predicted_Prob']
            df_merged.loc[i, 'pred_rif'] = 0 if pred_prob < 0.5 else 1
            idx_rif += 1 

        if df_label.loc[i, 'label_inh'] == -1:
            df_merged.loc[i, 'pred_inh'] = -1
        else:
            pred_prob = df_inh.loc[idx_inh, 'Predicted_Prob']
            df_merged.loc[i, 'pred_inh'] = 0 if pred_prob < 0.5 else 1
            idx_inh += 1
        
        if df_label.loc[i, 'label_emb'] == -1:
            df_merged.loc[i, 'pred_emb'] = -1
        else:
            pred_prob = df_emb.loc[idx_emb, 'Predicted_Prob']
            df_merged.loc[i, 'pred_emb'] = 0 if pred_prob < 0.5 else 1
            idx_emb += 1
        
        if df_label.loc[i, 'label_pyz'] == -1:
            df_merged.loc[i, 'pred_pyz'] = -1
        else:
            pred_prob = df_pyz.loc[idx_pyz, 'Predicted_Prob']
            df_merged.loc[i, 'pred_pyz'] = 0 if pred_prob < 0.5 else 1
            idx_pyz += 1
    
    df_no_neg1 = df_merged[~(df_merged == -1).any(axis=1)].copy()
    
    all_susceptible = (
        (df_no_neg1['label_rif'] == 1) & 
        (df_no_neg1['label_inh'] == 1) & 
        (df_no_neg1['label_emb'] == 1) & 
        (df_no_neg1['label_pyz'] == 1)
    )
    df_merged_non_ambigous = df_no_neg1[~all_susceptible].copy()
    
    df_merged_non_ambigous.reset_index(drop=True, inplace=True)
    
    df_merged_non_ambigous['true_pheno'] = ''
    df_merged_non_ambigous['pred_pheno'] = ''
    
    for i in range(len(df_merged_non_ambigous)):
        row = df_merged_non_ambigous.iloc[i]
        
        if row['label_inh'] == 1 and row['label_rif'] == 0 and row['label_emb'] == 1 and row['label_pyz'] == 1:
            true_val = "mono"
        elif row['label_inh'] == 0 and row['label_rif'] == 0:
            true_val = "MDR"
        else:
            true_val = "other"
        
        if row['pred_inh'] == 1 and row['pred_rif'] == 0 and row['pred_emb'] == 1 and row['pred_pyz'] == 1:
            pred_val = "mono"
        elif row['pred_inh'] == 0 and row['pred_rif'] == 0:
            pred_val = "MDR"
        else:
            pred_val = "other"
        
        df_merged_non_ambigous.loc[i, 'true_pheno'] = true_val
        df_merged_non_ambigous.loc[i, 'pred_pheno'] = pred_val
    
    categories = ['mono', 'MDR', 'other']
    
    confusion_matrix = pd.crosstab(
        df_merged_non_ambigous['true_pheno'], 
        df_merged_non_ambigous['pred_pheno'],
        rownames=['True'],
        colnames=['Predicted'],
        dropna=False
    )
    
    confusion_matrix = confusion_matrix.reindex(
        index=categories, 
        columns=categories, 
        fill_value=0
    )
    
    confusion_matrix = confusion_matrix.reset_index()
    
    total_samples = len(df_merged_non_ambigous)
    correct_predictions = (df_merged_non_ambigous['true_pheno'] == df_merged_non_ambigous['pred_pheno']).sum()
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    output = []
    
    output.append("Merged Data with Phenotype Classification")
    output.append(df_merged_non_ambigous.to_csv(index=False))
    
    if total_samples > 0:
        output.append("\nConfusion Matrix")
        output.append(confusion_matrix.to_csv(index=False))
        output.append(f"\nTotal samples: {total_samples}")
        output.append(f"Correct predictions: {correct_predictions}")
        output.append(f"Accuracy: {accuracy:.4f}")
    else:
        output.append("\nNo valid samples for phenotype classification (all samples were filtered out)")
    
    print("\n".join(output))
    df_merged_non_ambigous.to_csv("CRyPTIC_true_pred.csv",index=False)

if __name__ == "__main__":
    main() 
