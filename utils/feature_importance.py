import shap
import torch
import pandas as pd
import numpy as np
# from eli5.permutation_importance import get_score_importances_multimodal
from utils.metrics import ablation_epochVal, eli5_epochVal


def ablation_feature_importance(model, test_loader, gene_list):
    difference_acc_list = ablation_epochVal(model, test_loader, len(gene_list))
    print(difference_acc_list)
    df_difference_acc_list = pd.DataFrame(difference_acc_list)
    df_difference_acc_list.to_csv('difference_acc_list.csv', index = False)
    
    sorted_id = sorted(range(len(difference_acc_list)), key=lambda k: difference_acc_list[k], reverse=True)
    gene_importance = []
    # for i in range(len(gene_list)):
    for i in range(2):
        print('sorted index', sorted_id[i])
        gene_importance.append(gene_list[sorted_id[i]])
    df_gene_importance = pd.DataFrame(gene_importance)
    df_gene_importance.to_csv('gene_importance.csv', index = False)
    print('gene_importance.csv has been saved!')

def eli5_feature_importance_multimodal(model, test_loader, gene_list):
    feature_importances = []
    # accScore = eli5_epochVal(model, x_path, x_omic, label, gene_list_length)
    for i, (x_path, x_omic, label) in enumerate(test_loader):
        x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
        base_score, score_decreases = get_score_importances_multimodal(score_func = eli5_epochVal, model=model, X1 = x_path, X2 = x_omic, y = label)
        feature_importances.append(np.mean(score_decreases, axis=0))
        print('The ', i, ' batch feature_importances', feature_importances)
    avg_feature_importances = np.mean(feature_importances, axis=0)
    print('average feature_importances:', avg_feature_importances)


def shap_feature_importance(model, test_loader):
    # Extract all samples from the test_loader to form the background dataset
    background_genes = []
    background_wsi = []
    
    # for i, (x_path, x_omic, label) in enumerate(test_loader):
    for sample_wsi, sample_genes, _ in test_loader:
        background_genes.append(sample_genes.numpy())
        background_wsi.append(sample_wsi.numpy())
    
    # Convert lists to numpy arrays
    background_genes = np.vstack(background_genes)
    background_wsi = np.vstack(background_wsi)

    # fea_data = np.hstack((background_wsi, background_genes))
    fea_data = [background_wsi, background_genes]
    # Create a SHAP explainer using the entire test dataset as the background
    explainer = shap.DeepExplainer(model, fea_data)

    # Compute SHAP values for all test samples
    shap_values = explainer.shap_values(fea_data)

    # SHAP values will be a list with contributions for each class for each modality
    # Here we assume a binary classification for simplicity; adjust as needed
    genes_contributions = np.array(shap_values[0][1])
    wsi_contributions = np.array(shap_values[0][0])

    # Calculate the average SHAP values across all test samples
    avg_genes_contributions = np.mean(genes_contributions, axis=0)
    avg_wsi_contributions = np.mean(wsi_contributions, axis=0)

    # Now you can visualize or return the average importance of each feature
    df_gene_list = pd.read('gene_list.csv')
    shap.summary_plot(avg_genes_contributions, feature_names=df_gene_list['gene_name'].values.tolist())
    
    # return avg_genes_contributions, avg_wsi_contributions


