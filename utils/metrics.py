import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_score
from utils.utils import CIndex_sksurv
from imblearn.metrics import sensitivity_score, specificity_score


def compute_avg_metrics(groundTruth, activations):
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average='macro')
    try:
        auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class='ovr')
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    spec_macro = specificity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    prec_macro = precision_score(y_true=groundTruth, y_pred=predictions, average='macro')

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro, prec_macro


def compute_confusion_matrix(groundTruth, activations, labels):

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions, labels=labels)

    return cm

# for deformpathomic model
def epochVal(model, dataLoader, args):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        # for i, (x_path, x_omic, label) in enumerate(dataLoader):
            # x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            # output = F.softmax(logits, dim=1)
            # groundTruth = torch.cat((groundTruth, label[:, 5]))
            # activations = torch.cat((activations, output))
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            output = F.softmax(logits[2], dim=1)
            if args.task_type == "diag2021":
                groundTruth = torch.cat((groundTruth, label[:, 5]))
            elif args.task_type == "grade":
                groundTruth = torch.cat((groundTruth, label[:, 4]))
            elif args.task_type == "subtype":
                groundTruth = torch.cat((groundTruth, label[:, 7]))
            activations = torch.cat((activations, output))
            
        acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec, prec

def epochVal_survival(model, dataLoader, args):
    training = model.training
    model.eval()

    # groundTruth = torch.Tensor().cuda()
    # activations = torch.Tensor().cuda()
    
    risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    with torch.no_grad():
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            S = torch.cumprod(1 - logits[2], dim=1) #[B,4]
            risk = -torch.sum(S, dim=1) #[B]
            # logits:[hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
            risk_pred_all = np.concatenate((risk_pred_all, risk.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            # event_all = np.concatenate((event_all, label[:, 10].detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information
        
        # print('risk_pred_all.shape:', risk_pred_all.shape) #236
        # print('event_all.shape:', event_all.shape) #236

        cindex_epoch = CIndex_sksurv(all_risk_scores=risk_pred_all, all_censorships=censor_all, all_event_times=survtime_all)

    model.train(training)

    return cindex_epoch

# for BaselineModel val
def epochBaselineModelVal(model, dataLoader, args):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        # for i, (x_path, x_omic, label) in enumerate(dataLoader):
            # x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
            # fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            # output = F.softmax(logits, dim=1)
            # groundTruth = torch.cat((groundTruth, label[:, 5]))
            # activations = torch.cat((activations, output))
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            # fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            if args.mode == 'path':
                path_vec, logits, _ = model(x_path)  # (BS,2500,1024), x_path x pathology
            elif args.mode == 'omic':
                omic_vec, logits, _ = model(x_omic=x_omic)
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
            elif args.mode == 'mcat':
                logits, hazards, S = model(x_path=x_path, x_omic=x_omic) # return hazards, S, Y_hat, A_raw, results_dict
            elif args.mode == 'cmta':
                # hazards= model(x_path=x_path, x_omic=x_omic) # return hazards, S, Y_hat, A_raw, results_dict
                # hazards, S, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_genomics_encoder, cls_token_genomics_decoder 
                logits, hazards, S, P, P_hat, G, G_hat = model(x_path=x_path, x_omic=x_omic)

            if args.mode == 'path' or args.mode == 'omic' or args.mode == 'mcat' or args.mode == 'cmta':
                output = F.softmax(logits, dim=1)
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                output = F.softmax(logits[2], dim=1)
            if args.task_type == "diag2021":
                groundTruth = torch.cat((groundTruth, label[:, 5]))
            elif args.task_type == "grade":
                groundTruth = torch.cat((groundTruth, label[:, 4]))
            elif args.task_type == "subtype":
                groundTruth = torch.cat((groundTruth, label[:, 7]))
            activations = torch.cat((activations, output))
            
        acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec, prec

def epochBaselineModelVal_survival(model, dataLoader, args):
    training = model.training
    model.eval()
    
    risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
    with torch.no_grad():
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(dataLoader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            # fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            
            if args.mode == 'path':
                path_vec, logits, _ = model(x_path)  # (BS,2500,1024), x_path x pathology
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'omic':
                omic_vec, logits, _ = model(x_omic=x_omic)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
                hazards = torch.sigmoid(logits[2])
                S = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(S, dim=1) #[8]
            elif args.mode == 'mcat':
                logits, hazards, S = model(x_path=x_path, x_omic=x_omic) # return hazards, S, Y_hat, A_raw, results_dict
                risk = -torch.sum(S, dim=1) #[8,4]
            elif args.mode == 'cmta':
                logits, hazards, S, P, P_hat, G, G_hat = model(x_path=x_path, x_omic=x_omic)
                risk = -torch.sum(S, dim=1) #[8]
            
            # logits:[hazard_tumor, hazard_immune, hazard, omic_tumor, vgrid_tumor, omic_immune, vgrid_immune]
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
                
            # risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            risk_pred_all = np.concatenate((risk_pred_all, risk.detach().cpu().numpy().reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, label[:, 9].detach().cpu().numpy().reshape(-1)))   # Logging Information
            event_all = np.concatenate((event_all, label[:, 10].detach().cpu().numpy().reshape(-1)))
            # print('event_all.shape:', event_all.shape)
            survtime_all = np.concatenate((survtime_all, label[:, 11].detach().cpu().numpy().reshape(-1)))   # Logging Information
        
        cindex_epoch = CIndex_sksurv(all_risk_scores=risk_pred_all, all_censorships=censor_all, all_event_times=survtime_all)
        # acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return cindex_epoch


def ablation_epochVal(model, dataLoader, gene_list_length):
    model.eval()

    with torch.no_grad():
        # Store the importance of each gene feature
        # Ablation study for each gene feature
        difference_acc_list = []
        # for i in range(gene_list_length):
        for i in range(2):
            groundTruth = torch.Tensor().cuda()
            prediction = torch.Tensor().cuda()
            modified_prediction = torch.Tensor().cuda()
            print("Processing {:d}/{:d} gene".format(i+1, gene_list_length))
            for j, (x_path, x_omic, label) in enumerate(dataLoader):
                x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
                
                fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
                output = F.softmax(logits, dim=1)
                groundTruth = torch.cat((groundTruth, label[:, 5]))
                prediction = torch.cat((prediction, output))
                
                ablated_genes = x_omic.clone()
                ablated_genes[:, i] = 0  # Zero out the ith gene feature
                fuse_feat, path_feat, omic_feat, modified_logits, _, _, _ = model(x_path=x_path, x_omic=ablated_genes)
                modified_output = F.softmax(modified_logits, dim=1)
                modified_prediction = torch.cat((modified_prediction, modified_output))

            acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, prediction)
            modified_acc, f1, auc, bac, sens, spec, prec = compute_avg_metrics(groundTruth, modified_prediction)
            print('acc, modif_acc:', acc, modified_acc)
            
            difference_acc_list.append(acc - modified_acc) 
            
    return difference_acc_list

def eli5_epochVal(model, x_path, x_omic, label):
    model.eval()
    with torch.no_grad():
        # Store the importance of each gene feature
        # for i in range(gene_list_length):
        x_path, x_omic, label = x_path.cuda(), x_omic.cuda(), label.cuda()
                
        fuse_feat, path_feat, omic_feat, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
        # calculate the accuracy
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        acc = pred.eq(label[:, 5].view_as(pred)).sum().item() / float(label[:, 5].shape[0])
                    
    return acc


def epochTest(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions)
    model.train(training)

    return cm