# train the encoder
import os
import time
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist
from utils.metrics import epochVal, epochVal_survival, epochBaselineModelVal, epochBaselineModelVal_survival
from utils.loss import BatchLoss
from utils.utils import CIndex_sksurv
import torch.nn.functional as F
from utils.utils import NLLSurvLoss
from models.cmta_utils import define_loss
# for feature
# from utils.feature_importance import ablation_feature_importance, eli5_feature_importance_multimodal
import pandas as pd
import numpy as np

# Function to calculate cosine similarity
def cosine_similarity(grad1, grad2):
    sim = torch.dot(grad1.flatten(), grad2.flatten()) / (grad1.norm() * grad2.norm())
    return sim

def trainDeformPathomicModel(model, dataloader, optimizer, scheduler, logger, args):
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    nll_loss_func = NLLSurvLoss(alpha=0.15)
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)  
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh() 
            
    start = time.time()
    cur_iters = 0
    model.train()
    if args.novalset:
        train_loader, test_loader = dataloader
    else:
        train_loader, val_loader, test_loader = dataloader
    cur_lr = args.lr
    if args.task_type == "survival":
        best_cindex = 0.0
    else:
        best_auc = 0.0
        best_acc = 0.0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.task_type == "survival":
            risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
            
            # features, path_vec, omic_vec, logits, pred, pred_path, pred_omic, fuse_grads, path_grads, omic_grads
            fuse_feat, pathomic_feat_tumor, pathomic_feat_immune, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            S = torch.cumprod(1 - logits[2], dim=1)
                
            if args.task_type == "diag2021":
                loss3 = diag2021_loss_func(logits[2], label[:, 5])
            elif args.task_type == "survival":
                hazard_pred = logits[2]
                loss_nll = nll_loss_func(hazards=hazard_pred, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                loss3 = loss_nll

            elif args.task_type == "grade":
                loss3 = grade_loss_func(logits[2], label[:, 4])
            elif args.task_type == "subtype":
                loss3 = subtype_loss_func(logits[2], label[:, 7])
            if args.return_vgrid:
                batch_sim_loss_tumor = torch.sum(batch_sim_loss_func(logits[3], logits[4]))
                batch_sim_loss_immune = torch.sum(batch_sim_loss_func(logits[5], logits[6]))
                batch_sim_loss = 0.5*batch_sim_loss_tumor + 0.5*batch_sim_loss_immune
                loss = loss3+batch_sim_loss
            else:
                loss = loss3       
            
            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            
            if args.gradient_modulate:
                # Align gradients only if they contradict
                hs = args.mmhid
                out_t = (torch.mm(pathomic_feat_tumor, torch.transpose(model.module.classifier.weight[:, :hs], 0, 1)) +
                            model.module.classifier.bias / 2)
                out_i = (torch.mm(pathomic_feat_immune, torch.transpose(model.module.classifier.weight[:, hs:], 0, 1)) +
                            model.module.classifier.bias / 2)
                
                if args.task_type == "diag2021":
                    loss_t = diag2021_loss_func(out_t, label[:, 5])
                    loss_i = diag2021_loss_func(out_i, label[:, 5])
                elif args.task_type == "survival":
                    hazard_pred_t = torch.sigmoid(out_t)
                    hazard_pred_i = torch.sigmoid(out_i)
                    S_t = torch.cumprod(1 - hazard_pred_t, dim=1)
                    S_i = torch.cumprod(1 - hazard_pred_i, dim=1)

                    loss_nll_t = nll_loss_func(hazards=hazard_pred_t, S=S_t, Y=label[:,8], c=label[:,9], alpha=0)
                    loss_nll_i = nll_loss_func(hazards=hazard_pred_i, S=S_i, Y=label[:,8], c=label[:,9], alpha=0)
                    
                    loss_t = loss_nll_t
                    loss_i = loss_nll_i

                elif args.task_type == "grade":
                    loss_t = grade_loss_func(out_t, label[:, 4])
                    loss_i = grade_loss_func(out_i, label[:, 4])
                elif args.task_type == "subtype":
                    loss_t = subtype_loss_func(out_t, label[:, 7])
                    loss_i = subtype_loss_func(out_i, label[:, 7])
                    
                # Modulation starts here !
                if args.task_type == "diag2021":
                    score_t = sum([F.softmax(out_t)[i][label[:, 5][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 5][i]] for i in range(out_i.size(0))])
                elif args.task_type == "survival":
                    # use cindex values
                    risk_t = -torch.sum(S_t, dim=1) #[B]
                    risk_i = -torch.sum(S_i, dim=1) #[B]
                    censor = label[:, 9]
                    survtime = label[:, 11]
                    if censor.float().mean() != 1:
                        cindex_t = CIndex_sksurv(all_risk_scores=risk_t.detach().cpu().numpy().reshape(-1), all_censorships=censor.detach().cpu().numpy().reshape(-1), all_event_times=survtime.detach().cpu().numpy().reshape(-1))
                        cindex_i = CIndex_sksurv(all_risk_scores=risk_i.detach().cpu().numpy().reshape(-1), all_censorships=censor.detach().cpu().numpy().reshape(-1), all_event_times=survtime.detach().cpu().numpy().reshape(-1))
                    else:
                        print('\ncensor:', censor)
                        print("All samples are censored")
                        cindex_t = None
                        cindex_i = None
                elif args.task_type == "grade":
                    score_t = sum([F.softmax(out_t)[i][label[:, 4][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 4][i]] for i in range(out_i.size(0))])
                elif args.task_type == "subtype":
                    score_t = sum([F.softmax(out_t)[i][label[:, 7][i]] for i in range(out_t.size(0))])
                    score_i = sum([F.softmax(out_i)[i][label[:, 7][i]] for i in range(out_i.size(0))])
                
                
                if args.task_type == 'survival':
                    if cindex_t is not None and cindex_i is not None:
                        ratio_t = cindex_t / cindex_i
                        ratio_i = 1 / ratio_t
                    else:
                        ratio_t = None
                        ratio_i = None
                elif args.task_type != 'survival':
                    ratio_t = score_t / score_i
                    ratio_i = 1 / ratio_t
                
                # print('ratio_t:', ratio_t)

                if ratio_t is not None and ratio_i is not None:
                    i_index=0
                    for grad_t, grad_i in zip(model.module.classifier.weight.grad[:, :hs], model.module.classifier.weight.grad[:, hs:]):
                        if grad_t is not None and grad_i is not None:
                            sim = cosine_similarity(grad_t, grad_i)
                            if sim < 0:
                                if ratio_t < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad_t.flatten(), grad_i.flatten())
                                    proj_scale = dot_product / grad_i.norm()**2
                                    proj_component = proj_scale * grad_i
                                    grad_t = grad_t - proj_component
                                    # model.module.classifier.weight.grad[i_index, :hs] = grad_t
                                    perpen = grad_t - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad_t = grad_t.norm() * unit_perpen
                                    model.module.classifier.weight.grad[i_index, :hs] = grad_t
                                elif ratio_i < 1:
                                    # Calculate the projection of gradient of classifier_tumor onto the direction perpendicular to gradient of classifier
                                    dot_product = torch.dot(grad_i.flatten(), grad_t.flatten())
                                    proj_scale = dot_product / grad_t.norm()**2
                                    proj_component = proj_scale * grad_t
                                    grad_i = grad_i - proj_component
                                    # model.module.classifier.weight.grad[i_index, hs:] = grad_i
                                    perpen = grad_i - proj_component
                                    unit_perpen = perpen / perpen.norm()
                                    grad_i = grad_i.norm() * unit_perpen   
                                    model.module.classifier.weight.grad[i_index, hs:] = grad_i
                        i_index = i_index+1
                        
            
            # Update parameters based on projected gradients
            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))
            
            cur_iters += 1
            if args.rank == 0: 
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    if args.task_type == "survival":
                        test_cindex = epochVal_survival(model, test_loader, args)
                        if not args.novalset:
                            val_cindex = epochVal_survival(model, val_loader, args)
                        if logger is not None and args.return_vgrid:
                            logger.log({'training': {'total loss': loss.item(),
                                                    'batch_sim_loss': batch_sim_loss.item()}})
                            logger.log({'test': {'cindex': test_cindex},
                                        'validation': {'cindex': val_cindex}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test': {'cindex': test_cindex},
                                        'validation': {'cindex': val_cindex}})
                    else:
                        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochVal(model, test_loader, args)
                        if not args.novalset:
                            val_acc, val_f1, val_auc, val_bac, val_sens, val_spec, val_prec = epochVal(model, val_loader, args)
                        if logger is not None and args.return_vgrid:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'task loss3': loss3.item(),
                                                     'batch_sim_loss': batch_sim_loss.item()}})
                            logger.log({'test': {'Accuracy': test_acc,
                                                    'F1 score': test_f1,
                                                    'AUC': test_auc,
                                                    'Balanced Accuracy': test_bac,
                                                    'Sensitivity': test_sens,
                                                    'Specificity': test_spec,
                                                    'Precision': test_prec},
                                        'validation': {'Accuracy': val_acc,
                                                   'F1 score': val_f1,
                                                   'AUC': val_auc,
                                                   'Balanced Accuracy': val_bac,
                                                   'Sensitivity': val_sens,
                                                   'Specificity': val_spec,
                                                   'Precision': val_prec}})
                        elif logger is not None:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'task loss3': loss3.item()}})
                            logger.log({'test': {'Accuracy': test_acc,
                                                    'F1 score': test_f1,
                                                    'AUC': test_auc,
                                                    'Balanced Accuracy': test_bac,
                                                    'Sensitivity': test_sens,
                                                    'Specificity': test_spec,
                                                    'Precision': test_prec},
                                        'validation': {'Accuracy': val_acc,
                                                   'F1 score': val_f1,
                                                   'AUC': val_auc,
                                                   'Balanced Accuracy': val_bac,
                                                   'Sensitivity': val_sens,
                                                   'Specificity': val_spec,
                                                   'Precision': val_prec}})
                    
                    if not args.return_vgrid:
                        print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                            epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item()), end='', flush=True)
                    elif args.return_vgrid:
                        print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f || Loss3M: %.4f || Lossbb: %.4f' % (
                            epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                            cur_lr, loss.item(), loss3.item(), batch_sim_loss.item()), end='', flush=True)  
        scheduler.step()
        
        # method2: save best model
        if args.rank == 0:
            if args.task_type == "survival":
                test_cindex = epochVal_survival(model, test_loader, args)
                if not args.novalset:
                    val_cindex = epochVal_survival(model, val_loader, args)
                if val_cindex > best_cindex:
                    best_cindex = val_cindex
                    saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_cindex_{:f}_.pth'.format(
                        epoch + 1, test_cindex)) 
                    if dist.is_available() and dist.is_initialized():
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()        
                    torch.save(state_dict, saveModelPath)    
            else: 
                test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochVal(model, test_loader, args)
                if not args.novalset:
                    val_acc, val_f1, val_auc, val_bac, val_sens, val_spec, val_prec = epochVal(model, val_loader, args)
                if (val_auc > best_auc) or (val_acc > best_acc):
                    best_auc = val_auc
                    best_acc = val_acc
                    saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_AUC_{:f}_ACC_{:f}_Sens_{:f}_Spec_{:f}_F1_{:f}_.pth'.format(
                        epoch + 1, test_auc, test_acc, test_sens, test_spec, test_f1))
                    if dist.is_available() and dist.is_initialized():
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()        
                    torch.save(state_dict, saveModelPath)

def trainBaselineModel(model, dataloader, optimizer, scheduler, logger, args):
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.15, 2.93, 2.43])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.47, 1.51, 1.0])).float().cuda()).cuda()
    subtype_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 1.72, 2.43])).float().cuda()).cuda()
    nll_loss_func = NLLSurvLoss(alpha=0.15)
    survival_criterion = define_loss(survival_loss="nll_surv")
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size) 
    sim_loss_func = nn.L1Loss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh() 
            
    start = time.time()
    cur_iters = 0
    model.train()
    if args.novalset:
        train_loader, test_loader = dataloader
    else:
        train_loader, val_loader, test_loader = dataloader
    cur_lr = args.lr
    if args.task_type == "survival":
        best_cindex = 0.0
    else:
        best_auc = 0.0
        best_acc = 0.0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if args.task_type == "survival":
            risk_pred_all, censor_all, event_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(train_loader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            # np.asarray([0:label_IDH,1:label_1p19q,2:label_CDKN,3:label_His,4:label_Grade,5:label_Diag,6:label_His_2class, 7:label_Subtype, 8:label_survival, 9:label_censor])
            
            # features, path_vec, omic_vec, logits, pred, pred_path, pred_omic, fuse_grads, path_grads, omic_grads
            if args.mode == 'path':
                # print('training x_path model')
                path_vec, logits, _ = model(x_path)  # (BS,2500,1024), x_path x pathology
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                
            elif args.mode == 'omic':
                # print('training x_omic model')
                omic_vec, logits, _ = model(x_omic=x_omic)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                # print('training x_pathomic model')
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
                hazards = torch.sigmoid(logits[2])
                S = torch.cumprod(1 - hazards, dim=1)
                
            elif args.mode == 'mcat':
                logits, hazards, S = model(x_path=x_path, x_omic=x_omic) 
            elif args.mode == 'cmta':
                logits, hazards, S, P, P_hat, G, G_hat = model(x_path=x_path, x_omic=x_omic)
            if args.mode == 'path' or args.mode == 'omic' or args.mode == 'mcat' or args.mode == 'cmta':    
                if args.task_type == "diag2021":
                    loss3 = diag2021_loss_func(logits, label[:, 5])
                elif args.task_type == "survival":
                    loss_nll = nll_loss_func(hazards=hazards, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                    loss3 = loss_nll
                elif args.task_type == "grade":
                    loss3 = grade_loss_func(logits, label[:, 4])
                elif args.task_type == "subtype":
                    loss3 = subtype_loss_func(logits, label[:, 7])

            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                if args.task_type == "diag2021":
                    loss3 = diag2021_loss_func(logits[2], label[:, 5])
                elif args.task_type == "survival":
                    loss_nll = nll_loss_func(hazards=hazards, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                    loss3 = loss_nll
                elif args.task_type == "grade":
                    loss3 = grade_loss_func(logits[2], label[:, 4])
                elif args.task_type == "subtype":
                    loss3 = subtype_loss_func(logits[2], label[:, 7]) 
            if args.mode == 'cmta':
                sim_loss_P = sim_loss_func(P.detach(), P_hat)
                sim_loss_G = sim_loss_func(G.detach(), G_hat)
                loss = loss3 + 0.5 * (sim_loss_P + sim_loss_G)
            else:
                loss = loss3       
            
            # log loss value only for rank 0
            # to make it consistent with other losses
            if args.rank == 0:
                rank0_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()    
            
            # Update parameters based on projected gradients
            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))
            
            cur_iters += 1
            if args.rank == 0: 
                if cur_iters % 10 == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    # evaluate on test and val set
                    if args.task_type == "survival":
                        test_cindex = epochBaselineModelVal_survival(model, test_loader, args)
                        if not args.novalset:
                            val_cindex = epochBaselineModelVal_survival(model, val_loader, args)
                        if logger is not None:
                            logger.log({'training': {'total loss': loss.item()}})
                            logger.log({'test': {'cindex': test_cindex},
                                        'validation': {'cindex': val_cindex}})
                    else:
                        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochBaselineModelVal(model, test_loader, args)
                        if not args.novalset:
                            val_acc, val_f1, val_auc, val_bac, val_sens, val_spec, val_prec = epochBaselineModelVal(model, val_loader, args)
                        if logger is not None:
                            logger.log({'training': {'total loss': loss.item(),
                                                     'task loss3': loss3.item()}})
                            logger.log({'test': {'Accuracy': test_acc,
                                                    'F1 score': test_f1,
                                                    'AUC': test_auc,
                                                    'Balanced Accuracy': test_bac,
                                                    'Sensitivity': test_sens,
                                                    'Specificity': test_spec,
                                                    'Precision': test_prec},
                                        'validation': {'Accuracy': val_acc,
                                                   'F1 score': val_f1,
                                                   'AUC': val_auc,
                                                   'Balanced Accuracy': val_bac,
                                                   'Sensitivity': val_sens,
                                                   'Specificity': val_spec,
                                                   'Precision': val_prec}})
                    
                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, loss.item()), end='', flush=True) 
                        
        scheduler.step()
        
        # method2: save best model
        if args.rank == 0:
            if args.task_type == "survival":
                test_cindex = epochBaselineModelVal_survival(model, test_loader, args)
                if not args.novalset:
                    val_cindex = epochBaselineModelVal_survival(model, val_loader, args)
                if val_cindex > best_cindex:
                    best_cindex = val_cindex
                    saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_cindex_{:f}_.pth'.format(
                        epoch + 1, test_cindex)) 
                    if dist.is_available() and dist.is_initialized():
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()        
                    torch.save(state_dict, saveModelPath)    
            else: 
                test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochBaselineModelVal(model, test_loader, args)
                if not args.novalset:
                    val_acc, val_f1, val_auc, val_bac, val_sens, val_spec, val_prec = epochBaselineModelVal(model, val_loader, args)
                if (val_auc > best_auc) or (val_acc > best_acc):
                    best_auc = val_auc
                    best_acc = val_acc
                    saveModelPath = os.path.join(args.checkpoints, 'epoch_{:d}_AUC_{:f}_ACC_{:f}_Sens_{:f}_Spec_{:f}_F1_{:f}_.pth'.format(
                        epoch + 1, test_auc, test_acc, test_sens, test_spec, test_f1))
                    if dist.is_available() and dist.is_initialized():
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()        
                    torch.save(state_dict, saveModelPath)
               

def testDeformPathomicModel(model, dataloader, logger, args):
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.56, 3.21, 2.65])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss().cuda()
    subtype_loss_func = nn.CrossEntropyLoss().cuda()
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)  
    nll_loss_func = NLLSurvLoss(alpha=0.15)

    start = time.time()
    
    # Assuming that dataloader: (train_loader, test_loader)
    _, test_loader = dataloader
    
    model.eval()  
    with torch.no_grad(): 
        total_loss = 0.0
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(test_loader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            
            # Forward pass
            _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            
            if args.task_type == "diag2021":
                loss3 = diag2021_loss_func(logits[2], label[:, 5])
            elif args.task_type == "survival":
                S = torch.cumprod(1 - logits[2], dim=1)
                hazard_pred = logits[2]
                loss_nll = nll_loss_func(hazards=hazard_pred, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                loss3 = loss_nll
            elif args.task_type == "grade":
                loss3 = grade_loss_func(logits[2], label[:, 4])
            elif args.task_type == "subtype":
                loss3 = subtype_loss_func(logits[2], label[:, 7])
            if args.return_vgrid:
                batch_sim_loss_tumor = torch.sum(batch_sim_loss_func(logits[3], logits[4]))
                batch_sim_loss_immune = torch.sum(batch_sim_loss_func(logits[5], logits[6]))
                batch_sim_loss = 0.5*batch_sim_loss_tumor + 0.5*batch_sim_loss_immune
                loss = loss3+batch_sim_loss
            else:
                loss = loss3   
                
            total_loss += loss.item() 

            print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)
        
        avg_loss = total_loss / len(test_loader)
        if logger is not None:
            logger.log({'test': {'Average Loss': avg_loss}})
        
        if args.task_type == "survival":
            test_cindex = epochVal_survival(model, test_loader, args)
            if logger is not None:
                logger.log({'training': {'total loss': loss.item()}})
                logger.log({'test': {'cindex': test_cindex}})
        else:
            test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochVal(model, test_loader, args)
            if logger is not None:
                logger.log({'test': {'Accuracy': test_acc,
                                        'F1 score': test_f1,
                                        'AUC': test_auc,
                                        'Balanced Accuracy': test_bac,
                                        'Sensitivity': test_sens,
                                        'Specificity': test_spec,
                                        'Precision': test_prec}})
            
    print("\nTesting completed. Average Loss: {:.4f}".format(avg_loss))
           

def testBaselineModel(model, dataloader, logger, args):
    diag2021_loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, 4.56, 3.21, 2.65])).float().cuda()).cuda()
    grade_loss_func = nn.CrossEntropyLoss().cuda()
    subtype_loss_func = nn.CrossEntropyLoss().cuda()
    batch_sim_loss_func = BatchLoss(args.batch_size, args.world_size)  
    nll_loss_func = NLLSurvLoss(alpha=0.15)

    start = time.time()
    
    # Assuming that dataloader: (train_loader, test_loader)
    _, test_loader = dataloader
    
    model.eval()  
    with torch.no_grad(): 
        total_loss = 0.0
        for i, (x_path, x_omic, x_omic_tumor, x_omic_immune, label) in enumerate(test_loader):
            x_path, x_omic, x_omic_tumor, x_omic_immune, label = x_path.cuda(), x_omic.cuda(), x_omic_tumor.cuda(), x_omic_immune.cuda(), label.cuda()
            
            # Forward pass
            # _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic, x_omic_tumor=x_omic_tumor, x_omic_immune=x_omic_immune)
            if args.mode == 'path':
                path_vec, logits, _ = model(x_path)  # (BS,2500,1024), x_path x pathology
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
            elif args.mode == 'omic':
                omic_vec, logits, _ = model(x_omic=x_omic)
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                _, _, _, logits, _, _, _ = model(x_path=x_path, x_omic=x_omic)
                hazards = torch.sigmoid(logits[2])
                S = torch.cumprod(1 - hazards, dim=1)
            elif args.mode == 'mcat':
                logits, hazards, S = model(x_path=x_path, x_omic=x_omic)
            elif args.mode == 'cmta':
                logits, hazards, S, P, P_hat, G, G_hat = model(x_path=x_path, x_omic=x_omic)
            # Compute loss
            if args.mode == 'path' or args.mode == 'omic':
                if args.task_type == "diag2021":
                    loss3 = diag2021_loss_func(logits, label[:, 5])
                elif args.task_type == "survival":
                    loss_nll = nll_loss_func(hazards=hazards, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                    loss3 = loss_nll
                elif args.task_type == "grade":
                    loss3 = grade_loss_func(logits, label[:, 4])
                elif args.task_type == "subtype":
                    loss3 = subtype_loss_func(logits, label[:, 7])

            elif args.mode == 'pathomic' or args.mode == 'pathomic_original':
                if args.task_type == "diag2021":
                    loss3 = diag2021_loss_func(logits[2], label[:, 5])
                elif args.task_type == "survival":
                    loss_nll = nll_loss_func(hazards=hazards, S=S, Y=label[:,8], c=label[:,9], alpha=0)
                    loss3 = loss_nll
                elif args.task_type == "grade":
                    loss3 = grade_loss_func(logits[2], label[:, 4])
                elif args.task_type == "subtype":
                    loss3 = subtype_loss_func(logits[2], label[:, 7]) 
            loss = loss3   
                
            total_loss += loss.item() 

            print('\rTest Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' % (
                i + 1, len(test_loader), time.time() - start, loss.item()), end='', flush=True)
        
        avg_loss = total_loss / len(test_loader)
        if logger is not None:
            logger.log({'test': {'Average Loss': avg_loss}})
        
        if args.task_type == "survival":
            test_cindex = epochBaselineModelVal_survival(model, test_loader, args)
            print('test_cindex:', test_cindex)
            if logger is not None:
                logger.log({'training': {'total loss': loss.item()}})
                logger.log({'test': {'cindex': test_cindex}})
        else:
            test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec = epochBaselineModelVal(model, test_loader, args)
            if logger is not None:
                logger.log({'test': {'Accuracy': test_acc,
                                        'F1 score': test_f1,
                                        'AUC': test_auc,
                                        'Balanced Accuracy': test_bac,
                                        'Sensitivity': test_sens,
                                        'Specificity': test_spec,
                                        'Precision': test_prec}})
            
    print("\nTesting completed. Average Loss: {:.4f}".format(avg_loss))
    
            


            