import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score, root_mean_squared_error
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import pandas as pd
from utils import Meter
from vis_utils import weight_visulize
def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)

    return smiles, bg, labels,  mask


def run_a_train_epoch_heterogeneous(args, epoch, model, data_loader, loss_criterion_c, loss_criterion_r, optimizer, task_weight=None):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        mask = mask.float().to(args['device'])
        labels.float().to(args['device'])
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
        bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
        bg = bg.to(args['device'])
        logits = model(bg, atom_feats, bond_feats, norm=None)
        labels = labels.type_as(logits).to(args['device'])
        # calculate loss according to different task class
        if args['task_class'] == 'classification_regression':
            # split classification and regression
            logits_c = logits[:,:args['classification_num']]
            labels_c = labels[:,:args['classification_num']]
            mask_c = mask[:,:args['classification_num']]

            logits_r = logits[:,args['classification_num']:]
            labels_r = labels[:,args['classification_num']:]
            mask_r = mask[:,args['classification_num']:]
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float()).mean() \
                       + (loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float()).mean()
            else:
                task_weight_c = task_weight[:args['classification_num']]
                task_weight_r = task_weight[args['classification_num']:]
                loss = (torch.mean(loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float(), dim=0)*task_weight_c).mean() \
                       + (torch.mean(loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float(), dim=0)*task_weight_r).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update(logits_c, labels_c, mask_c)
            train_meter_r.update(logits_r, labels_r, mask_r)
            del bg, mask, labels, atom_feats, bond_feats, loss, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
            torch.cuda.empty_cache()
        elif args['task_class'] == 'classification':
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_c(logits, labels) * (mask != 0).float(),dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update(logits, labels, mask)
            del bg, mask, labels, atom_feats, bond_feats, loss,  logits
            torch.cuda.empty_cache()
        else:
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_r(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_r(logits, labels) * (mask != 0).float(), dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_r.update(logits, labels, mask)
            del bg, mask, labels, atom_feats, bond_feats, loss,  logits
            torch.cuda.empty_cache()
    if args['task_class'] == 'classification_regression':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']) +
                              train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], 'r2+auc', train_score))
    elif args['task_class'] == 'classification':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['classification_metric_name'], train_score))
    else:
        train_score = np.mean(train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['regression_metric_name'], train_score))


def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])
            logits = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                torch.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            return eval_meter_c.compute_metric(args['classification_metric_name'])
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])


def run_an_eval_epoch_pih(args, model, data_loader, output_path):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    smiles_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            smiles_list = smiles_list + smiles
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])
            logits = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                torch.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            y_pred, y_true = eval_meter_c.compute_metric('return_pred_true')
            result = pd.DataFrame(columns=['smiles', 'pred', 'true'])
            result['smiles'] = smiles_list
            result['pred'] = np.squeeze(y_pred.numpy()).tolist()
            result['true'] = np.squeeze(y_true.numpy()).tolist()
            result.to_csv(output_path, index=None)
        else:
            y_pred, y_true = eval_meter_r.compute_metric('return_pred_true')
            result = pd.DataFrame(columns=['smiles', 'pred', 'true'])
            result['smiles'] = smiles_list
            result['pred'] = np.squeeze(y_pred.numpy()).tolist()
            result['true'] = np.squeeze(y_true.numpy()).tolist()
            result.to_csv(output_path, index=None)
            
def run_an_eval_epoch_re(args, model, data_loader, output_path):
    model.eval()
    eval_meter_r = Meter()
    smiles_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            smiles_list = smiles_list + smiles
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])
            logits = model(bg, atom_feats, bond_feats, norm=None).to(args['device'])
            labels = labels.type_as(logits).to(args['device'])
            # Mask non-existing labels
            eval_meter_r.update(logits, labels, mask)
            del smiles, bg, mask, labels, atom_feats, bond_feats, logits
            torch.cuda.empty_cache()

        y_pred, y_true = eval_meter_r.return_pred_true_re()

        result = pd.DataFrame(columns=['smiles', 'pred', 'true'])
        result['smiles'] = smiles_list
        result['pred'] = np.squeeze(y_pred.numpy()).tolist()
        result['true'] = np.squeeze(y_true.numpy()).tolist()
        result.to_csv(output_path, index=None)
        
        return y_pred
        


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import torch

def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])

            logits = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])

            if args['task_class'] == 'classification_regression':
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]

                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)

                all_preds.append(logits_r.cpu())
                all_labels.append(labels_r.cpu())

            elif args['task_class'] == 'classification':
                eval_meter_c.update(logits, labels, mask)

            else:  # regression only
                eval_meter_r.update(logits, labels, mask)
                all_preds.append(logits.cpu())
                all_labels.append(labels.cpu())

            del smiles, bg, mask, labels, atom_feats, bond_feats, logits
            torch.cuda.empty_cache()

    if args['task_class'] == 'classification_regression':
        metric_c = eval_meter_c.compute_metric(args['classification_metric_name'])  
        r2 = eval_meter_r.compute_metric('r2')
        mae, rmse = compute_mae_rmse(all_preds, all_labels)
        return r2, mae, rmse

    elif args['task_class'] == 'classification':
        return eval_meter_c.compute_metric(args['classification_metric_name']), [], []

    else:  # regression only
        r2 = eval_meter_r.compute_metric('r2')
        mae, rmse = compute_mae_rmse(all_preds, all_labels)
        return r2, mae, rmse


    


def run_an_eval_epoch_heterogeneous_return_weight_py(args, model, data_loader, vis_list=None, vis_task='can'):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            #####
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])
            logits, atom_weight_list, node_feats = model(bg, atom_feats, bond_feats, norm=None)
            labels = labels.type_as(logits).to(args['device'])
            logits_c = logits[:, :args['classification_num']]
            logits_c = torch.sigmoid(logits_c)
            # different tasks with different atom weight

            for mol_index in range(len(smiles)):
                atom_smiles = smiles[mol_index]
                if atom_smiles in vis_list:
                    for tasks_index in range(1):
                        if args['all_task_list'][tasks_index] == vis_task:
                            if labels[mol_index, tasks_index]!=123456:
                                bg.ndata['w'] = atom_weight_list[tasks_index]
                                bg.ndata['feats'] = node_feats
                                unbatch_bg = dgl.unbatch(bg)
                                one_atom_weight = unbatch_bg[mol_index].ndata['w']
                                one_atom_feats = unbatch_bg[mol_index].ndata['feats']
                                # visual selected molecules
                                print('Tasks:', tasks_index, args['all_task_list'][tasks_index], "**********************")
                                if tasks_index < 26:
                                    print('Predict values:', logits_c[mol_index, tasks_index])
                                else:
                                    print('Predict values:', logits[mol_index, tasks_index])
                                print('True values:', labels[mol_index, tasks_index])
                                weight_visulize_py(atom_smiles, one_atom_weight)
                else:
                    continue


def run_an_eval_epoch_heterogeneous_generate_weight(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id+1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats, bond_feats, norm=None)
            for atom_weight in atom_weight_list:
                atom_list_all.append(atom_weight[args['select_task_index']])
    task_name = args['select_task_list'][0]
    atom_weight_list = pd.DataFrame(atom_list_all, columns=['atom_weight'])
    atom_weight_list.to_csv(task_name+"_atom_weight.csv", index=None)


def generate_chemical_environment(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id + 1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            print(bg.ndata[args['atom_data_field']][1])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            bg = bg.to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats, bond_feats, norm=None)
            print('after training:', bg.ndata['h'][1])


def generate_mol_feats(args, model, data_loader, dataset_output_path):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            bg = bg.to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            feats = model(bg, atom_feats, bond_feats, norm=None).numpy().tolist()
            feats_name = ['graph-feature' + str(i+1) for i in range(64)]
            data = pd.DataFrame(feats, columns=feats_name)
            data['smiles'] = smiles
            data['labels'] = labels.squeeze().numpy().tolist()
    data.to_csv(dataset_output_path, index=None)


class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop20250529pred.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename ='D:/model//{}_early_stop20250529pred.pth'.format(task_name)
            

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
        # print(self.filename)
    

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])
        

    def load_pretrained_model(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked']
        if torch.cuda.is_available():
            pretrained_model = torch.load('D:/model/'+self.pretrained_model)
        else:
            pretrained_model = torch.load('D:/model/'+self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        

    def load_model_attention(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.bias',
                                 'weighted_sum_readout.shared_weighting.0.weight',
                                 'weighted_sum_readout.shared_weighting.0.bias',
                                 ]
        if torch.cuda.is_available():
            pretrained_model = torch.load('D:/model/' + self.pretrained_model)
        else:
            pretrained_model = torch.load('D:/model/' + self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)



        from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def compute_mae_rmse(preds_list, labels_list):
    preds = torch.cat(preds_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()

    mae_list = []
    rmse_list = []
    for i in range(preds.shape[1]):
        true = labels[:, i]
        pred = preds[:, i]

        mask = (true != 123456) & (~np.isnan(true))
        if np.sum(mask) > 0:
            mae = mean_absolute_error(true[mask], pred[mask])
            rmse = root_mean_squared_error(true[mask], pred[mask])
        else:
            mae = rmse = np.nan

        mae_list.append(mae)
        rmse_list.append(rmse)

    return mae_list, rmse_list
