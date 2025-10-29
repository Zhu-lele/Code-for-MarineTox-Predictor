import numpy as np
#from utils import build_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
#from utils.MY_GNN import collate_molgraphs, EarlyStopping, run_a_train_epoch_heterogeneous, \run_an_eval_epoch_heterogeneous, set_random_seed, MGA, pos_weight
import os
import time
import pandas as pd
start = time.time()


# fix parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['atom_data_field'] = 'atom'
args['bond_data_field'] = 'etype'
args['classification_metric_name'] = 'roc_auc'
args['regression_metric_name'] = 'r2'
# model parameter
args['num_epochs'] = 500
args['patience'] = 50
args['batch_size'] = 128
args['mode'] = 'higher'
args['in_feats'] = 41
args['rgcn_hidden_feats'] = [64, 64]
args['classifier_hidden_feats'] = 64
args['rgcn_drop_out'] = 0.2
args['drop_out'] = 0.2
args['lr'] = 3
args['weight_decay'] = 5
args['loop'] = True

# task name (model name)
args['task_name'] = 'aquatic data'  # change
args['data_name'] = 'aquatic data'  # change
args['times'] = 1

# selected task, generate select task index, task class, and classification_num
args['select_task_list'] = ['Americamysis bahia_acute',
                            'Americamysis bahia_chronic',
                            'Artemia salina_acute',                
                            'Brachionus plicatilis_acute',
                           'Chaetogammarus marinus_acute',
                            'Crangon septemspinosa_acute',
                            'Crassostrea virginica_acute',                
                            'Cyprinodon variegatus_acute',
                           'Cyprinodon variegatus_chronic',
                            'Danio rerio_chronic',
                            'Farfantepenaeus duorarum_acute',                
                            'Fundulus heteroclitus_acute',
                             'Fundulus heteroclitus_chronic',
                            'Gadus morhua_chronic',                                                      
                            'Gasterosteus aculeatus_acute',    
                            'Lepeophtheirus salmonis_acute',
                            'Lepomis macrochirus_acute',
                            'Litopenaeus vannamei_acute',
                            'Menidia beryllina_acute',                
                            'Menidia menidia_acute',
                            'Mytilus galloprovincialis_chronic',
                            'Nitocra spinipes_acute',
                            'Oncorhynchus mykiss_chronic',
                            'Oncorhynchus mykiss_acute',
                           'Oryzias melastigma_chronic',
                            'Palaemonetes pugio_acute',
                             'Palaemonetes pugio_chronic',  
                            'Palaemonetes sp._acute',
                            'Pimephales promelas_acute',
                            'Pimephales promelas_chronic',
                            'Raphidocelis subcapitata_acute',
                            'Rivulus marmoratus_chronic',
                            'Scolelepis fulginosa_acute',
                            'Skeletonema costatum_acute',
                                'Skeletonema costatum_chronic',                
                            'Dunaliella tertiolecta_acute',
                           'Isochrysis galbana_acute',
                            'Phaeodactylum tricornutum_acute']  # change
args['select_task_index'] = []
args['classification_num'] = 0
args['regression_num'] = 0
args['all_task_list'] = ['Americamysis bahia_acute',
                            'Americamysis bahia_chronic',
                            'Artemia salina_acute',                
                            'Brachionus plicatilis_acute',
                           'Chaetogammarus marinus_acute',
                            'Crangon septemspinosa_acute',
                            'Crassostrea virginica_acute',                
                            'Cyprinodon variegatus_acute',
                           'Cyprinodon variegatus_chronic',
                            'Danio rerio_chronic',
                            'Farfantepenaeus duorarum_acute',                
                            'Fundulus heteroclitus_acute',
                             'Fundulus heteroclitus_chronic',
                            'Gadus morhua_chronic',                                                      
                            'Gasterosteus aculeatus_acute',    
                            'Lepeophtheirus salmonis_acute',
                            'Lepomis macrochirus_acute',
                            'Litopenaeus vannamei_acute',
                            'Menidia beryllina_acute',                
                            'Menidia menidia_acute',
                            'Mytilus galloprovincialis_chronic',
                            'Nitocra spinipes_acute',
                            'Oncorhynchus mykiss_chronic',
                            'Oncorhynchus mykiss_acute',
                           'Oryzias melastigma_chronic',
                            'Palaemonetes pugio_acute',
                             'Palaemonetes pugio_chronic',  
                            'Palaemonetes sp._acute',
                            'Pimephales promelas_acute',
                            'Pimephales promelas_chronic',
                            'Raphidocelis subcapitata_acute',
                            'Rivulus marmoratus_chronic',
                            'Scolelepis fulginosa_acute',
                            'Skeletonema costatum_acute',
                                'Skeletonema costatum_chronic',                
                            'Dunaliella tertiolecta_acute',
                           'Isochrysis galbana_acute',
                            'Phaeodactylum tricornutum_acute']  # change
# generate select task index
for index, task in enumerate(args['all_task_list']):
    if task in args['select_task_list']:
        args['select_task_index'].append(index)
# generate classification_num
for task in args['select_task_list']:
    if task in []:
        args['classification_num'] = args['classification_num'] + 1
    if task in['Americamysis bahia_acute',
                            'Americamysis bahia_chronic',
                            'Artemia salina_acute',                
                            'Brachionus plicatilis_acute',
                           'Chaetogammarus marinus_acute',
                            'Crangon septemspinosa_acute',
                            'Crassostrea virginica_acute',                
                            'Cyprinodon variegatus_acute',
                           'Cyprinodon variegatus_chronic',
                            'Danio rerio_chronic',
                            'Farfantepenaeus duorarum_acute',                
                            'Fundulus heteroclitus_acute',
                             'Fundulus heteroclitus_chronic',
                            'Gadus morhua_chronic',                                                      
                            'Gasterosteus aculeatus_acute',    
                            'Lepeophtheirus salmonis_acute',
                            'Lepomis macrochirus_acute',
                            'Litopenaeus vannamei_acute',
                            'Menidia beryllina_acute',                
                            'Menidia menidia_acute',
                            'Mytilus galloprovincialis_chronic',
                            'Nitocra spinipes_acute',
                            'Oncorhynchus mykiss_chronic',
                            'Oncorhynchus mykiss_acute',
                           'Oryzias melastigma_chronic',
                            'Palaemonetes pugio_acute',
                             'Palaemonetes pugio_chronic',  
                            'Palaemonetes sp._acute',
                            'Pimephales promelas_acute',
                            'Pimephales promelas_chronic',
                            'Raphidocelis subcapitata_acute',
                            'Rivulus marmoratus_chronic',
                            'Scolelepis fulginosa_acute',
                            'Skeletonema costatum_acute',
                                'Skeletonema costatum_chronic',                
                            'Dunaliella tertiolecta_acute',
                           'Isochrysis galbana_acute',
                            'Phaeodactylum tricornutum_acute']:  # change
        args['regression_num'] = args['regression_num'] + 1

# generate classification_num
if args['classification_num'] != 0 and args['regression_num'] != 0:
    args['task_class'] = 'classification_regression'
if args['classification_num'] != 0 and args['regression_num'] == 0:
    args['task_class'] = 'classification'
if args['classification_num'] == 0 and args['regression_num'] != 0:
    args['task_class'] = 'regression'
print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))
args['bin_path'] = 'D:/data/' + args['data_name'] + '.bin'
args['group_path'] = 'D:/data/' + args['data_name'] + '_group.csv'
task_number = 38
