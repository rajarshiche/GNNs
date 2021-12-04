# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:24:44 2021

@author: Raj Guha
"""

import dgl
import dgllife
from dgllife.data import Tox21
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph, one_hot_encoding, RandomSplitter
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit.Chem import Draw 
import matplotlib.pyplot as plt 
import random

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))


    bg = dgl.batch(graphs)
    
    # print("The batched graph shape: ", len(graphs))
    
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    # labels = torch.stack(labels, dim=1)
    
    labels_collate = []
    for g in labels:
        labels_collate.append(g)
    
    
    print("The batched labels shape: ", len(labels))
    print("The batched labels_collate shape: ", len(labels_collate))


    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

def load_dataset_for_classification(args):
    """Load dataset for classification tasks.

    Parameters
    ----------
    args : dict
        Configurations.

    Returns
    -------
    dataset
        The whole dataset.
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Tox21']
    if args['dataset'] == 'Tox21':
        # from dgl.data.chem import Tox21 ## Older verson
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_bigraph, args['atom_featurizer'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set



    
def load_model(args):
    if args['model'] == 'GAT':
        model = model_zoo.gat_predictor.GATPredictor(in_feats=args['in_feats'],
                                              hidden_feats=args['gat_hidden_feats'],
                                              num_heads=args['num_heads'],
                                              # classifier_hidden_feats=args['classifier_hidden_feats'],
                                              predictor_hidden_feats=args['classifier_hidden_feats'],
                                              n_tasks=args['n_tasks'])
    return model




## args
args = {} 
args['dataset'] = 'Tox21' 
args['model'] = 'GAT' 
args['exp'] = 'GAT_Tox21' 
experimental_config = {
    'random_seed': 0,
    'batch_size': 128,  ## 'batch_size' = 1 will work as Target size [1,12] is then same as input size [1,12]
    'lr': 1e-3,
    'num_epochs': 10, ##
    'atom_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': 74,
    'gat_hidden_feats': [32, 32],
    'classifier_hidden_feats': 64,
    # 'classifier_hidden_feats': 128,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc'
    }
args.update(experimental_config) 



class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self, epoch_):
        """Compute roc-auc score for each task.

        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0) 
        y_pred = torch.sigmoid(y_pred) 
        #print('y_pred', y_pred.shape) 
        #print(y_pred) 
        predictions = np.round(y_pred) 
        #print(predictions) 
        #print('y_true', y_true.shape) 
        #print(y_true) 
        num_correct = np.where(predictions==y_true)[0] 
        true_task1 = y_true[:, 0] 
        pred_task1 = predictions[:, 0] 


        # Creating the histogram for plotting the # of tasks correctly 
        # predicted for each molecule 

        indicators = np.zeros((783, 12))  
        for i in range(783): 
            for j in range(12): 
                if(predictions[i, j] == y_true[i, j]): 
                    indicators[i, j] = 1 
                else: 
                    indicators[i, j] = 0 
        #print("indictators", len(indicators))  
        #print(indicators) 
        summed = np.sum(indicators, axis = 1) 
        #print("summed", len(summed)) 
        #print(summed) 
        dict1 = {} 
        dict2 = {} 
        index = -1 
        for element in summed: 
            index += 1 
            key = element  
            if key in dict1:
                dict1[key] += 1
            else:
                dict1[key] = 1 
            if key in dict2: 
                dict2[key].append(index) 
            else: 
                dict2[key] = [index] 
        print(dict1) 
        plt.bar(dict1.keys(), dict1.values(), 1.0, color='g')
        filename = 'hist_' + str(epoch_) + '.png' 
        plt.savefig(filename) 

        # Examples molecules for 3/12, 6/12, 9/12, 12/12 buckets
        list3 = [] 
        list6 = [] 
        list9 = [] 
        list12 = [] 
        examples3 = [] 
        examples6 = [] 
        examples9 = [] 
        examples12 = [] 
        if 3 in dict2: 
            list3 = dict2[3] 
            if(len(list3) >= 5): 
                examples3 = random.sample(list3, 5) 
        if 6 in dict2: 
            list6 = dict2[6] 
            if(len(list6) >= 5): 
                examples6 = random.sample(list6, 5) 
        if 9 in dict2: 
            list9 = dict2[9] 
            if(len(list9) >= 5): 
                examples9 = random.sample(list9, 5) 
        if 12 in dict2: 
            list12 = dict2[12] 
            if(len(list12) >= 5): 
                examples12 = random.sample(list12, 5)         

        # ROC-AUC scores 
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred)) 
        #print(scores) 
        return scores, examples3, examples6, examples9, examples12 

    def compute_metric(self, metric_name, epoch_=0, reduction='mean'): 
        """Compute metric for each task.

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task

        Returns
        -------
        list of float
            Metric value for each task
        """
        if metric_name == 'roc_auc':
            return self.roc_auc_score(epoch_) 

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    
    model.eval()
    
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        

        # print("Input batch_data smiles size:{}".format(len(batch_data[0])) )
        # print("Input batch_data bg size:{}".format( (batch_data[1])) )
        # print("Input batch_data label size:{}".format(batch_data[2].shape) )
        # print("Input batch_data mask size:{}".format(batch_data[3].shape) )
        
        
        bg = dgl.add_self_loop(bg)  ## Added to ward off self-loop problem
        ##allow_zero_in_degree = True
        
        atom_feats = bg.ndata.pop(args['atom_data_field'])
        # print("Input batch atom_feats size:{}".format(atom_feats.shape) )
        
        # atom_feats, labels, masks = atom_feats.to(args['device']), \
        #                             labels.to(args['device']), \
        #                             masks.to(args['device'])
                                       
                    
        logits = model(bg, atom_feats)  #### The vector of raw (non-normalized) predictions that a classification model generates => Logits are values that are used as input to softmax
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        train_meter.update(logits, labels, masks) 
    # Separated stuff here 
    scores, examples3, examples6, examples9, examples12 = train_meter.compute_metric(args['metric_name']) 
    train_score = np.mean(scores) 
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], train_score))
    return loss.item() 


def run_an_eval_epoch(args, model, data_loader, epoch, last): 
    model.eval()
    eval_meter = Meter() 
    listOfMolSmiles = [] # 
    with torch.no_grad(): 
        for batch_id, batch_data in enumerate(data_loader): 
            smiles, bg, labels, masks = batch_data 
            
            bg = dgl.add_self_loop(bg)  ## Added to ward off self-loop problem
            ##allow_zero_in_degree = True
        
            for smile in smiles: 
                listOfMolSmiles.append(smile) 
            #print('labels', len(labels)) 
            #print(labels) 
            atom_feats = bg.ndata.pop(args['atom_data_field'])
            atom_feats, labels = atom_feats.to(args['device']), labels.to(args['device'])
            logits = model(bg, atom_feats) 
            #print(len(logits)) 
            #print(logits) 
            eval_meter.update(logits, labels, masks)
    # This is the val_score 
    scores, examples3, examples6, examples9, examples12 = eval_meter.compute_metric(args['metric_name'], epoch_ = epoch) 
    print('scores') 
    print(scores) 

    print('examples3', len(examples3)) 
    print(examples3) 
    print('examples6', len(examples6)) 
    print(examples6) 
    print('examples9', len(examples6)) 
    print(examples9) 
    print('examples12', len(examples6)) 
    print(examples12) 

    print('listOfMolSmiles', len(listOfMolSmiles)) 


    exampleMols = [] 
    for ex in examples3: 
        exampleMols.append(listOfMolSmiles[ex]) 
    for ex in examples6: 
        exampleMols.append(listOfMolSmiles[ex]) 
    for ex in examples9: 
        exampleMols.append(listOfMolSmiles[ex]) 
    for ex in examples12: 
        exampleMols.append(listOfMolSmiles[ex]) 
    
    flags = [True, True, True, True] 
    if(len(examples3) == 0): 
        flags[0] = False 
    if(len(examples6) == 0): 
        flags[0] = False 
    if(len(examples9) == 0): 
        flags[0] = False 
    if(len(examples12) == 0): 
        flags[0] = False 

    if(last == True): ## 
        print(flags) 
        for i in range(len(exampleMols)): 
            smiles = exampleMols[i] 
            m = Chem.MolFromSmiles(smiles) 
            fileName = 'test_' + str(epoch) + '.' + str(i) + '.png' 
            Draw.MolToFile(m, fileName) 

    return np.mean(scores) 
    #return np.mean(eval_meter.compute_metric(args['metric_name'])) 


## TRAIN
args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set_random_seed(args['random_seed'])

dataset, train_set, val_set, test_set = load_dataset_for_classification(args)


train_loader = DataLoader(train_set, batch_size=args['batch_size'],collate_fn=collate_molgraphs,drop_last=True)
val_loader = DataLoader(val_set, batch_size=args['batch_size'],collate_fn=collate_molgraphs,drop_last=True)
test_loader = DataLoader(test_set, batch_size=args['batch_size'],collate_fn=collate_molgraphs,drop_last=True)

args['n_tasks'] = dataset.n_tasks  ## dataset.n_tasks =  12 => # of outputs/ classes
model = load_model(args)

#### loss_criterion = BCEWithLogitsLoss(reduction='none')  ###nn.BCEWihtLogitsLoss expects the model output and target to have the same shape.
### BCEWithLogitsLoss is only used for binary classification (2 classes) for >2 classes you need to usenn.CrossEntropyLoss
###loss_criterion = nn.CrossEntropyLoss(reduction='none')
loss_criterion =  torch.nn.BCEWithLogitsLoss()
    

optimizer = Adam(model.parameters(), lr=args['lr'])
# stopper = EarlyStopping(patience=args['patience']) ## Early stopping disabled
model.to(args['device']) 
    

epochx = 0 
losses = [] 

for epoch in range(args['num_epochs']): 
    # Train
    loss = run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
    losses.append(loss) 

    # Validation and early stop
    epochx += 1 
    val_score = run_an_eval_epoch(args, model, val_loader, epochx, False)  
    print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} '.format(epoch + 1, args['num_epochs'], args['metric_name'],val_score, args['metric_name']))
    # early_stop = stopper.step(val_score, model)
    # print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(epoch + 1, args['num_epochs'], args['metric_name'],val_score, args['metric_name'], stopper.best_score))
        # if early_stop:
        #     break

    # stopper.load_checkpoint(model)


# Print out the test set score 
 
test_score = run_an_eval_epoch(args, model, test_loader, epochx, True)
print('test {} {:.4f}'.format(args['metric_name'], test_score))

# Making the loss per epoch figure 
#print('losses', len(losses)) 
print(losses) 
epoch_list = [i+1 for i in range(len(losses))] ## 
plt.clf() 
plt.plot(epoch_list, losses) 
plt.xlabel("# of Epochs")
plt.ylabel("Loss") 
plt.rcParams['axes.facecolor'] = 'white'
plt.savefig("Loss.Per.Epoch.png") 


