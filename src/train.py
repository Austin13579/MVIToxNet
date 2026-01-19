import pandas as pd
import argparse
import torch
import numpy as np
import random
from sklearn.utils import class_weight
import copy

from model import MVIToxNet
from utils import Data_Encoder
from metrics import evaluate
from sklearn.metrics import f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# training function at each epoch
def train(model, loader, optimizer):
    print('Training on {} samples...'.format(len(loader.dataset)))
    model.train()
    losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    for batch_idx, (fp,seq1,seq2,seq3,label) in enumerate(loader):
        optimizer.zero_grad()

        output = model(fp,seq1,seq2,seq3)
        score=torch.squeeze(output)
        loss = loss_fn(score, label.float())

        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

        total_preds = torch.cat((total_preds, score.cpu()), 0)
        total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return np.mean(losses),total_labels.numpy().flatten(), total_preds.detach().numpy().flatten()


def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (fp, seq1,seq2,seq3, label) in enumerate(loader):
            output= model(fp,seq1,seq2,seq3)

            score = torch.squeeze(torch.sigmoid(output))
            total_preds = torch.cat((total_preds, score.cpu()), 0)
            total_labels = torch.cat((total_labels, label.flatten().cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default='Rat', help='which dataset')
    parser.add_argument('--rs', type=int, default=0, help='which random seed')
    args = parser.parse_args()
    setup_seed(18) # atom 19
    print(args.ds)

    # Read datasets
    train_path='../dataset/datas/'+args.ds+'_train'+str(args.rs)+'.csv'
    train_df = pd.read_csv(train_path)

    valid_path='../dataset/datas/'+args.ds+'_valid'+str(args.rs)+'.csv'
    valid_df = pd.read_csv(valid_path)

    external_path='../dataset/'+args.ds+'_external.csv'
    external_df = pd.read_csv(external_path)

    tox_model=MVIToxNet()

    batch_size = 64
    LR = 1e-5
    NUM_EPOCHS = 50
    
    train_set = Data_Encoder(train_df.index.values, train_df,args.ds)
    valid_set = Data_Encoder(valid_df.index.values, valid_df,args.ds)
    external_set = Data_Encoder(external_df.index.values, external_df,args.ds)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    external_loader = torch.utils.data.DataLoader(external_set, batch_size=batch_size, shuffle=False)

    train_label=train_df.Label
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_label),y=train_label)
    class_weights_dict = dict(enumerate(class_weights))
    
    p_weight=class_weights_dict[1]/class_weights_dict[0]
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([p_weight]))
    optim = torch.optim.Adam(tox_model.parameters(), lr=LR)

    best_auc=0
    res={}
    val,tmp=[],[]
    val2=[]
    for epoch in range(NUM_EPOCHS):
        print("Epoch: ", epoch + 1)
        train_loss,aa,bb = train(tox_model, train_loader, optim)
        print("Train Loss: ", train_loss)

        print("Validation")
        valid_true, valid_prob = predicting(tox_model, valid_loader)
        v_auprc = evaluate(valid_true, valid_prob)['AUPRC']
        val.append(v_auprc)
        print("Val AUPRC: ", v_auprc)


        test_true, test_prob = predicting(tox_model, external_loader)
        res=evaluate(test_true, test_prob)
        tmp.append((res, copy.deepcopy(tox_model.state_dict())))
        print(res)
    

    # Save the results of the best model
    best_val, best_index = torch.topk(torch.tensor(np.array(val)), k=1)
    best_res=tmp[best_index][0]
    
    print(best_index)
    
    try:
        val_idx=best_index.tolist()
        _, v_idx = torch.topk(torch.tensor(np.array(val)[best_index+1:]), k=2)
        for v in v_idx.tolist():
            val_idx.append(v+best_index.tolist()[0]+1)
    except:
        _, val_idx = torch.topk(torch.tensor(np.array(val)), k=3)
    
    best_queue = []
    for idx in val_idx:
        print(idx)
        best_queue.append(tmp[idx])


    new_model=MVIToxNet()
    weighted_params = {}
    for key in best_queue[0][1].keys():
        weighted_params[key] = torch.zeros_like(best_queue[0][1][key])+1e-10

    ww=0
    if args.data=='Rat':
        ww=0.4
    elif args.data=='Rabbit':
        ww=0.2
    weight=[1-2*ww,ww,ww]

    for i,(res,state_dict) in enumerate(best_queue):
        print(res)
        for key in state_dict:
            weighted_params[key] += weight[i]*state_dict[key]

    for key in best_queue[0][1].keys():
        weighted_params[key] -= 1e-10


    new_model.load_state_dict(weighted_params)
    test_true, test_prob = predicting(new_model, external_loader)


    wma_res = evaluate(test_true, test_prob)
    print("WMA:",wma_res)
    print('results_'+model_type+'/wma'+'_' + args.ds + str(args.rs) + '.csv')
    pd.DataFrame([wma_res]).to_csv('results_'+model_type+'/wma'+'_' + args.ds + str(args.rs) + '.csv',columns=['Accuracy', 'AUROC', 'AUPRC', 'Recall', 'F1', 'MCC'], index=False)

