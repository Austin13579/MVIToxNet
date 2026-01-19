import pandas as pd
from rdkit import Chem
import json
import numpy as np

from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys


datas=['Rat','Rabbit']
for data in datas:
    print(data)
    df1 = pd.read_csv('../dataset/' + data + '.csv')
    df2 = pd.read_csv('../dataset/' + data + '_external.csv')
    df=pd.concat([df1, df2]).reset_index()
    smile_dic={}

    for i,smile in df.SMILES.items():
        print(i)
        mol = Chem.MolFromSmiles(smile)
        fp_path=data+str(i)

        # ECFP4 + MACCS
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048).ToList()
        fp2 = MACCSkeys.GenMACCSKeys(mol).ToList()

        fp = np.concatenate((fp1, fp2))
        np.save('fps/' + fp_path + '.npy', fp)

        smile_dic[smile]=fp_path

    with open('fps/' + data + '.json', 'w', encoding='utf-8') as f:
        json.dump(smile_dic, f, ensure_ascii=False, indent=4)
