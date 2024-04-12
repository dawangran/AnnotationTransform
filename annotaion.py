import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def get_shared_genes(adata1, adata2):
    genes1 = set(adata1.var_names)
    genes2 = set(adata2.var_names)
    shared_genes = list(genes1.intersection(genes2))
    return shared_genes

def train_scRNA_model(fileLocation, predictType, shared_genes,saveLocation):
    adata = sc.read(fileLocation)
    adata.var_names_make_unique()
    labels = adata.obs[predictType]

    indata = adata[:, shared_genes].X
    labels = np.array(labels)

    if isinstance(indata, pd.DataFrame):
        indata = indata.values

    le = LabelEncoder()
    y_data = le.fit_transform(labels)

    rclf = RandomForestClassifier(n_estimators=116, n_jobs=-1, max_depth=10)
    rclf.fit(indata, y_data)
    pickle.dump(rclf,open(saveLocation,"wb"))

    return rclf

def predict_scATAC_data(scATAC_obj, model, shared_genes,labels):
    genes = scATAC_obj.var_names

    if isinstance(model, str):
        loaded_model = pickle.load(open(model, "rb"))
    else:
        loaded_model = model

    selected_genes = set(shared_genes)

    data = scATAC_obj[:, shared_genes].X

    if isinstance(data, pd.DataFrame):
        data = data.values

    predictions = loaded_model.predict(data)
    
    le = LabelEncoder()
    le.fit(labels)  # Fit the encoder on the original labels

    predicted_labels = le.inverse_transform(predictions)  # Convert encoded predictions back to original labels
    
    scATAC_obj.obs['predicted_labels'] = predicted_labels
    
    df=scATAC_obj.obs[[labels,'predicted_labels']]
    df=df.loc[df.groupby(labels)['predicted_labels'].idxmax()]
    label={k:v for k,v in zip(df[labels].to_list(),df['predicted_labels'].to_list())}
    scATAC_obj.obs['majority_voting']=None
    for k,v in label.items():  
        scATAC_obj.obs.loc[scATAC_obj.obs[labels]==k,'majority_voting']=v
    return scATAC_obj