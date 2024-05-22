import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_shared_genes(adata1, adata2):
    """
    Find and return the shared genes between two AnnData objects.

    Parameters:
    adata1 (AnnData): The first AnnData object.
    adata2 (AnnData): The second AnnData object.

    Returns:
    list: A list of shared gene names.
    """
    logging.info("Identifying shared genes between two datasets.")
    
    # Validate input types
    if not isinstance(adata1, sc.AnnData):
        logging.error("adata1 is not an AnnData object.")
        raise TypeError("adata1 must be an AnnData object.")
        
    if not isinstance(adata2, sc.AnnData):
        logging.error("adata2 is not an AnnData object.")
        raise TypeError("adata2 must be an AnnData object.")
    
    # Validate var_names attribute
    if not hasattr(adata1, 'var_names') or not hasattr(adata2, 'var_names'):
        logging.error("One of the AnnData objects does not have 'var_names' attribute.")
        raise AttributeError("Both AnnData objects must have 'var_names' attribute.")
    
    # Validate non-empty var_names
    if len(adata1.var_names) == 0 or len(adata2.var_names) == 0:
        logging.error("One of the AnnData objects has an empty 'var_names' list.")
        raise ValueError("Both AnnData objects must have non-empty 'var_names' lists.")
    
    genes1 = set(adata1.var_names)
    genes2 = set(adata2.var_names)
    
    shared_genes = list(genes1.intersection(genes2))
    
    if len(shared_genes) == 0:
        logging.warning("No shared genes found between the two datasets.")
    else:
        logging.info(f"Found {len(shared_genes)} shared genes out of {len(genes1)} in adata1 and {len(genes2)} in adata2.")
    
    return shared_genes


def train_scRNA_model(fileLocation, predictType, shared_genes, saveLocation, **kwargs):
    """
    Train a RandomForest model on scRNA-seq data and save the model.

    Parameters:
    fileLocation (str): Path to the scRNA-seq data file.
    predictType (str): The column name in .obs containing the labels.
    shared_genes (list): List of shared genes.
    saveLocation (str): Path to save the trained model.
    **kwargs: Additional parameters for RandomForestClassifier.

    Returns:
    RandomForestClassifier: The trained RandomForest model.
    """
    logging.info(f"Reading scRNA-seq data from {fileLocation}.")
    
    # Validate fileLocation
    if not isinstance(fileLocation, str):
        logging.error("fileLocation should be a string.")
        raise TypeError("fileLocation should be a string.")
    
    try:
        adata = sc.read(fileLocation)
    except Exception as e:
        logging.error(f"Failed to read scRNA-seq data from {fileLocation}: {e}")
        raise
    
    adata.var_names_make_unique()

    # Validate predictType
    if predictType not in adata.obs.columns:
        logging.error(f"{predictType} is not found in .obs columns.")
        raise ValueError(f"{predictType} is not found in .obs columns.")
    
    labels = adata.obs[predictType]

    # Validate shared_genes
    if not isinstance(shared_genes, list):
        logging.error("shared_genes should be a list.")
        raise TypeError("shared_genes should be a list.")
    
    missing_genes = [gene for gene in shared_genes if gene not in adata.var_names]
    if missing_genes:
        logging.error(f"Missing genes in the dataset: {missing_genes}")
        raise ValueError(f"Missing genes in the dataset: {missing_genes}")

    indata = adata[:, shared_genes].X

    # Ensure indata is an array
    if isinstance(indata, pd.DataFrame):
        indata = indata.values
    elif not isinstance(indata, np.ndarray):
        logging.error("indata should be a numpy array or pandas DataFrame.")
        raise TypeError("indata should be a numpy array or pandas DataFrame.")

    labels = np.array(labels)

    le = LabelEncoder()
    y_data = le.fit_transform(labels)

    logging.info("Training RandomForest model.")
    
    # Default parameters
    default_params = {
        'n_estimators': 116,
        'n_jobs': -1,
        'max_depth': 10
    }
    
    # Update default parameters with any user-provided parameters
    default_params.update(kwargs)

    # Initialize RandomForestClassifier with the final parameters
    rclf = RandomForestClassifier(**default_params)
    rclf.fit(indata, y_data)
    
    logging.info(f"Saving the trained model to {saveLocation}.")
    
    # Validate saveLocation
    if not isinstance(saveLocation, str):
        logging.error("saveLocation should be a string.")
        raise TypeError("saveLocation should be a string.")
    
    try:
        with open(saveLocation, "wb") as model_file:
            pickle.dump(rclf, model_file)
    except Exception as e:
        logging.error(f"Failed to save the model to {saveLocation}: {e}")
        raise
    
    return rclf


def predict_scATAC_data(scATAC_obj, model, shared_genes, labels):
    """
    Predict cell types for scATAC-seq data using a trained model.

    Parameters:
    scATAC_obj (AnnData): The scATAC-seq data object.
    model (str or RandomForestClassifier): The path to the trained model or the model object.
    shared_genes (list): List of shared genes.
    labels (str): The column name in .obs containing the original labels.

    Returns:
    AnnData: The scATAC-seq data object with added predictions.
    """
    logging.info("Loading model for prediction.")
    
    # Validate scATAC_obj
    if not isinstance(scATAC_obj, sc.AnnData):
        logging.error("scATAC_obj should be an AnnData object.")
        raise TypeError("scATAC_obj should be an AnnData object.")
    
    # Validate model
    if isinstance(model, str):
        try:
            with open(model, "rb") as model_file:
                loaded_model = pickle.load(model_file)
        except Exception as e:
            logging.error(f"Failed to load the model from {model}: {e}")
            raise
    elif hasattr(model, 'predict'):
        loaded_model = model
    else:
        logging.error("model should be a path to a pickled model or a model object with a predict method.")
        raise TypeError("model should be a path to a pickled model or a model object with a predict method.")
    
    # Validate shared_genes
    if not isinstance(shared_genes, list):
        logging.error("shared_genes should be a list.")
        raise TypeError("shared_genes should be a list.")
    
    missing_genes = [gene for gene in shared_genes if gene not in scATAC_obj.var_names]
    if missing_genes:
        logging.error(f"Missing genes in the dataset: {missing_genes}")
        raise ValueError(f"Missing genes in the dataset: {missing_genes}")

    logging.info("Selecting shared genes for prediction.")
    data = scATAC_obj[:, shared_genes].X

    # Ensure data is an array
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        logging.error("data should be a numpy array or pandas DataFrame.")
        raise TypeError("data should be a numpy array or pandas DataFrame.")

    # Validate labels
    if labels not in scATAC_obj.obs.columns:
        logging.error(f"{labels} is not found in .obs columns.")
        raise ValueError(f"{labels} is not found in .obs columns.")
    
    original_labels = scATAC_obj.obs[labels].values

    logging.info("Predicting cell types.")
    try:
        predictions = loaded_model.predict(data)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
    
    le = LabelEncoder()
    le.fit(original_labels)  # Fit the encoder on the original labels

    try:
        predicted_labels = le.inverse_transform(predictions)  # Convert encoded predictions back to original labels
    except Exception as e:
        logging.error(f"Label inverse transformation failed: {e}")
        raise
    
    scATAC_obj.obs['predicted_labels'] = predicted_labels
    
    logging.info("Performing majority voting for label consistency.")
    try:
        df = scATAC_obj.obs[[labels, 'predicted_labels']]
        df = df.loc[df.groupby(labels)['predicted_labels'].idxmax()]
        label = {k: v for k, v in zip(df[labels].to_list(), df['predicted_labels'].to_list())}
        scATAC_obj.obs['majority_voting'] = None
        for k, v in label.items():
            scATAC_obj.obs.loc[scATAC_obj.obs[labels] == k, 'majority_voting'] = v
    except Exception as e:
        logging.error(f"Majority voting failed: {e}")
        raise

    return scATAC_obj


