import logging
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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
    fileLocation (str or AnnData): Path to the scRNA-seq data file or an AnnData object.
    predictType (str): The column name in .obs containing the labels.
    shared_genes (list): List of shared genes.
    saveLocation (str): Path to save the trained model.
    **kwargs: Additional parameters for RandomForestClassifier.

    Returns:
    RandomForestClassifier: The trained RandomForest model.
    """
    logging.info(f"Reading scRNA-seq data from {fileLocation}.")

    # Validate fileLocation
    if isinstance(fileLocation, str):
        try:
            adata = sc.read(fileLocation)
        except Exception as e:
            logging.error(f"Failed to read scRNA-seq data from {fileLocation}: {e}")
            raise
    elif isinstance(fileLocation, sc.AnnData):
        adata = fileLocation
    else:
        logging.error("fileLocation should be a string or an AnnData object.")
        raise TypeError("fileLocation should be a string or an AnnData object.")
    
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

    if  isinstance(adata[:, shared_genes].X, np.ndarray):
        indata = adata[:, shared_genes].X
    else:
        indata = adata[:, shared_genes].X.A

    # Ensure indata is an array
    if isinstance(indata, np.ndarray):
        indata = indata
    else:
        logging.error("indata should be a numpy array.")
        raise TypeError("indata should be a numpy array.")

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


def predict_scATAC_data(scATAC_obj, model, shared_genes, scRNA_obj,labels,majority_voting=None):
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
    if isinstance(scATAC_obj[:, shared_genes].X, np.ndarray):
        data = scATAC_obj[:, shared_genes].X
    else:
        data = scATAC_obj[:, shared_genes].X.A
        
    # Ensure data is an array
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        logging.error("data should be a numpy array or pandas DataFrame.")
        raise TypeError("data should be a numpy array or pandas DataFrame.")

    # Validate label
    if labels not in scRNA_obj.obs.columns:
        logging.error(f"{labels} is not found in .obs columns.")
        raise ValueError(f"{labels} is not found in .obs columns.")
    labels_list=scRNA_obj.obs[labels].to_list()
    
    logging.info("Predicting cell types.")
    try:
        predictions = loaded_model.predict(data)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
    
    le = LabelEncoder()
    le.fit(labels_list)  # Fit the encoder on the original labels

    try:
        predicted_labels = le.inverse_transform(predictions)  # Convert encoded predictions back to original labels
    except Exception as e:
        logging.error(f"Label inverse transformation failed: {e}")
        raise
    
    scATAC_obj.obs['predicted_labels'] = predicted_labels

    if majority_voting:
    
        logging.info("Performing majority voting for label consistency.")
        try:
            df=scATAC_obj.obs[['leiden','predicted_labels']]
            grouped_counts = df.groupby('leiden')['predicted_labels'].value_counts()
            max_count_elements = grouped_counts.groupby('leiden').idxmax()
            label={k:v for k,v in max_count_elements}
            scATAC_obj.obs['majority_voting'] = None
            for k, v in label.items():
                scATAC_obj.obs.loc[scATAC_obj.obs[majority_voting] == k, 'majority_voting'] = v
        except Exception as e:
            logging.error(f"Majority voting failed: {e}")
            raise

    return scATAC_obj

def evaluate_predictions(true_labels, predicted_labels):
    logging.info("Evaluating predictions.")
    
    # 验证输入参数
    if not isinstance(true_labels, (list, tuple, np.ndarray)):
        logging.error("true_labels must be a list, tuple, or numpy array.")
        raise TypeError("true_labels must be a list, tuple, or numpy array.")
    
    if not isinstance(predicted_labels, (list, tuple, np.ndarray)):
        logging.error("predicted_labels must be a list, tuple, or numpy array.")
        raise TypeError("predicted_labels must be a list, tuple, or numpy array.")
    
    if len(true_labels) != len(predicted_labels):
        logging.error("true_labels and predicted_labels must have the same length.")
        raise ValueError("true_labels and predicted_labels must have the same length.")

    # 计算评价指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm = confusion_matrix(true_labels, predicted_labels)
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
    
    return results


def plot_confusion_matrix(cm,title='Confusion Matrix'):
    """
    Plot the confusion matrix.

    Parameters:
    cm (array-like): Confusion matrix.
    labels (list): List of labels.
    title (str): Title of the plot.
    """
    # 验证输入参数
    if not isinstance(cm, (np.ndarray, list)):
        logging.error("Confusion matrix must be a numpy array or a list.")
        raise TypeError("Confusion matrix must be a numpy array or a list.")
    
    cm = np.array(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        logging.error("Confusion matrix must be a square matrix.")
        raise ValueError("Confusion matrix must be a square matrix.")
    
    logging.info("Plotting confusion matrix.")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()
    
def plot_umap(sc_obj, color_by='predicted_labels', title='UMAP Plot'):
    """
    Plot UMAP for the given scRNA-seq or scATAC-seq data.

    Parameters:
    sc_obj (AnnData): The data object to plot.
    color_by (str): The column in .obs to color by.
    title (str): Title of the plot.
    """
    # 验证输入参数
    if not isinstance(sc_obj, sc.AnnData):
        logging.error("The data object must be an instance of AnnData.")
        raise TypeError("The data object must be an instance of AnnData.")
    
    if not isinstance(color_by, str):
        logging.error("The color_by parameter must be a string.")
        raise TypeError("The color_by parameter must be a string.")
    
    if color_by not in sc_obj.obs.columns:
        logging.error(f"The column '{color_by}' is not present in the .obs attribute of the data object.")
        raise ValueError(f"The column '{color_by}' is not present in the .obs attribute of the data object.")
    
    if not isinstance(title, str):
        logging.error("The title must be a string.")
        raise TypeError("The title must be a string.")

    logging.info("Plotting UMAP.")
    
    # 绘制UMAP
    sc.pl.umap(sc_obj, color=color_by, title=title)
    plt.show()
    
def plot_tsne(sc_obj, color_by='predicted_labels', title='t-SNE Plot'):
    """
    Plot t-SNE for the given scRNA-seq or scATAC-seq data.

    Parameters:
    sc_obj (AnnData): The data object to plot.
    color_by (str): The column in .obs to color by.
    title (str): Title of the plot.
    """
    # 验证输入参数
    if not isinstance(sc_obj, sc.AnnData):
        logging.error("The data object must be an instance of AnnData.")
        raise TypeError("The data object must be an instance of AnnData.")
    
    if not isinstance(color_by, str):
        logging.error("The color_by parameter must be a string.")
        raise TypeError("The color_by parameter must be a string.")
    
    if color_by not in sc_obj.obs.columns:
        logging.error(f"The column '{color_by}' is not present in the .obs attribute of the data object.")
        raise ValueError(f"The column '{color_by}' is not present in the .obs attribute of the data object.")
    
    if not isinstance(title, str):
        logging.error("The title must be a string.")
        raise TypeError("The title must be a string.")

    logging.info("Plotting t-SNE.")
    
    # 计算t-SNE
    sc.tl.tsne(sc_obj)
    
    # 绘制t-SNE
    sc.pl.tsne(sc_obj, color=color_by, title=title)
    plt.show()
    
def plot_pca(sc_obj, color_by='predicted_labels', title='PCA Plot'):
    """
    Plot PCA for the given scRNA-seq or scATAC-seq data.

    Parameters:
    sc_obj (AnnData): The data object to plot.
    color_by (str): The column in .obs to color by.
    title (str): Title of the plot.
    """
    # 验证输入参数
    if not isinstance(sc_obj, sc.AnnData):
        logging.error("The data object must be an instance of AnnData.")
        raise TypeError("The data object must be an instance of AnnData.")
    
    if not isinstance(color_by, str):
        logging.error("The color_by parameter must be a string.")
        raise TypeError("The color_by parameter must be a string.")
    
    if color_by not in sc_obj.obs.columns:
        logging.error(f"The column '{color_by}' is not present in the .obs attribute of the data object.")
        raise ValueError(f"The column '{color_by}' is not present in the .obs attribute of the data object.")
    
    if not isinstance(title, str):
        logging.error("The title must be a string.")
        raise TypeError("The title must be a string.")

    logging.info("Plotting PCA.")

    # 绘制PCA
    sc.pl.pca(sc_obj, color=color_by, title=title)
    plt.show()