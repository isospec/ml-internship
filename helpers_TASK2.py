import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Machine Learning methods
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import umap
from sklearn.manifold import TSNE

# Glycowork
from glycowork.motif.processing import min_process_glycans

def combine_and_normalize_embeddings(embedding_list, index):
    """
    Concatenates and normalizes a list of glycan embeddings.
    
    Parameters
    ----------
    embedding_list : list of np.ndarray
        List of embedding arrays (e.g., from sequence, composition, tissue, etc.).
        Each array must have the same number of rows (one per glycan).
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the combined and normalized embedding matrix, ready for learning.
    """
    
    # Concatenate all embeddings horizontally (column-wise)
    emb_combined = np.hstack(embedding_list)
    
    # Normalize all features to zero mean and unit variance
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb_combined)
    
    emb_scaled_df = pd.DataFrame(emb_scaled, index=index)
    
    return emb_scaled_df


def evaluate_embedding_sil_score(embedding_df, n_glycans_df, label_col='N-glycan'):
    """
    Evaluates the quality of an embedding space using the silhouette score based on N-glycan clustering.
    
    Parameters
    ----------
    embedding_df : pd.DataFrame
        DataFrame containing the embedding matrix with glycan sequences as index.
        
    n_glycans_df : pd.DataFrame
        DataFrame containing a column 'glycan' listing the known N-glycans.
    
    label_col : str, optional
        Name to assign to the label column in the embedding matrix (default is 'N-glycan').

    Returns
    -------
    float
        Silhouette score indicating the quality of separation between N-glycans and other glycans.
    """
    # Create binary label vector: 1 for N-glycan, 0 for others
    n_glycan_set = set(n_glycans_df['glycan'])
    labels_binary = [1 if glycan in n_glycan_set else 0 for glycan in embedding_df.index]
    
    # Copy and attach the label vector
    temp = embedding_df.copy()
    temp[label_col] = labels_binary
    
    # Check for NaN values
    nan_count = int(temp.isna().sum().sum())
    
    if nan_count != 0:
        temp.dropna()
        
    else:
        # Define labels and features
        labels = temp[label_col]
        features = temp.drop(columns=[label_col])
            
        sil_score = silhouette_score(features, labels)
    
    return round(sil_score, 3)

def evaluate_embedding_nn_purity(embedding_df, n_glycans_df, label_col='N-glycan', k=5):
    """
    Evaluates the quality of an embedding space using nearest-neighbor purity for N-glycans.
    
    Parameters
    ----------
    embedding_df : pd.DataFrame
        DataFrame containing the embedding matrix with glycan sequences as index.
    
    n_glycans_df : pd.DataFrame
        DataFrame containing a column 'glycan' listing the known N-glycans.
    
    label_col : str, optional
        Name to assign to the label column in the embedding matrix (default is 'N-glycan').
        
    k : int, optional
        Number of nearest neighbors to consider (default is 5).
    
    Returns
    -------
    float
        Nearest-neighbor purity score for N-glycans.
    """
    # Create binary label vector: 1 for N-glycan, 0 for others
    n_glycan_set = set(n_glycans_df['glycan'])
    labels_binary = [1 if glycan in n_glycan_set else 0 for glycan in embedding_df.index]
    
    # Copy and attach the label column
    temp = embedding_df.copy()
    temp[label_col] = labels_binary

    # Define labels and features
    labels = temp[label_col].values
    features = temp.drop(columns=[label_col]).values

    # Fit nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k + 1).fit(features)  # +1 to skip self
    _, indices = nn.kneighbors(features)

    # Calculate purity for each N-glycan
    purities = []
    for i, is_nglycan in enumerate(labels):
        if is_nglycan == 1:
            neighbors = indices[i][1:]  # Exclude self
            purity = np.mean(labels[neighbors])
            purities.append(purity)

    # Average purity over all N-glycans
    nn_purity = np.mean(purities) if purities else None

    return round(nn_purity, 3)

def one_hot_embedding(feature_name, glycan_df):
    """
    Compute a one-hot embedding for a given glycan feature.

    Parameters
    ----------
    feature_name : str
        Feature to embed ('Composition', 'Tissue', 'Species', or 'Disease').

    glycan_df : pd.DataFrame
        Full glycan dataset.

    Returns
    -------
    emb_df : pd.DataFrame
        Glycan embeddings as one-hot encoded features.

    time_emb : float
        Time taken to compute the embedding (in seconds).

    model : str
        Name of the embedding model ('One-Hot').
    """
    time_emb = 0
    if feature_name == 'Sequence':
        print('One-Hot not applicable for Sequence.')
    
    elif feature_name == 'Composition':
        # Get all unique monosaccharies across the dataset
        monosaccharides = set()
        t0 = time.time()
        for composition in glycan_df['Composition']:
            monosaccharides.update(composition.keys())

        # Create a DataFrame with monosaccharide counts
        emb_composition = pd.DataFrame([
            {mono: composition.get(mono, 0) for mono in monosaccharides}
            for composition in glycan_df['Composition']
        ], index=glycan_df.index)
        
        time_emb += round(time.time() - t0, 2) # take into account the time it takes
        emb_df = emb_composition.copy()
    
    elif feature_name == 'Tissue':
        # Get the list of unique tissues
        seen = set() # to track seen species
        unique_tissues = [] # list of unique species

        t0 = time.time()
        for tissue_sample in glycan_df['tissue_sample']:
            for tissue in tissue_sample:
                if tissue not in seen:
                    unique_tissues.append(tissue)
                    seen.add(tissue)
                    
        # Create a matrix of with rows = glycans, columns = tissues, 
        # values = 1 if glycan is associated to that tissue, 0 if not
        emb_tissues = pd.DataFrame(0, index=glycan_df.index, columns=unique_tissues)

        # Fill matrix with 1 where species is present for each glycan
        for idx, tissues_list in glycan_df['tissue_sample'].items():
            for tissue in tissues_list:
                if tissue in emb_tissues.columns:
                    emb_tissues.at[idx, tissue] = 1

        time_emb += round(time.time() - t0, 2) # take into account the time it takes
                    
        emb_df = emb_tissues.copy()
    
    elif feature_name == 'Species':
        # Get the list of unique species
        seen = set() # to track seen species
        unique_species = [] # list of unique species

        t0 = time.time()
        for tissue_species in glycan_df['tissue_species']:
            for species in tissue_species:
                if species not in seen:
                    unique_species.append(species)
                    seen.add(species)
                    
        # Create a matrix of with rows = glycans, columns = tissue species, 
        # values = 1 if glycan is associated to that species, 0 if not
        emb_species = pd.DataFrame(0, index=glycan_df.index, columns=unique_species)

        # Fill matrix with 1 where species is present for each glycan
        for idx, species_list in glycan_df['tissue_species'].items():
            for species in species_list:
                if species in emb_species.columns:
                    emb_species.at[idx, species] = 1
        
        time_emb += round(time.time() - t0, 2) # take into account the time it takes
        
        emb_df = emb_species.copy()
        
    elif feature_name == 'Disease':
        # Get the list of unique diseases
        seen = set() # to track seen diseases
        unique_diseases = [] # list of unique diseases
        
        t0 = time.time()
        for disease_association in glycan_df['disease_association']:
            for disease in disease_association:
                if disease not in seen:
                    unique_diseases.append(disease)
                    seen.add(disease)
                    
        # Create a matrix of with rows = glycans, columns = diseases, 
        # values = 1 if glycan is associated to that disease, 0 if not 
        emb_diseases = pd.DataFrame(0, index=glycan_df.index, columns=unique_diseases)
        
        # Fill matrix with 1 where species is present for each glycan\n",
        for idx, disease_list in glycan_df['disease_association'].items():
            for disease in disease_list:
                if disease in emb_diseases.columns:
                    emb_diseases.at[idx, disease] = 1
                    
        time_emb += round(time.time() - t0, 2) # take into account the time it takes
        
        emb_df = emb_diseases.copy()
    
    model = 'One-Hot'
        
    return emb_df, time_emb, model

def tfidf_embedding(feature_name, glycan_df):
    """
    Compute a TF-IDF embedding for a given feature of glycans.

    Parameters
    ----------
    feature_name : str
        Feature to embed ('Sequence', 'Tissue', 'Species', or 'Disease').

    glycan_df : pd.DataFrame
        Full glycan dataset.

    Returns
    -------
    emb_df : pd.DataFrame
        Glycan embeddings based on TF-IDF vectors.

    time_tfid : float
        Time taken to compute the embedding (in seconds).

    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    """
    glycan_df = glycan_df.copy()
    
    if feature_name == 'Sequence':
        # Use glycowork to preprocess glycan sequences
        glycan_df['Processed Sequence'] = min_process_glycans(list(glycan_df.index))
        glycan_df['Sequence_str'] = glycan_df['Processed Sequence'].apply(lambda seq: ' '.join(seq))
        docs = glycan_df['Sequence_str']
    
    elif feature_name == 'Tissue':
        raw_values = glycan_df['tissue_sample']
        docs = raw_values.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    elif feature_name == 'Species':
        raw_values = glycan_df['tissue_species']
        docs = raw_values.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    elif feature_name == 'Disease':
        raw_values = glycan_df['disease_association']
        docs = raw_values.apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    elif feature_name == 'Composition':
        # TF-IDF doesn't make sense here — handled elsewhere
        print('TF-IDF not applicable for Composition.')
        return None, 0
    
    else:
        print(f"Unrecognized feature for TF-IDF: {feature_name}")
        return None, 0

    # Now apply TF-IDF
    vectorizer = TfidfVectorizer()
    t0 = time.time()
    X_tfidf = vectorizer.fit_transform(docs)
    embeddings_tfidf = X_tfidf.toarray()

    emb_df = pd.DataFrame(
        embeddings_tfidf,
        index=glycan_df.index,
        columns=vectorizer.get_feature_names_out()
    )

    time_tfid = round(time.time() - t0, 2)
    
    return emb_df, time_tfid, vectorizer


def counts_embedding(feature_name, glycan_df):
    """
    Compute a Counts (Bag-of-Words) embedding for glycans based on their sequence.

    Parameters
    ----------
    feature_name : str
        Name of the feature to embed (must be 'Sequence' or 'glycan').

    glycan_df : pd.DataFrame
        Full glycan dataset containing sequences.

    Returns
    -------
    emb_seq_counts : pd.DataFrame
        Glycan embeddings based on glycoletter counts.

    time_counts : float
        Time taken to compute the embedding (in seconds).

    vec : CountVectorizer
        Fitted CountVectorizer model.
    """
    if feature_name == 'Sequence' or 'glycan': 
        glycan_df = glycan_df.copy()
        
        if feature_name == 'Sequence':
            column = glycan_df.index
        
        else:
            column = glycan_df['glycan']    
                
        # Use glycowork's min_process_glycans() to convert glycans sequence into a nested lists of glycoletters
        glycan_df['Processed Sequence'] = min_process_glycans(list(column))
            
        # Join glycoletters into space-separated "sentences"
        glycan_df['Sequence_str'] = glycan_df['Processed Sequence'].apply(lambda seq: ' '.join(seq))
        
        # Fit Counts vectorizer on glycoletter sentences
        # This turns each gylcan into a fixed-length vector of token counts
        vec = CountVectorizer(token_pattern=r"[^ ]+")
        t0 = time.time()
        X_counts = vec.fit_transform(glycan_df['Sequence_str'])
        
        # Save Counts glycan embeddings based on sequence similarity
        embeddings_counts = X_counts.toarray()
        
        # Create DataFrame with glycoletter features
        emb_seq_counts = pd.DataFrame(
            embeddings_counts,
            index=glycan_df.index,
            columns=vec.get_feature_names_out()
            )
        
        time_counts = round(time.time() - t0, 2) # take into account the time it takes
        
        return emb_seq_counts, time_counts, vec
    
    else:
        print(f'Impossible to use this method for {feature_name}')
        return None, 0, None
        
    
def choose_embedding_method_1(feature_name, method, glycan_df):
    """
    Apply feature representation method (before dimensionality reduction).

    Parameters
    ----------
    feature_name : str
        Name of the feature to embed (e.g., 'Sequence', 'Species').

    method : str
        Embedding method to use: 'One-Hot', 'TF-IDF', or 'Counts'.
    
    glycan_df : pd.DataFrame
        Full glycan dataset containing the feature to embed.

    Returns
    -------
    emb : pd.DataFrame
        Embedded feature matrix.

    time_emb : float
        Time taken to compute the embedding (in seconds).

    model_used : dict
        Dictionary mapping feature_name to the fitted embedding model.
    """
    model_used = {}
    total_time = 0
    if method == 'One-Hot':
        emb, time_emb, model = one_hot_embedding(feature_name, glycan_df)
        total_time += time_emb
        print('------------One-Hot DONE')
        
    elif method == 'TF-IDF':
        emb, time_emb, model = tfidf_embedding(feature_name, glycan_df)
        total_time += time_emb
        print('------------TF-IDF DONE')
    
    elif method == 'Counts':
        emb, time_emb, model = counts_embedding(feature_name, glycan_df)
        total_time += time_emb
        print('------------Counts DONE')
    
    model_used[feature_name] = model
    
    print(f'Shape of Matrix after embedding: {emb.shape}')
    
    return emb, time_emb, model_used

def choose_embedding_method_2(feature, method, n_components, glycan_df, emb1):
    """
    Apply dimensionality reduction method to a feature embedding.

    Parameters
    ----------
    feature : str
        The feature being embedded (e.g., 'Sequence', 'Composition').

    method : str or None
        Dimensionality reduction method to use: 'PCA', 'SVD', 'TSNE', or None.
    
    n_components : int
        Number of components to reduce to (ignored if method is None).

    glycan_df : pd.DataFrame
        Full glycan dataframe to use for index alignment.

    emb1 : pd.DataFrame
        Input embedding matrix before dimensionality reduction.

    Returns
    -------
    emb_final : pd.DataFrame
        Embedding matrix after applying the specified dimensionality reduction.
        
    total_time : float
        Total runtime (in seconds) for the dimensionality reduction step.
    
    model_used : dict
        Dictionary mapping feature to the fitted dimensionality reduction model.
    """
    model_used = {}
    total_time = 0
    if method == 'PCA':
        # Learn embedding space using PCA
        t0 = time.time()
        
        max_components = min(emb1.shape[0], emb1.shape[1])
        n_comp = min(n_components, max_components)
        emb1.columns = emb1.columns.astype(str)

        pca = PCA(n_comp, random_state=42)
        reduced = pca.fit_transform(emb1)

        emb_final = pd.DataFrame(reduced, index=glycan_df.index)
        
        total_time += round(time.time() - t0, 2)
        
        model_used[feature] = pca
        
        print('------------PCA DONE')
        
    elif method == 'SVD':
        # Learn embedding space using SVD
        t0 = time.time()

        # Adjust n_components if it's greater than the number of original features
        max_components = min(n_components, emb1.shape[1])
        emb1.columns = emb1.columns.astype(str)
        
        svd = TruncatedSVD(max_components, random_state=42)
        reduced = svd.fit_transform(emb1)

        emb_final = pd.DataFrame(reduced, index=glycan_df.index)
        
        total_time += round(time.time() - t0, 2)
        
        model_used[feature] = svd
                
        print('------------SVD DONE')
    
    if method == 'TSNE':
        # Learn embedding space using TSNE 
        t0 = time.time()
        
        emb1.columns = emb1.columns.astype(str)
        
        tsne = TSNE(n_components, random_state=42)
        reduced = tsne.fit_transform(emb1)

        emb_final = pd.DataFrame(reduced, index=glycan_df.index)
        
        total_time += round(time.time() - t0, 2)
        
        model_used[feature] = tsne
                
        print('------------TSNE DONE')

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.6)
        plt.title('TSNE 2D Plot of Glycan Embedding')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.grid(True)
        plt.show()
        
    elif method is None:
        emb_final = emb1
        model_used[feature] = None
        print('------------NO REDUCTION')
    
    print(f'Shape of Final Matrix after PCA/SVD/TSNE/no reduction: {emb_final.shape}')
        
    return emb_final, total_time, model_used

def learn_and_evaluate_embedding(feature_dict, glycan_df, df_n_glycans):
    """
    Learn an embedding space for glycans based on selected features and evaluate its quality.

    Parameters
    ----------
    feature_dict : dict
        Dictionary mapping each feature ('Sequence', 'Composition', etc.) to its two methods and number of components.
        
    glycan_df : pd.DataFrame
        DataFrame containing glycans and associated features.
        
    df_n_glycans : pd.DataFrame
        DataFrame listing known N-glycans for ground truth evaluation.
        
    glycan_binding_df : pd.DataFrame
        DataFrame containing protein-glycan binding information (may be None).

    Returns
    -------
    tuple
        Tuple containing:
        - Silhouette score (float)
        - Nearest-neighbor purity (float)
        - Total embedding time (float, in seconds)
        - Models used for each feature (dict)
        - Final combined and normalized embedding (pd.DataFrame)
    """
    embeddings_list = [] # keep track of embeddings used
    total_time = 0 # keep track of time it takes to learn a specific embedding space
    models_used = {}
    
    for feature in list(feature_dict.keys()):
        method1 = feature_dict[feature][0]
        method2 = feature_dict[feature][1]
        n_comp = feature_dict[feature][2]
        
        if feature == 'Sequence':
            print('--------------------------------Sequence START--------------------------------')
            emb1_seq, time1_seq, model1_seq = choose_embedding_method_1(feature, method1, glycan_df)
            emb_seq, time_seq, model2_seq = choose_embedding_method_2(feature, method2, n_comp, glycan_df, emb1_seq)
            embeddings_list.append(emb_seq)
            total_time += time1_seq + time_seq
            models_used[feature] = [model1_seq[feature], model2_seq[feature]]
            print('--------------------------------Sequence DONE--------------------------------')
        
        elif feature == 'Composition':
            print('--------------------------------Composition START--------------------------------')
            emb1_comp, time1_comp, model1_comp = choose_embedding_method_1(feature, method1, glycan_df)
            emb_comp, time_comp, model2_comp = choose_embedding_method_2(feature, method2, n_comp, glycan_df, emb1_comp)
            embeddings_list.append(emb_comp)
            total_time += time1_comp + time_comp
            models_used[feature] = [model1_comp[feature], model2_comp[feature]]
            print('--------------------------------Composition DON--------------------------------')
            
        elif feature == 'Species':
            print('--------------------------------Species START--------------------------------')
            emb1_species, time1_species, model1_species = choose_embedding_method_1(feature, method1, glycan_df)
            emb_species, time_species, model2_species = choose_embedding_method_2(feature, method2, n_comp, glycan_df, emb1_species)
            embeddings_list.append(emb_species)
            total_time += time1_species + time_species
            models_used[feature] = [model1_species[feature], model2_species[feature]]
            print('--------------------------------Species DONE--------------------------------')
            
        elif feature == 'Tissue':
            print('--------------------------------Tissue START--------------------------------')
            emb1_tissue, time1_tissue, model1_tissue = choose_embedding_method_1(feature, method1, glycan_df)
            emb_tissue, time_tissue, model2_tissue = choose_embedding_method_2(feature, method2, n_comp, glycan_df, emb1_tissue)
            embeddings_list.append(emb_tissue)
            total_time += time1_tissue + time_tissue
            models_used[feature] = [model1_tissue[feature], model2_tissue[feature]]
            print('--------------------------------Tissue DONE--------------------------------')
        
        elif feature == 'Disease':
            print('--------------------------------Disease START--------------------------------')
            emb1_disease, time1_disease, model1_disease= choose_embedding_method_1(feature, method1, glycan_df)
            emb_disease, time_disease, model2_disease = choose_embedding_method_2(feature, method2, n_comp, glycan_df, emb1_disease)
            embeddings_list.append(emb_disease)
            total_time += time1_disease + time_disease
            models_used[feature] = [model1_disease[feature], model2_disease[feature]]
            print('--------------------------------Disease DONE--------------------------------')
        
        elif feature == 'Protein-Glycan Binding':
            print('--------------------------------Protein-Glycan START--------------------------------')
            model1_pgb = {'Protein-Glycan Binding': 'No Method'}
            emb1_pgb = glycan_df.copy()
            emb1_pgb = emb1_pgb.drop(['Composition', 'tissue_species', 'tissue_sample', 'disease_association'], axis=1)
            emb_pgb, time_pgb, model2_pgb = choose_embedding_method_2(feature, method2, n_comp, glycan_df, emb1_pgb)
            embeddings_list.append(emb_pgb)
            total_time += time_pgb
            models_used[feature] = [model1_pgb[feature], model2_pgb[feature]]
            print('--------------------------------Protein-Glycan DONE--------------------------------')
        
        else:
            raise ValueError(f"Unrecognized feature: {feature}")


    # Combine all embeddings together and normalize all features
    t0 = time.time()
        
    scaled_emb = combine_and_normalize_embeddings(embeddings_list, index=glycan_df.index) # TF-IDF
    total_time += round(time.time() - t0, 2)

    # Evaluate embedding method with silhouette scores
    sil_score = evaluate_embedding_sil_score(scaled_emb, df_n_glycans)
    
    # Evaluate embedding method with nearest neighbors purity
    nn_purity = evaluate_embedding_nn_purity(scaled_emb, df_n_glycans)
    
    return sil_score, nn_purity, total_time, models_used, scaled_emb

def compare_embeddings(embedding_names, feature_dicts, df_glycan, df_n_glycans, glycan_binding_df=None):
    """
    Evaluates multiple embedding configurations and summarizes the results in a DataFrame.

    Parameters
    ----------
    embedding_names : list of str
        Names (IDs) for each embedding configuration, used as row index.
        
    feature_dicts : list of dict
        Each dictionary contains features and their associated embedding method for the corresponding embedding.
        
    df_glycan : pd.DataFrame
        Full glycan dataset used to build embeddings.
        
    df_n_glycans : pd.DataFrame
        DataFrame listing known N-glycans for labeling and evaluation.

    glycan_binding_df : pd.DataFrame, optional
        DataFrame containing protein-glycan binding information (default is None).

    Returns
    -------
    pd.DataFrame
        A DataFrame where:
        - Rows = embedding configurations (embedding_names)
        - Columns = selected features (1 if used, else 0), silhouette score, NN purity, runtime, embedding model, and embedding matrix.
    """
    # Get all unique features across all glycans
    seen = set()  # Track seen features
    unique_features = []  # List of unique feature strings

    for feature_dict in feature_dicts:
        for k, v in feature_dict.items():
            feature = f"{k}_{v[0]}_{v[1]}"
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
    # Create matrix: rows = glycans, columns = features (e.g. tissues or species)
    summary_df = pd.DataFrame(0, index=embedding_names, columns=unique_features)

    # Fill matrix: 1 if glycan has that feature, 0 otherwise
    for glycan_name, feature_dict in zip(embedding_names, feature_dicts):
        for k, v in feature_dict.items():
            feature = f"{k}_{v[0]}_{v[1]}"
            if feature in summary_df.columns:
                summary_df.at[glycan_name, feature] = 1
    
    # Prepare data to collect
    s_scores = []
    nn_purities = []
    times = []
    all_models = []
    all_embeddings = []

    # Loop through embeddings
    for name, feature_dict in zip(embedding_names, feature_dicts):
        print(f"Embedding name: {name}")
        print(f"Features used: {feature_dict}")
        # Evaluate the embedding
        sil_score, nn_purity, time, models_used, final_emb = learn_and_evaluate_embedding(feature_dict, df_glycan, df_n_glycans, glycan_binding_df)

        # Build feature presence dict
        s_scores.append(round(sil_score, 3))
        nn_purities.append(round(nn_purity, 3))
        times.append(round(time, 2))
        all_models.append(models_used)
        all_embeddings.append(final_emb)

    # Build DataFrame
    summary_df['Silhouette Score'] = s_scores
    summary_df['NN Purity'] = nn_purities
    summary_df['Time (s)'] = times
    summary_df['Methods'] = all_models
    summary_df['Embedding Matrix'] = all_embeddings
    summary_df.index.name = 'Embedding Name'
    
    return summary_df