import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline

def compute_class_cv(data_matrix, data_matrix_types, class_name, batch, visualize=False, xlim=None):
    """
    Compute the coefficient of variation (CV) for each feature within a specified class and batch.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        Data matrix with samples as index and features as columns.
    data_matrix_types : pd.DataFrame
        Same as Data matrix but with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    class_name : str
        The sample class to compute CV for (e.g., 'QC').
    batch : int
        Batch number to filter the samples.
    visualize : bool, optional
        If True, displays a histogram of CV values. Default is False.
    xlim : tuple or None, optional
        x-axis limits for the histogram, e.g., (0, 100). Default is None.

    Returns
    -------
    pd.Series
        Coefficient of variation (CV in %) for each feature.
    """
    data = data_matrix.copy()
    
    # Select only the samples measured in a specific batch
    data = data_matrix[(data_matrix_types['batch'] == batch)]
    
    # Select only the samples corresponding to the specified class
    data = data[(data_matrix_types['class'] == class_name)]
    
    # Calculate CV (%) for each feature: (standard deviation / mean) * 100
    cv = data.std() / data.mean() * 100
    
    # Optionally visualize the distribution of CVs
    if visualize:
        # Plot a histogram of CV values across all features
        plt.figure(figsize=(10, 6))
        plt.hist(cv, bins=30, edgecolor='black')
        plt.title(f"Distribution of Coefficient of Variation (CV) Across Features for class: {class_name} (Batch {batch})") 
        plt.xlabel("CV (%)")
        if xlim is not None:
            plt.xlim(xlim)
        plt.ylabel("Number of Features")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return cv

def inspect_blanks(blanks, batch):
    """
    Inspect blank samples for a specific batch by listing the unique values of peak areas.

    Parameters
    ----------
    blanks : pd.DataFrame
        Data Matrix with metadata restricted to blank samples (columns: 'batch', 'class', 'order', 'id' and features).
    batch : int
        Batch number to inspect.

    Returns
    -------
    None
        Prints the number and example of unique values in each column for the specified batch.
    """
    # Filter blanks for the specified batch and drop the 'batch' column
    blanks = blanks[blanks['batch'] == batch].drop('batch', axis=1)
    
    # Print number of unique values and first 5 unique entries for each column
    for col in blanks.columns:
        unique_vals = blanks[col].unique()
        print(f"(Batch {batch}) Column '{col}' has {len(unique_vals)} unique value(s): {unique_vals[:5]}")

def sn_ratio(blanks, non_blanks, batch):
    """
    Compute the signal-to-noise (S/N) ratio for each feature in a given batch.

    Parameters
    ----------
    blanks : pd.DataFrame
        Data Matrix with metadata restricted to blank samples (columns: 'batch', 'class', 'order', 'id' and features).
    non_blanks : pd.DataFrame
        Data Matrix with metadata restricted to non-blank samples (columns: 'batch', 'class', 'order', 'id' and features).
    batch : int
        Batch number to filter both datasets.

    Returns
    -------
    pd.Series
        Signal-to-noise ratio for each feature (mean signal / standard deviation of blank).
    """
    # Filter blanks and non-blanks for the specified batch
    blanks = blanks[blanks['batch'] == batch].drop('batch', axis=1)
    non_blanks = non_blanks[non_blanks['batch'] == batch].drop('batch', axis=1)

    # Compute noise as std dev in blanks, and signal as mean in non-blanks
    noise_std = blanks.std()
    signal_mean = non_blanks.mean()
    
    return signal_mean/noise_std

def stat_threshold(blanks, batch):
    """
    Compute a statistical threshold for detection based on blank samples from a specific batch.

    The threshold is defined as the average of (mean + 3×std) across all features in the batch's blanks.

    Parameters
    ----------
    blanks : pd.DataFrame
        Data Matrix with metadata restricted to blank samples (columns: 'batch', 'class', 'order', 'id' and features).
    batch : int
        Batch number to filter the blanks.

    Returns
    -------
    float
        Statistical threshold value computed as mean + 3*std across features, then averaged.
    """
    # Filter blanks for the specified batch and drop the 'batch' column
    blanks = blanks[blanks['batch'] == batch].drop('batch', axis=1)
    
    # Compute threshold as the average of (mean + 3 * std) across features
    threshold = (blanks.mean() + 3*blanks.std()).mean()
    
    return threshold

def detected_features(data_matrix, batch, T): 
    """
    Print the number of features detected in a given batch based on a threshold.

    A feature is considered 'detected' in a sample if its value is greater than the specified threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Data matrix with samples as rows and features as columns.
    batch : int
        Identifier for the batch being analyzed.
    T : float
        Threshold value to determine whether a feature is detected.

    Returns
    -------
    None
        Prints two statistics:
        - Number of features detected in at least one sample.
        - Total number of detected features across all samples.
    """
    # Count features detected in at least one sample
    num_features_detected = (data_matrix > T).any(axis=0).sum()
    print(f"(Batch {batch}) Features detected in at least one sample: {num_features_detected}")

    # Count total number of times features are detected across all samples
    total_detections = (data_matrix > T).sum().sum()
    print(f"(Batch {batch}) Total number of feature detections across all samples: {total_detections}")


def avg_detected_features(data_matrix_types, batch, T):
    """
    Plot the average number of detected features per sample class for a given batch.

    A feature is considered detected in a sample if its peak area is greater than the specified threshold.

    Parameters
    ----------
    data_matrix_types : pd.DataFrame
        Data matrix with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    batch : int
        Batch number (used for plot title context).
    T : float
        Detection threshold; features above this value are considered detected.

    Returns
    -------
    None
        Displays a bar plot showing the average number of detected features per sample class.
    """
    # Plot the average number of detected features (features for which peak area is > threshold) for each sample type
    average_features_detected = data_matrix_types.groupby('class').apply(lambda group: (group > T).sum(axis=1)).groupby('class').mean()

    # Plot the average number of features detected for each sample type
    plt.figure(figsize=(10, 6))
    average_features_detected.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title(f'Average Number of Features Detected across Classes (Batch {batch})')
    plt.xlabel('Sample Type')
    plt.ylabel('Average Number of Features Detected')
    plt.xticks(rotation=45, ha='right') 
    plt.tight_layout() 
    plt.show()

def compute_d_ratio(data_matrix, data_matrix_types, batch):
    """
    Compute and plot the D-Ratio for a given batch.

    The D-Ratio reflects technical variability (std in QCs) relative to total variability
    (combined variance across sample classes). Lower D-Ratios indicate stronger biological signal.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        Matrix of feature intensities (rows = samples, columns = features).
    data_matrix_types : pd.DataFrame
        Same as Data Matrix but with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    batch : int
        Batch number for which the D-Ratio should be computed.

    Returns
    -------
    pd.Series
        D-Ratio (%) for each feature.
    """
    # Compute numerator: std of QC samples in the batch
    numerator = data_matrix[(data_matrix_types['class'] == 'QC') & (data_matrix_types['batch'] == batch)].std()

    # Compute variance of each class within the batch
    filtered_types = data_matrix_types[data_matrix_types['batch'] == batch]
    filtered_data = data_matrix.loc[filtered_types.index]
    
    # Merge for group-wise variance
    merged = filtered_data.copy()
    merged['class'] = filtered_types['class']

    # Group by class and compute variance
    samples_var = merged.groupby('class').var()
    
    # Denominator: sqrt of sum of variances (QC + biological classes)
    denominator = np.sqrt(
        samples_var.loc['QC'] +
        samples_var.loc['French'] +
        samples_var.loc['Dunn'] +
        samples_var.loc['LMU']
    )
    
    # Compute d-ratio
    d_ratio = numerator / denominator * 100

    # Plot
    x = np.arange(len(d_ratio))
    y = d_ratio.values
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='D-Ratio')
    plt.plot(x, line, color='red', linestyle='--', label='Linear Fit')
    plt.title(f'D-Ratio across Features (Batch {batch})')
    plt.xlabel('Feature Index')
    plt.ylabel('D-Ratio (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return d_ratio

def contamination(data_matrix_types, batch, T):
    """
    Visualize the most frequently detected features in blank samples, indicating potential contamination.

    Parameters
    ----------
    data_matrix_types : pd.DataFrame
        Data Matrix with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    batch : int
        Batch number to filter blank samples for inspection.
    T : float
        Detection threshold for determining if a feature is considered present.

    Returns
    -------
    None
        Displays a heatmap of the top N most frequently detected features in blanks.
    """
    data_matrix_types = data_matrix_types[data_matrix_types['batch'] == batch].drop('batch', axis=1)
    detected_blanks = data_matrix_types[data_matrix_types>T]

    # Count number of blanks each feature is detected
    feature_counts = detected_blanks.count(axis=0).sort_values(ascending=False)

    top_features = feature_counts.head(3).index  # Change number as needed
    top_data = detected_blanks[top_features]

    plt.figure(figsize=(12, 6))
    sns.heatmap(top_data.T, cmap='Reds', cbar_kws={'label': 'Peak Area'}, annot=True, fmt='.1f')

    plt.title(f'Top 20 Most Frequently Detected Features in Blanks (Batch {batch})')
    plt.xlabel('Blank Samples')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
def plot_standard_feature_trends(standard_name, std_features_mapping, standards_data_matrix, threshold=10):
    """
    Plot the trend of standard compound features across samples, including linear regression fits.

    Parameters
    ----------
    standard_name : str
        Name of the standard compound.
    std_features_mapping : dict
        Dictionary mapping standard names to lists of associated feature names.
    standards_data_matrix : pd.DataFrame
        Data matrix containing standard features (samples x features).
    threshold : float, optional
        Detection threshold to filter out low signals. Default is 10.

    Returns
    -------
    None
    """
    # Get feature names for the selected standard
    standard_features = std_features_mapping[standard_name]
    
    # Select relevant columns
    standard_data = standards_data_matrix.loc[:, standards_data_matrix.columns.isin(standard_features)]
    
    # Filter: only keep samples where all selected features are detected (above threshold)
    standard_data = standard_data[standard_data > threshold].dropna(axis=0)

    # x-axis: sample indexes
    x = np.arange(len(standard_data)).reshape(-1, 1)

    # Line plot of feature values across samples
    plt.figure(figsize=(20, 6))
    for feature in standard_data.columns:
        plt.plot(x, standard_data[feature], label=feature)

    plt.xlabel('Sample Index')
    plt.ylabel('Peak Area')
    plt.title(f'Peak Area Evolution Across Samples for {standard_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Subplots with linear regression fits
    n_features = len(standard_data.columns)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i, feature in enumerate(standard_data.columns):
        y = standard_data[feature].values
        mask = ~np.isnan(y)

        if np.sum(mask) < 2:
            axs[i].set_visible(False)
            continue

        model = LinearRegression()
        model.fit(x[mask], y[mask])
        y_pred = model.predict(x)

        axs[i].plot(x, y, label="Data", alpha=0.6)
        axs[i].plot(x, y_pred, label="Trend", linestyle='--')
        axs[i].set_title(feature)
        axs[i].set_xlabel('Sample Index')
        axs[i].set_ylabel('Peak Area')
        axs[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_total_intensity_vs_order(data_matrix, data_matrix_types, ylim=None):
    """
    Plot total intensity of each sample against injection order, colored by sample class.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        Data Matrix (samples x features).
    data_matrix_types : pd.DataFrame
        Same as Data Matrix but with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    ylim : tuple, optional
        Limits for the y-axis.

    Returns
    -------
    None
    """
    # Compute total intensity per sample
    sample_total_intensity = data_matrix.sum(axis=1)

    # Merge intensity with metadata
    intensity_df = data_matrix_types[['class', 'batch', 'order']].copy()
    intensity_df['total_intensity'] = sample_total_intensity

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=intensity_df, x='order', y='total_intensity', hue='class', alpha=0.8)
    plt.title("Sample Total Intensity vs. Run Order")
    plt.xlabel("Injection Order")
    plt.ylabel("Total Intensity")
    if ylim != None:
        plt.ylim(ylim)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    
def median_normalization(data_matrix, data_matrix_types, qc_class='QC'):
    """
    Apply QC-based median normalization using the 3 nearest QCs in acquisition order.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        Data matrix with samples as rows and metabolites as columns.
    data_matrix_types : pd.DataFrame
        Same as Data Matrix but with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    qc_class : str, optional
        Class label indicating QC samples.

    Returns
    -------
    pd.DataFrame
        Normalized data matrix.
    """
    # Ensure 'order' is sorted
    data_matrix_types = data_matrix_types.sort_values('order')
    data_matrix = data_matrix.loc[data_matrix_types.index]
    
    # Identify QC samples
    qc_indices = data_matrix_types[data_matrix_types['class'] == qc_class].index
    qc_orders = data_matrix_types.loc[qc_indices, 'order']
    
    # Initialize normalized data matrix
    normalized_data = data_matrix.copy()
    
    for idx in data_matrix.index:
        sample_order = data_matrix_types.loc[idx, 'order']
        
        # Calculate distances to QC samples
        distances = abs(qc_orders - sample_order)
        nearest_qcs = distances.nsmallest(3).index
        
        # Compute median of the nearest QCs for each metabolite
        qc_median = data_matrix.loc[nearest_qcs].median()
        
        # Normalize sample by QC median
        normalized_data.loc[idx] = data_matrix.loc[idx] / qc_median
        
    return normalized_data

def plot_cv_boxplots(data_matrix: pd.DataFrame,
                     data_matrix_types: pd.DataFrame,
                     class_col: str = 'class',
                     scale: float = 100.0,
                     figsize: tuple = (12, 6)):
    """
    Plot boxplots of the coefficient of variation (CV) for each sample class.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        Data matrix (samples x features).
    data_matrix_types : pd.DataFrame
        Same as Data Matrix but with metadata (columns: 'batch', 'class', 'order', 'id' and features).
    class_col : str
        Name of the class label column.
    scale : float
        Scaling factor for CV (e.g., 100 for %).
    figsize : tuple
        Size of the figure.

    Returns
    -------
    None
    """
    # Ensure alignment
    common_samples = data_matrix.index.intersection(data_matrix_types.index)
    data_matrix = data_matrix.loc[common_samples]
    data_matrix_types = data_matrix_types.loc[common_samples]

    # Compute CV for each class
    class_labels = data_matrix_types[class_col].unique()
    class_labels = class_labels[class_labels!='B']
    class_labels = class_labels[class_labels!='SS']
    cv_dict = {}
    for cls in class_labels:
        # Subset samples of this class
        samples = data_matrix_types[data_matrix_types[class_col] == cls].index
        subset = data_matrix.loc[samples]
        means = subset.mean(axis=0)
        stds = subset.std(axis=0, ddof=1)
        cv = (stds / means) * scale
        cv_dict[cls] = cv

    # Create DataFrame and plot
    cv_df = pd.DataFrame(cv_dict)[class_labels]  # preserve order
    plt.figure(figsize=figsize)
    cv_df.boxplot()
    plt.xlabel('Class')
    plt.ylabel(f'Coefficient of Variation (×{scale})')
    plt.ylim([0, 35])
    plt.title('CV Distribution by Sample Class')
    plt.show()
    
def global_scaling_normalization(data_matrix, method='mean'):
    """
    Apply global scaling normalization to a data matrix.

    Parameters
    ----------
    data_matrix : pd.DataFrame
        Matrix of metabolite intensities (samples x features).
    method : str
        Scaling method: 'mean', 'median', or 'total'.

    Returns
    -------
    pd.DataFrame
        Globally scaled normalized data matrix.
    """
    # Calculate the scaling factor based on the specified method
    if method == 'mean':
        scaling_factors = data_matrix.mean(axis=1)
    elif method == 'median':
        scaling_factors = data_matrix.median(axis=1)
    elif method == 'total':
        scaling_factors = data_matrix.sum(axis=1)
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', or 'total'.")

    # Compute the global scaling factor (e.g., median of all sample scaling factors)
    global_factor = scaling_factors.median()

    # Normalize each sample by its scaling factor and multiply by the global factor
    normalized_data = data_matrix.div(scaling_factors, axis=0) * global_factor

    return normalized_data


def random_forest_classifier(X, y, label_encoder=None, split=False, test_size=0.2, random_state=42):
    """
    Train and evaluate a Random Forest classifier using either cross-validation or train-test split.

    Parameters:
        X (DataFrame): Feature matrix.
        y (array-like): Target labels.
        label_encoder (LabelEncoder, optional): Fitted label encoder for class display.
        split (bool): If True, use train-test split. If False, use 5-fold cross-validation.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        clf (RandomForestClassifier): Trained Random Forest model.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

    if split:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Accuracy
        acc = clf.score(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=label_encoder.classes_ if label_encoder else None)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Test Set Confusion Matrix")
        plt.show()

        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_ if label_encoder else None))
        
        return clf

    else:
        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=5)
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        y_pred_cv = cross_val_predict(clf, X, y, cv=5)

        cm = confusion_matrix(y, y_pred_cv)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=label_encoder.classes_ if label_encoder else None)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Cross-Validated Confusion Matrix")
        plt.show()

        print(classification_report(y, y_pred_cv, target_names=label_encoder.classes_ if label_encoder else None))
        
        return clf
    
def lasso_classifier(X, y, label_encoder=None, split=False, test_size=0.2, random_state=42):
    """
    Train and evaluate a LASSO-regularized logistic regression classifier (via LogisticRegressionCV)
    using either cross-validation or train-test split.

    Parameters:
        X (DataFrame): Feature matrix.
        y (array-like): Target labels.
        label_encoder (LabelEncoder, optional): Fitted label encoder for class display.
        split (bool): If True, use train-test split. If False, use 5-fold cross-validation.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        clf (Pipeline): Trained scikit-learn pipeline with LogisticRegressionCV.
    """
    # Define the LASSO pipeline (scaling + L1 logistic regression)
    clf = make_pipeline(
        LogisticRegressionCV(
            cv=5,
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            max_iter=5000,
            random_state=random_state
        )
    )

    if split:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=label_encoder.classes_ if label_encoder else None)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("LASSO - Test Set Confusion Matrix")
        plt.show()
        
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_ if label_encoder else None))

    else:
        # Cross-validation mode
        cv_scores = cross_val_score(clf, X, y, cv=5)
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        y_pred_cv = cross_val_predict(clf, X, y, cv=5)
        cm = confusion_matrix(y, y_pred_cv)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=label_encoder.classes_ if label_encoder else None)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("LASSO - Cross-Validated Confusion Matrix")
        plt.show()
        
        print(classification_report(y, y_pred_cv, target_names=label_encoder.classes_ if label_encoder else None))

        # Fit the model on full data for returning
        clf.fit(X, y)

    return clf