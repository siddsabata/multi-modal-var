"""
Train logistic regression models for variant pathogenicity prediction.

Evaluates five configurations:
1. DNA delta embeddings only
2. Protein delta embeddings only
3. Combined DNA + Protein embeddings (unscaled - baseline)
4. Combined DNA + Protein with StandardScaler (scaled concatenation)
5. CCA fusion (PCA reduction → CCA → concatenate projections)

Uses PCA for dimensionality reduction, class weighting for imbalance,
and reports both holdout test and cross-validation performance.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# XGBoost will be imported lazily when needed
XGBOOST_AVAILABLE = None  # Will be checked on first use


def load_data(data_dir):
    """
    Load embeddings and labels from analysis directory.

    Returns:
        X_dna: DNA delta embeddings (66, 4096)
        X_protein: Protein delta embeddings (66, 1024)
        y: Labels (66,)
        variant_ids: Variant IDs (66,)
        failed_indices: Indices of failed protein embeddings
    """
    data_dir = Path(data_dir)

    print("Loading data...")

    # Load DNA embeddings
    dna_data = np.load(data_dir / 'dna_delta_embeddings.npz')
    X_dna = dna_data['delta_embeddings']
    dna_variant_ids = dna_data['variant_ids']

    # Load protein embeddings
    protein_data = np.load(data_dir / 'protein_delta_embeddings.npz')
    X_protein = protein_data['delta_embeddings']
    protein_variant_ids = protein_data['variant_ids']
    failed_indices = protein_data.get('failed_indices', np.array([]))

    # Load labels from parquet
    df = pd.read_parquet(data_dir / 'snvs.parquet')
    y = df['label'].values
    parquet_variant_ids = df['variant_id'].values

    # Verify alignment
    assert np.array_equal(dna_variant_ids, protein_variant_ids), "Variant IDs don't match between embeddings!"
    assert np.array_equal(dna_variant_ids, parquet_variant_ids), "Variant IDs don't match with parquet!"

    print(f"Loaded {len(y)} total samples")
    print(f"DNA embeddings shape: {X_dna.shape}")
    print(f"Protein embeddings shape: {X_protein.shape}")
    print(f"Failed protein embeddings: {len(failed_indices)}")

    return X_dna, X_protein, y, dna_variant_ids, failed_indices


def filter_failed_samples(X_dna, X_protein, y, variant_ids, failed_indices):
    """
    Remove samples with failed protein embeddings and uncertain labels.

    Filters out:
    - Samples with failed protein embeddings (zero rows)
    - Samples with uncertain labels (label == -1)

    Returns:
        Filtered versions of all inputs
    """
    # Start with all valid
    valid_mask = np.ones(len(y), dtype=bool)

    # Filter failed protein embeddings
    if len(failed_indices) > 0:
        valid_mask[failed_indices] = False
        print(f"\nFiltered out {len(failed_indices)} samples with failed protein embeddings")

    # Filter uncertain labels (label == -1)
    uncertain_mask = (y == -1)
    n_uncertain = uncertain_mask.sum()
    if n_uncertain > 0:
        valid_mask &= ~uncertain_mask
        print(f"Filtered out {n_uncertain} samples with uncertain labels (label=-1)")

    # Filter arrays
    X_dna_filtered = X_dna[valid_mask]
    X_protein_filtered = X_protein[valid_mask]
    y_filtered = y[valid_mask]
    variant_ids_filtered = variant_ids[valid_mask]

    # Final stats
    print(f"\nRemaining samples after filtering: {len(y_filtered)}")

    # Robust label counting (handles any integer labels)
    unique_labels, counts = np.unique(y_filtered, return_counts=True)
    print(f"Label distribution:")
    for label, count in zip(unique_labels, counts):
        label_name = "Benign" if label == 0 else "Pathogenic" if label == 1 else f"Label-{label}"
        print(f"  {label_name} ({label}): {count} ({100 * count / len(y_filtered):.1f}%)")

    # Sanity check: ensure only binary labels remain
    if not np.all(np.isin(y_filtered, [0, 1])):
        unexpected = np.unique(y_filtered[~np.isin(y_filtered, [0, 1])])
        raise ValueError(f"Unexpected labels found after filtering: {unexpected}")

    return X_dna_filtered, X_protein_filtered, y_filtered, variant_ids_filtered


def evaluate_model(y_true, y_pred, y_prob):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob)
    }


def create_model(model_type, imbalance_ratio, random_seed):
    """
    Create a classifier based on model_type.

    Args:
        model_type: 'logistic' or 'xgboost'
        imbalance_ratio: Positive/negative class ratio
        random_seed: Random seed for reproducibility
    """
    if model_type == 'logistic':
        return LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_seed,
            solver='lbfgs'
        )
    elif model_type == 'xgboost':
        # Lazy import XGBoost
        try:
            from xgboost import XGBClassifier
        except (ImportError, Exception) as e:
            raise ImportError(
                f"XGBoost not available: {str(e)}\n"
                "On Mac: Run 'brew install libomp' then reinstall xgboost\n"
                "Or run with --model logistic"
            )

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            eval_metric='logloss',
            use_label_encoder=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_and_evaluate_holdout(X, y, pca_variance, test_size, random_seed, model_type='logistic'):
    """
    Train with holdout test set evaluation.

    Returns:
        Dictionary with metrics and PCA info
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Apply PCA on training data
    pca = PCA(n_components=pca_variance, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()

    # Calculate imbalance ratio
    imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

    # Train model with class balancing
    model = create_model(model_type, imbalance_ratio, random_seed)
    model.fit(X_train_pca, y_train)

    # Predictions
    y_pred = model.predict(X_test_pca)
    y_prob = model.predict_proba(X_test_pca)[:, 1]

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob)

    return {
        'metrics': metrics,
        'n_components': n_components,
        'explained_variance': explained_var,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }


def train_and_evaluate_cv(X, y, pca_variance, n_folds, random_seed, model_type='logistic'):
    """
    Train with stratified k-fold cross-validation.

    Returns:
        Dictionary with mean and std of metrics
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Storage for predictions and true labels
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    # Metrics for each fold
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply PCA on this fold's training data
        pca = PCA(n_components=pca_variance, random_state=random_seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Calculate imbalance ratio
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

        # Train model
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train_pca, y_train)

        # Predictions
        y_pred = model.predict(X_val_pca)
        y_prob = model.predict_proba(X_val_pca)[:, 1]

        # Store for aggregation
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Compute fold metrics
        fold_result = evaluate_model(y_val, y_pred, y_prob)
        for metric_name, value in fold_result.items():
            fold_metrics[metric_name].append(value)

    # Aggregate metrics across folds
    cv_metrics = {
        metric_name: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric_name, values in fold_metrics.items()
    }

    return cv_metrics


def train_and_evaluate_concat_scaled(X_dna, X_protein, y, pca_variance, test_size, random_seed, model_type='logistic'):
    """
    Train with scaled concatenation (holdout test set).

    Scales each modality separately before concatenation to balance contributions.
    """
    # Split data first
    X_dna_train, X_dna_test, X_protein_train, X_protein_test, y_train, y_test = train_test_split(
        X_dna, X_protein, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Scale each modality separately
    scaler_dna = StandardScaler()
    scaler_protein = StandardScaler()

    X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train)
    X_dna_test_scaled = scaler_dna.transform(X_dna_test)

    X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
    X_protein_test_scaled = scaler_protein.transform(X_protein_test)

    # Concatenate scaled features
    X_train = np.concatenate([X_dna_train_scaled, X_protein_train_scaled], axis=1)
    X_test = np.concatenate([X_dna_test_scaled, X_protein_test_scaled], axis=1)

    # Apply PCA on concatenated scaled features
    pca = PCA(n_components=pca_variance, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()

    # Calculate imbalance ratio
    imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

    # Train model
    model = create_model(model_type, imbalance_ratio, random_seed)
    model.fit(X_train_pca, y_train)

    # Predictions
    y_pred = model.predict(X_test_pca)
    y_prob = model.predict_proba(X_test_pca)[:, 1]

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob)

    return {
        'metrics': metrics,
        'n_components': n_components,
        'explained_variance': explained_var,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }


def train_and_evaluate_concat_scaled_cv(X_dna, X_protein, y, pca_variance, n_folds, random_seed, model_type='logistic'):
    """
    Train with scaled concatenation (cross-validation).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dna, y)):
        X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
        X_protein_train, X_protein_val = X_protein[train_idx], X_protein[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale each modality
        scaler_dna = StandardScaler()
        scaler_protein = StandardScaler()

        X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train)
        X_dna_val_scaled = scaler_dna.transform(X_dna_val)

        X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
        X_protein_val_scaled = scaler_protein.transform(X_protein_val)

        # Concatenate
        X_train = np.concatenate([X_dna_train_scaled, X_protein_train_scaled], axis=1)
        X_val = np.concatenate([X_dna_val_scaled, X_protein_val_scaled], axis=1)

        # PCA
        pca = PCA(n_components=pca_variance, random_state=random_seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Calculate imbalance ratio
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

        # Train model
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train_pca, y_train)

        # Predictions
        y_pred = model.predict(X_val_pca)
        y_prob = model.predict_proba(X_val_pca)[:, 1]

        # Store
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Fold metrics
        fold_result = evaluate_model(y_val, y_pred, y_prob)
        for metric_name, value in fold_result.items():
            fold_metrics[metric_name].append(value)

    # Aggregate
    cv_metrics = {
        metric_name: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric_name, values in fold_metrics.items()
    }

    return cv_metrics


def train_and_evaluate_cca(X_dna, X_protein, y, pca_dna_components, pca_protein_components,
                           cca_n_components, test_size, random_seed, model_type='logistic'):
    """
    Train with CCA fusion (holdout test set).

    Uses PCA for initial dimensionality reduction, then CCA to find canonical correlations.
    """
    # Split data
    X_dna_train, X_dna_test, X_protein_train, X_protein_test, y_train, y_test = train_test_split(
        X_dna, X_protein, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Step 1: PCA reduction on each modality
    pca_dna = PCA(n_components=pca_dna_components, random_state=random_seed)
    pca_protein = PCA(n_components=pca_protein_components, random_state=random_seed)

    X_dna_train_pca = pca_dna.fit_transform(X_dna_train)
    X_dna_test_pca = pca_dna.transform(X_dna_test)

    X_protein_train_pca = pca_protein.fit_transform(X_protein_train)
    X_protein_test_pca = pca_protein.transform(X_protein_test)

    # Step 2: CCA to find canonical components
    cca = CCA(n_components=cca_n_components)
    X_dna_train_cca, X_protein_train_cca = cca.fit_transform(X_dna_train_pca, X_protein_train_pca)
    X_dna_test_cca, X_protein_test_cca = cca.transform(X_dna_test_pca, X_protein_test_pca)

    # Step 3: Concatenate CCA projections
    X_train = np.concatenate([X_dna_train_cca, X_protein_train_cca], axis=1)
    X_test = np.concatenate([X_dna_test_cca, X_protein_test_cca], axis=1)

    n_components = X_train.shape[1]

    # Calculate imbalance ratio
    imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

    # Train model
    model = create_model(model_type, imbalance_ratio, random_seed)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob)

    return {
        'metrics': metrics,
        'n_components': n_components,
        'explained_variance': 1.0,  # CCA doesn't have explained variance
        'n_train': len(y_train),
        'n_test': len(y_test),
        'pca_dna_components': pca_dna_components,
        'pca_protein_components': pca_protein_components,
        'cca_components': cca_n_components
    }


def train_and_evaluate_cca_cv(X_dna, X_protein, y, pca_dna_components, pca_protein_components,
                              cca_n_components, n_folds, random_seed, model_type='logistic'):
    """
    Train with CCA fusion (cross-validation).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dna, y)):
        X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
        X_protein_train, X_protein_val = X_protein[train_idx], X_protein[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # PCA reduction
        pca_dna = PCA(n_components=pca_dna_components, random_state=random_seed)
        pca_protein = PCA(n_components=pca_protein_components, random_state=random_seed)

        X_dna_train_pca = pca_dna.fit_transform(X_dna_train)
        X_dna_val_pca = pca_dna.transform(X_dna_val)

        X_protein_train_pca = pca_protein.fit_transform(X_protein_train)
        X_protein_val_pca = pca_protein.transform(X_protein_val)

        # CCA
        cca = CCA(n_components=cca_n_components)
        X_dna_train_cca, X_protein_train_cca = cca.fit_transform(X_dna_train_pca, X_protein_train_pca)
        X_dna_val_cca, X_protein_val_cca = cca.transform(X_dna_val_pca, X_protein_val_pca)

        # Concatenate
        X_train = np.concatenate([X_dna_train_cca, X_protein_train_cca], axis=1)
        X_val = np.concatenate([X_dna_val_cca, X_protein_val_cca], axis=1)

        # Calculate imbalance ratio
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

        # Train model
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Store
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Fold metrics
        fold_result = evaluate_model(y_val, y_pred, y_prob)
        for metric_name, value in fold_result.items():
            fold_metrics[metric_name].append(value)

    # Aggregate
    cv_metrics = {
        metric_name: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric_name, values in fold_metrics.items()
    }

    return cv_metrics


def train_and_evaluate_feature_selection(X_dna, X_protein, y, k_features, pca_variance, test_size, random_seed, model_type='logistic'):
    """
    Train with DNA feature selection (holdout test set).

    Selects top K most discriminative DNA features using mutual information
    before scaling and concatenation.
    """
    # Split data first
    X_dna_train, X_dna_test, X_protein_train, X_protein_test, y_train, y_test = train_test_split(
        X_dna, X_protein, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Select top K DNA features using mutual information
    selector = SelectKBest(mutual_info_classif, k=k_features)
    X_dna_train_selected = selector.fit_transform(X_dna_train, y_train)
    X_dna_test_selected = selector.transform(X_dna_test)

    # Scale each modality separately
    scaler_dna = StandardScaler()
    scaler_protein = StandardScaler()

    X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train_selected)
    X_dna_test_scaled = scaler_dna.transform(X_dna_test_selected)

    X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
    X_protein_test_scaled = scaler_protein.transform(X_protein_test)

    # Concatenate scaled features
    X_train = np.concatenate([X_dna_train_scaled, X_protein_train_scaled], axis=1)
    X_test = np.concatenate([X_dna_test_scaled, X_protein_test_scaled], axis=1)

    # Apply PCA on concatenated scaled features
    pca = PCA(n_components=pca_variance, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()

    # Calculate imbalance ratio
    imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

    # Train model
    model = create_model(model_type, imbalance_ratio, random_seed)
    model.fit(X_train_pca, y_train)

    # Predictions
    y_pred = model.predict(X_test_pca)
    y_prob = model.predict_proba(X_test_pca)[:, 1]

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob)

    return {
        'metrics': metrics,
        'n_components': n_components,
        'explained_variance': explained_var,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'k_features': k_features
    }


def train_and_evaluate_feature_selection_cv(X_dna, X_protein, y, k_features, pca_variance, n_folds, random_seed, model_type='logistic'):
    """
    Train with DNA feature selection (cross-validation).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dna, y)):
        X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
        X_protein_train, X_protein_val = X_protein[train_idx], X_protein[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Select top K DNA features
        selector = SelectKBest(mutual_info_classif, k=k_features)
        X_dna_train_selected = selector.fit_transform(X_dna_train, y_train)
        X_dna_val_selected = selector.transform(X_dna_val)

        # Scale each modality
        scaler_dna = StandardScaler()
        scaler_protein = StandardScaler()

        X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train_selected)
        X_dna_val_scaled = scaler_dna.transform(X_dna_val_selected)

        X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
        X_protein_val_scaled = scaler_protein.transform(X_protein_val)

        # Concatenate
        X_train = np.concatenate([X_dna_train_scaled, X_protein_train_scaled], axis=1)
        X_val = np.concatenate([X_dna_val_scaled, X_protein_val_scaled], axis=1)

        # PCA
        pca = PCA(n_components=pca_variance, random_state=random_seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Calculate imbalance ratio
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

        # Train model
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train_pca, y_train)

        # Predictions
        y_pred = model.predict(X_val_pca)
        y_prob = model.predict_proba(X_val_pca)[:, 1]

        # Store
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Fold metrics
        fold_result = evaluate_model(y_val, y_pred, y_prob)
        for metric_name, value in fold_result.items():
            fold_metrics[metric_name].append(value)

    # Aggregate
    cv_metrics = {
        metric_name: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric_name, values in fold_metrics.items()
    }

    return cv_metrics


def train_and_evaluate_weighted_fusion(X_dna, X_protein, y, dna_weight, pca_variance, test_size, random_seed, model_type='logistic'):
    """
    Train with weighted fusion (holdout test set).

    Downweights DNA features relative to protein features before concatenation.
    DNA features are multiplied by dna_weight (0.0-1.0), protein features kept at 1.0.
    """
    # Split data first
    X_dna_train, X_dna_test, X_protein_train, X_protein_test, y_train, y_test = train_test_split(
        X_dna, X_protein, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Scale each modality separately
    scaler_dna = StandardScaler()
    scaler_protein = StandardScaler()

    X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train)
    X_dna_test_scaled = scaler_dna.transform(X_dna_test)

    X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
    X_protein_test_scaled = scaler_protein.transform(X_protein_test)

    # Apply weights (downweight DNA relative to protein)
    X_dna_train_weighted = X_dna_train_scaled * dna_weight
    X_dna_test_weighted = X_dna_test_scaled * dna_weight

    X_protein_train_weighted = X_protein_train_scaled * 1.0  # Protein weight = 1.0
    X_protein_test_weighted = X_protein_test_scaled * 1.0

    # Concatenate weighted features
    X_train = np.concatenate([X_dna_train_weighted, X_protein_train_weighted], axis=1)
    X_test = np.concatenate([X_dna_test_weighted, X_protein_test_weighted], axis=1)

    # Apply PCA on concatenated weighted features
    pca = PCA(n_components=pca_variance, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components = pca.n_components_
    explained_var = pca.explained_variance_ratio_.sum()

    # Calculate imbalance ratio
    imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

    # Train model
    model = create_model(model_type, imbalance_ratio, random_seed)
    model.fit(X_train_pca, y_train)

    # Predictions
    y_pred = model.predict(X_test_pca)
    y_prob = model.predict_proba(X_test_pca)[:, 1]

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob)

    return {
        'metrics': metrics,
        'n_components': n_components,
        'explained_variance': explained_var,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'dna_weight': dna_weight
    }


def train_and_evaluate_weighted_fusion_cv(X_dna, X_protein, y, dna_weight, pca_variance, n_folds, random_seed, model_type='logistic'):
    """
    Train with weighted fusion (cross-validation).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dna, y)):
        X_dna_train, X_dna_val = X_dna[train_idx], X_dna[val_idx]
        X_protein_train, X_protein_val = X_protein[train_idx], X_protein[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale each modality
        scaler_dna = StandardScaler()
        scaler_protein = StandardScaler()

        X_dna_train_scaled = scaler_dna.fit_transform(X_dna_train)
        X_dna_val_scaled = scaler_dna.transform(X_dna_val)

        X_protein_train_scaled = scaler_protein.fit_transform(X_protein_train)
        X_protein_val_scaled = scaler_protein.transform(X_protein_val)

        # Apply weights
        X_dna_train_weighted = X_dna_train_scaled * dna_weight
        X_dna_val_weighted = X_dna_val_scaled * dna_weight

        X_protein_train_weighted = X_protein_train_scaled * 1.0
        X_protein_val_weighted = X_protein_val_scaled * 1.0

        # Concatenate
        X_train = np.concatenate([X_dna_train_weighted, X_protein_train_weighted], axis=1)
        X_val = np.concatenate([X_dna_val_weighted, X_protein_val_weighted], axis=1)

        # PCA
        pca = PCA(n_components=pca_variance, random_state=random_seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Calculate imbalance ratio
        imbalance_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)

        # Train model
        model = create_model(model_type, imbalance_ratio, random_seed)
        model.fit(X_train_pca, y_train)

        # Predictions
        y_pred = model.predict(X_val_pca)
        y_prob = model.predict_proba(X_val_pca)[:, 1]

        # Store
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Fold metrics
        fold_result = evaluate_model(y_val, y_pred, y_prob)
        for metric_name, value in fold_result.items():
            fold_metrics[metric_name].append(value)

    # Aggregate
    cv_metrics = {
        metric_name: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric_name, values in fold_metrics.items()
    }

    return cv_metrics


def print_results(config_name, original_dims, holdout_results, cv_results):
    """
    Print formatted results for a configuration.
    """
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"{'='*70}")
    print(f"Original dimensions: {original_dims}")

    # Handle different dimensionality reduction info
    if 'pca_dna_components' in holdout_results:
        # CCA configuration
        print(f"PCA reduction: DNA {holdout_results['pca_dna_components']}, "
              f"Protein {holdout_results['pca_protein_components']}")
        print(f"CCA components: {holdout_results['cca_components']} → "
              f"Final features: {holdout_results['n_components']}")
    else:
        # Regular PCA configuration
        expl_var = holdout_results.get('explained_variance', 1.0)
        if expl_var < 1.0:
            print(f"PCA reduced dimensions: {holdout_results['n_components']} "
                  f"({expl_var:.1%} variance)")
        else:
            print(f"Reduced dimensions: {holdout_results['n_components']}")

    print(f"\nTest Set Performance (n={holdout_results['n_test']}):")
    for metric, value in holdout_results['metrics'].items():
        print(f"  {metric.upper():12s}: {value:.4f}")

    print(f"\nCross-Validation Performance (5 folds):")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        mean = cv_results[metric]['mean']
        std = cv_results[metric]['std']
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")


def print_summary_table(results):
    """
    Print comparison table across all configurations.

    Dynamically adapts to any configurations present in results dictionary.
    """
    # Get all configuration names from results (maintains insertion order)
    configs = list(results.keys())

    # Calculate table width based on number of configs
    table_width = 12 + (19 * len(configs))  # 12 for metric name + 19 per config column

    print(f"\n{'='*table_width}")
    print("SUMMARY COMPARISON (Cross-Validation Results)")
    print(f"{'='*table_width}")

    # Header
    header = f"{'Metric':<12}"
    for config in configs:
        header += f" {config:>18}"
    print(header)
    print("-" * table_width)

    # Print each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        row = f"{metric.upper():<12}"
        for config in configs:
            mean = results[config]['cv'][metric]['mean']
            std = results[config]['cv'][metric]['std']
            row += f" {mean:>7.4f}±{std:<7.4f}"
        print(row)

    print("-" * table_width)

    # Print dimensionality reduction info
    print(f"\n{'Features':<12}", end="")
    for config in configs:
        n_comp = results[config]['holdout']['n_components']
        print(f" {n_comp:>18}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train logistic regression for variant pathogenicity prediction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="analysis",
        help="Directory containing embeddings and labels"
    )
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=0.95,
        help="Variance to preserve in PCA (0.0-1.0)"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data for test set"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cca_n_components",
        type=int,
        default=20,
        help="Number of CCA canonical components"
    )
    parser.add_argument(
        "--pca_dna_components",
        type=int,
        default=30,
        help="PCA components for DNA before CCA"
    )
    parser.add_argument(
        "--pca_protein_components",
        type=int,
        default=30,
        help="PCA components for protein before CCA"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["logistic", "xgboost", "both"],
        help="Model type to use: logistic, xgboost, or both for comparison"
    )
    parser.add_argument(
        "--dna_k_features",
        type=int,
        default=500,
        help="Number of DNA features to select (for feature selection config)"
    )
    parser.add_argument(
        "--dna_weight",
        type=float,
        default=0.3,
        help="Weight for DNA features in weighted fusion (0.0-1.0, protein=1.0)"
    )
    parser.add_argument(
        "--run_feature_selection",
        action="store_true",
        help="Run feature selection experiments in addition to standard configurations"
    )

    args = parser.parse_args()

    # Load data
    X_dna, X_protein, y, variant_ids, failed_indices = load_data(args.data_dir)

    # Filter failed samples
    X_dna, X_protein, y, variant_ids = filter_failed_samples(
        X_dna, X_protein, y, variant_ids, failed_indices
    )

    # Create combined features
    X_combined = np.concatenate([X_dna, X_protein], axis=1)

    # Determine which models to run
    models_to_run = []
    if args.model == "both":
        models_to_run = ["logistic", "xgboost"]
    else:
        models_to_run = [args.model]

    # Run for each model type
    for model_type in models_to_run:
        model_name = "Logistic Regression" if model_type == "logistic" else "XGBoost"

        print(f"\n{'='*80}")
        print(f"  TRAINING WITH {model_name.upper()}")
        print(f"{'='*80}")

        # Store results for all configurations
        all_results = {}

        # Configuration 1: DNA Only
        print(f"\n{'#'*70}")
        print(f"# {model_name}: DNA Only Model")
        print(f"{'#'*70}")
        holdout_dna = train_and_evaluate_holdout(
            X_dna, y, args.pca_variance, args.test_size, args.random_seed, model_type
        )
        cv_dna = train_and_evaluate_cv(
            X_dna, y, args.pca_variance, args.n_folds, args.random_seed, model_type
        )
        print_results("DNA Only", X_dna.shape[1], holdout_dna, cv_dna)
        all_results['DNA Only'] = {'holdout': holdout_dna, 'cv': cv_dna}

        # Configuration 2: Protein Only
        print(f"\n{'#'*70}")
        print(f"# {model_name}: Protein Only Model")
        print(f"{'#'*70}")
        holdout_protein = train_and_evaluate_holdout(
            X_protein, y, args.pca_variance, args.test_size, args.random_seed, model_type
        )
        cv_protein = train_and_evaluate_cv(
            X_protein, y, args.pca_variance, args.n_folds, args.random_seed, model_type
        )
        print_results("Protein Only", X_protein.shape[1], holdout_protein, cv_protein)
        all_results['Protein Only'] = {'holdout': holdout_protein, 'cv': cv_protein}

        # Configuration 3: Concat-Unscaled (baseline)
        print(f"\n{'#'*70}")
        print(f"# {model_name}: Concat-Unscaled (DNA + Protein, no scaling) Model")
        print(f"{'#'*70}")
        holdout_combined = train_and_evaluate_holdout(
            X_combined, y, args.pca_variance, args.test_size, args.random_seed, model_type
        )
        cv_combined = train_and_evaluate_cv(
            X_combined, y, args.pca_variance, args.n_folds, args.random_seed, model_type
        )
        print_results("Concat-Unscaled", X_combined.shape[1], holdout_combined, cv_combined)
        all_results['Concat-Unscaled'] = {'holdout': holdout_combined, 'cv': cv_combined}

        # Configuration 4: Concat-Scaled (NEW)
        print(f"\n{'#'*70}")
        print(f"# {model_name}: Concat-Scaled (DNA + Protein with StandardScaler) Model")
        print(f"{'#'*70}")
        holdout_scaled = train_and_evaluate_concat_scaled(
            X_dna, X_protein, y, args.pca_variance, args.test_size, args.random_seed, model_type
        )
        cv_scaled = train_and_evaluate_concat_scaled_cv(
            X_dna, X_protein, y, args.pca_variance, args.n_folds, args.random_seed, model_type
        )
        print_results("Concat-Scaled", X_combined.shape[1], holdout_scaled, cv_scaled)
        all_results['Concat-Scaled'] = {'holdout': holdout_scaled, 'cv': cv_scaled}

        # Configuration 5: CCA Fusion (NEW)
        print(f"\n{'#'*70}")
        print(f"# {model_name}: CCA Fusion Model")
        print(f"{'#'*70}")
        holdout_cca = train_and_evaluate_cca(
            X_dna, X_protein, y, args.pca_dna_components, args.pca_protein_components,
            args.cca_n_components, args.test_size, args.random_seed, model_type
        )
        cv_cca = train_and_evaluate_cca_cv(
            X_dna, X_protein, y, args.pca_dna_components, args.pca_protein_components,
            args.cca_n_components, args.n_folds, args.random_seed, model_type
        )
        cca_dims_str = f"DNA:{args.pca_dna_components}+Protein:{args.pca_protein_components}→CCA:{args.cca_n_components*2}"
        print_results("CCA", cca_dims_str, holdout_cca, cv_cca)
        all_results['CCA'] = {'holdout': holdout_cca, 'cv': cv_cca}

        # Optional: Feature selection experiments
        if args.run_feature_selection:
            # Configuration 6: Feature Selection
            print(f"\n{'#'*70}")
            print(f"# {model_name}: Feature Selection (k={args.dna_k_features} DNA features)")
            print(f"{'#'*70}")
            holdout_fs = train_and_evaluate_feature_selection(
                X_dna, X_protein, y, args.dna_k_features, args.pca_variance,
                args.test_size, args.random_seed, model_type
            )
            cv_fs = train_and_evaluate_feature_selection_cv(
                X_dna, X_protein, y, args.dna_k_features, args.pca_variance,
                args.n_folds, args.random_seed, model_type
            )
            fs_dims_str = f"DNA-Selected:{args.dna_k_features}+Protein:{X_protein.shape[1]}"
            print_results(f"FeatureSelect-k{args.dna_k_features}", fs_dims_str, holdout_fs, cv_fs)
            all_results[f'FeatureSelect-k{args.dna_k_features}'] = {'holdout': holdout_fs, 'cv': cv_fs}

            # Configuration 7: Weighted Fusion
            print(f"\n{'#'*70}")
            print(f"# {model_name}: Weighted Fusion (DNA weight={args.dna_weight})")
            print(f"{'#'*70}")
            holdout_wf = train_and_evaluate_weighted_fusion(
                X_dna, X_protein, y, args.dna_weight, args.pca_variance,
                args.test_size, args.random_seed, model_type
            )
            cv_wf = train_and_evaluate_weighted_fusion_cv(
                X_dna, X_protein, y, args.dna_weight, args.pca_variance,
                args.n_folds, args.random_seed, model_type
            )
            wf_dims_str = f"DNA(w={args.dna_weight})+Protein(w=1.0)"
            print_results(f"WeightedFusion-w{args.dna_weight}", wf_dims_str, holdout_wf, cv_wf)
            all_results[f'WeightedFusion-w{args.dna_weight}'] = {'holdout': holdout_wf, 'cv': cv_wf}

        # Print summary comparison
        print(f"\n{'='*80}")
        print(f"  {model_name.upper()} SUMMARY")
        print(f"{'='*80}")
        print_summary_table(all_results)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
