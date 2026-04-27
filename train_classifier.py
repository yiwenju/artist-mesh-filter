"""train_classifier.py — Train Stage 1 topology classifier."""

import numpy as np
import joblib
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import TRAINING_DATA_PATH, CLASSIFIER_PATH, FEATURE_COLUMNS, UV_DROPOUT_RATE, SENTINEL


UV_COLUMNS = {'uv_island_count', 'uv_faces_per_island', 'uv_island_size_entropy'}


def train():
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(str(TRAINING_DATA_PATH)).to_pandas()
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df['label'].values.astype(np.int32)

    # Apply UV dropout: mask UV features to SENTINEL for a fraction of positives
    uv_col_indices = [i for i, c in enumerate(FEATURE_COLUMNS) if c in UV_COLUMNS]
    pos_mask = y == 1
    rng = np.random.default_rng(42)
    dropout_mask = pos_mask & (rng.random(len(y)) < UV_DROPOUT_RATE)
    for col_idx in uv_col_indices:
        X[dropout_mask, col_idx] = SENTINEL
    n_dropped = dropout_mask.sum()
    print(f"UV dropout: masked {n_dropped}/{pos_mask.sum()} positives ({UV_DROPOUT_RATE:.0%})")

    print(f"Data: {len(y)} samples ({y.sum()} pos, {(1 - y).sum()} neg)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        'gbm': GradientBoostingClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        'rf': RandomForestClassifier(
            n_estimators=500, max_depth=12, random_state=42,
            class_weight='balanced',
        ),
    }

    best_name, best_clf, best_auc = None, None, 0.0

    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')
        auc = scores.mean()
        print(f"  {name}: AUC = {auc:.4f} +/- {scores.std():.4f}")
        if auc > best_auc:
            best_name, best_clf, best_auc = name, clf, auc

    best_clf.fit(X, y)
    print(f"\nBest: {best_name} (AUC={best_auc:.4f})")

    if hasattr(best_clf, 'feature_importances_'):
        imp = best_clf.feature_importances_
        uv_cols = {'uv_island_count', 'uv_faces_per_island', 'uv_island_size_entropy'}

        print("\nFeature importance:")
        for col, val in sorted(zip(FEATURE_COLUMNS, imp), key=lambda x: -x[1]):
            flag = ''
            if col in uv_cols:
                flag = '  <- UV'
            elif col == 'quad_ratio':
                flag = '  <- SENTINEL at GLB inference'
            print(f"  {col:35s} {val:.4f}{flag}")

        uv_total = sum(v for c, v in zip(FEATURE_COLUMNS, imp) if c in uv_cols)
        if uv_total > 0.30:
            print(f"\n  WARNING: UV features = {uv_total:.0%} of importance.")
            print(f"    Increase UV_DROPOUT_RATE in config.py and retrain.")

        qr_idx = FEATURE_COLUMNS.index('quad_ratio')
        if imp[qr_idx] > 0.15:
            print(f"\n  WARNING: quad_ratio = {imp[qr_idx]:.0%} of importance.")
            print(f"    This feature is SENTINEL for all GLB files at inference.")
            print(f"    Consider dropping it from FEATURE_COLUMNS and retraining.")

    joblib.dump(best_clf, str(CLASSIFIER_PATH))
    print(f"\nSaved: {CLASSIFIER_PATH}")


if __name__ == '__main__':
    train()
