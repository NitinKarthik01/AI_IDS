

# #!/usr/bin/env python3
# """Leakage-aware multiclass IDS ensemble: DT, RF, XGBoost and voting/stacking.

# Run from anywhere:
#   python ML/scripts/train_ensemble_multiclass.py --project-root /path/to/AI_IDS

# Dependencies: numpy pandas scikit-learn xgboost imbalanced-learn matplotlib joblib
# """
# import argparse
# import json
# import warnings
# from pathlib import Path

# import joblib
# import xgboost as xgb
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.base import BaseEstimator, ClassifierMixin, clone
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
#                              confusion_matrix, f1_score, precision_score, recall_score,
#                              roc_auc_score, ConfusionMatrixDisplay)
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier

# SEED = 42

# class SafeSMOTEClassifier(BaseEstimator, ClassifierMixin):
#     """Fit preprocessing and optional SMOTE only on training data."""
#     def __init__(self, estimator=None, variance_threshold=0.0, use_smote=False,
#                  smote_target=0, smote_max_total=300000, random_state=42):
#         self.estimator = estimator
#         self.variance_threshold = variance_threshold
#         self.use_smote = use_smote
#         self.smote_target = smote_target
#         self.smote_max_total = smote_max_total
#         self.random_state = random_state

#     def fit(self, X, y):
#         X = np.asarray(X, dtype=np.float32)
#         y = np.asarray(y, dtype=int)
#         self.classes_ = np.unique(y)
#         self.imputer_ = SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)
#         Xp = self.imputer_.fit_transform(X)
#         Xp = np.nan_to_num(Xp, nan=0, posinf=0, neginf=0)
#         self.selector_ = VarianceThreshold(threshold=self.variance_threshold)
#         Xp = self.selector_.fit_transform(Xp)
#         self.scaler_ = StandardScaler()
#         Xp = self.scaler_.fit_transform(Xp)
#         if self.use_smote and self.smote_target > 0:
#             counts = pd.Series(y).value_counts()
#             eligible = counts[counts >= 2]
#             target = {int(c): int(self.smote_target) for c, n in eligible.items()
#                       if n < self.smote_target}
#             if target:
#                 # Cap the synthetic data size to avoid memory blowups.
#                 existing = len(y)
#                 additions = sum(target[c] - int(counts[c]) for c in target)
#                 allowance = max(0, self.smote_max_total - existing)
#                 if additions > allowance:
#                     ratio = allowance / additions
#                     target = {c: int(counts[c] + np.floor((v - counts[c]) * ratio))
#                               for c, v in target.items()}
#                     target = {c: v for c, v in target.items() if v > counts[c]}
#                 if target:
#                     min_count = min(int(counts[c]) for c in target)
#                     smote = SMOTE(sampling_strategy=target, random_state=self.random_state,
#                                   k_neighbors=min(2, min_count - 1))
#                     Xp, y = smote.fit_resample(Xp, y)
#         self.model_ = clone(self.estimator)
#         if isinstance(self.model_, XGBClassifier):

#             self.model_.fit(
#                 Xp,
#                 y,
#                 verbose=False
#             )

#         else:

#             self.model_.fit(Xp, y)
#         # self.model_.fit(Xp, y)
#         return self

#     def predict_proba(self, X):
#         Xp = self.imputer_.transform(np.asarray(X, dtype=np.float32))
#         Xp = np.nan_to_num(Xp, nan=0, posinf=0, neginf=0)
#         Xp = self.scaler_.transform(self.selector_.transform(Xp))

#         if isinstance(self.model_, XGBClassifier):

#             booster = self.model_.get_booster()

#             dtest = xgb.QuantileDMatrix(Xp)

#             probabilities = booster.predict(dtest)

#         else:

#             probabilities = self.model_.predict_proba(Xp)

#         out = np.zeros((len(Xp), len(self.classes_)))

#         for j, cls in enumerate(self.model_.classes_):
#             out[:, np.where(self.classes_ == cls)[0][0]] = probabilities[:, j]
#         return out

#     def predict(self, X):
#         return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# def build_models(args, n_classes):
#     dt = DecisionTreeClassifier(criterion='gini', random_state=SEED)
#     rf = RandomForestClassifier(n_estimators=args.rf_trees, criterion='gini',
#                                 max_features='sqrt', class_weight='balanced_subsample',
#                                 n_jobs=args.jobs, random_state=SEED)
#     xgb = XGBClassifier(objective='multi:softprob', num_class=n_classes,
#                     n_estimators=args.xgb_trees, max_depth=12,
#                     learning_rate=0.03, min_child_weight=3,
#                     gamma=0, reg_alpha=0, reg_lambda=1,
#                     subsample=0.8, colsample_bytree=0.8,
#                     sampling_method='gradient_based',
#                     tree_method='hist', device='cuda', max_bin=256,
#                     eval_metric='mlogloss', random_state=SEED,
#                     n_jobs=1, verbosity=0)
#     return {
#         'DecisionTree': SafeSMOTEClassifier(dt, random_state=SEED),
#         'RandomForest': SafeSMOTEClassifier(rf, random_state=SEED),
#         'XGBoost': SafeSMOTEClassifier(xgb, variance_threshold=0.001,
#                                      use_smote=args.smote, smote_target=args.smote_target,
#                                      smote_max_total=args.smote_max_total,
#                                      random_state=SEED)
#     }


# def evaluate(y, probabilities, n_classes):
#     pred = probabilities.argmax(axis=1)
#     result = {'Accuracy': accuracy_score(y, pred),
#               'Balanced Accuracy': balanced_accuracy_score(y, pred),
#               'Macro Precision': precision_score(y, pred, average='macro', zero_division=0),
#               'Macro Recall': recall_score(y, pred, average='macro', zero_division=0),
#               'Macro F1': f1_score(y, pred, average='macro', zero_division=0),
#               'Weighted F1': f1_score(y, pred, average='weighted', zero_division=0)}
#     aucs, supports = [], []
#     for c in range(n_classes):
#         binary = (y == c).astype(int)
#         if binary.min() != binary.max():
#             aucs.append(roc_auc_score(binary, probabilities[:, c]))
#             supports.append(binary.sum())
#     result['Macro OVR AUC'] = float(np.mean(aucs)) if aucs else np.nan
#     result['Weighted OVR AUC'] = float(np.average(aucs, weights=supports)) if aucs else np.nan
#     return result, pred


# def choose_weights(models, X, y, args):
#     """Tune voting weights using only outer-training samples."""
#     counts = np.bincount(y)
#     if counts.min() < 2:
#         return np.array([1., 1., 1.])
#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
#     tr, val = next(split.split(X, y))
#     if len(np.unique(y[tr])) != len(np.unique(y)):
#         return np.array([1., 1., 1.])
#     probabilities = []
#     for estimator in models.values():
#         fitted = clone(estimator).fit(X[tr], y[tr])
#         probabilities.append(fitted.predict_proba(X[val]))
#     # Small predefined grid; weights selected solely on validation macro F1.
#     candidates = [(a, b, c) for a in (1, 2, 3) for b in (1, 2, 3) for c in (1, 2, 3)]
#     best, best_score = np.ones(3), -1
#     for weights in candidates:
#         p = np.average(probabilities, axis=0, weights=weights)
#         score = f1_score(y[val], p.argmax(axis=1), average='macro', zero_division=0)
#         if score > best_score:
#             best, best_score = np.asarray(weights, dtype=float), score
#     return best


# def save_confusion(y, pred, names, output, model):
#     cm = confusion_matrix(y, pred, labels=np.arange(len(names)))
#     pd.DataFrame(cm, index=names, columns=names).to_csv(output / f'{model}_confusion_matrix.csv')
#     fig, ax = plt.subplots(figsize=(max(11, len(names)*0.4), max(9, len(names)*0.4)))
#     ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, include_values=False,
#                                                            xticks_rotation=90, colorbar=False)
#     ax.set_title(f'{model} - out-of-fold confusion matrix')
#     fig.tight_layout()
#     fig.savefig(output / f'{model}_confusion_matrix.png', dpi=180)
#     plt.close(fig)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[2]))
#     parser.add_argument('--dataset', default=None)
#     parser.add_argument('--folds', type=int, default=5)
#     parser.add_argument('--stack-folds', type=int, default=3)
#     parser.add_argument('--rf-trees', type=int, default=300)
#     parser.add_argument('--xgb-trees', type=int, default=1000)
#     parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
#     parser.add_argument('--jobs', type=int, default=-1)
#     parser.add_argument('--smote', action='store_true', help='Apply training-only SMOTE to XGBoost')
#     parser.add_argument('--smote-target', type=int, default=5000)
#     parser.add_argument('--smote-max-total', type=int, default=300000)
#     parser.add_argument('--skip-stacking', action='store_true')
#     parser.add_argument('--no-final-fit', action='store_true')
#     args = parser.parse_args()
#     root = Path(args.project_root)
#     dataset = Path(args.dataset) if args.dataset else root / 'DATASETS' / 'MERGED_CSV_PCC_Selected.csv'
#     out = root / 'ML' / 'RESULTS' / 'Ensemble_MULTICLASS'
#     saved = root / 'ML' / 'MODELS' / 'Ensemble_MULTICLASS'
#     out.mkdir(parents=True, exist_ok=True)
#     saved.mkdir(parents=True, exist_ok=True)
#     df = pd.read_csv(dataset)
#     if 'Label' not in df.columns:
#         raise ValueError('Dataset must contain Label column')
#     names = df['Label'].astype(str).str.strip()
#     encoder = LabelEncoder()
#     y = encoder.fit_transform(names)
#     features = df.drop(columns=['Label'])
#     if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
#         raise ValueError('All feature columns must be numeric; encode categorical columns first')
#     X = features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
#     classes = encoder.classes_
#     n_classes = len(classes)
#     counts = np.bincount(y)
#     if n_classes < 3:
#         raise ValueError('Multiclass classification needs at least 3 classes')
#     if counts.min() < max(args.folds, args.stack_folds):
#         raise ValueError(f'Minority class has {counts.min()} samples; reduce --folds and/or --stack-folds')
#     (out / 'configuration.json').write_text(json.dumps(vars(args), indent=2))
#     pd.DataFrame({'Encoded': np.arange(n_classes), 'Label': classes, 'Count': counts}).to_csv(out / 'class_mapping.csv', index=False)
#     print(f'Dataset: {dataset}\nSamples: {len(y)} | Features: {X.shape[1]} | Classes: {n_classes}')
#     models = build_models(args, n_classes)
#     skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
#     labels = list(models) + ['HardVoting', 'SoftVoting', 'WeightedVoting']
#     if not args.skip_stacking:
#         labels.append('Stacking')
#     oof = {name: np.zeros((len(y), n_classes), dtype=float) for name in labels}
#     rows, weight_rows = [], []
#     for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
#         print(f'\n===== OUTER FOLD {fold}/{args.folds} =====', flush=True)
#         fitted = {}
#         for name, estimator in models.items():
#             print(f'Training {name}...', flush=True)
#             fitted[name] = clone(estimator).fit(X[tr], y[tr])
#             oof[name][te] = fitted[name].predict_proba(X[te])
#         probs = [oof[name][te] for name in models]
#         votes = np.stack([p.argmax(axis=1) for p in probs], axis=1)
#         # Deterministic tie break: highest soft-voting probability.
#         soft = np.mean(probs, axis=0)
#         hard_labels = np.array([np.argmax(np.bincount(row, minlength=n_classes) +
#                                           1e-6 * soft[i]) for i, row in enumerate(votes)])
#         oof['HardVoting'][te] = np.eye(n_classes)[hard_labels]
#         oof['SoftVoting'][te] = soft
#         print('Tuning voting weights on training fold...', flush=True)
#         weights = choose_weights(models, X[tr], y[tr], args)
#         weight_rows.append({'Fold': fold, **dict(zip(models, weights.tolist()))})
#         oof['WeightedVoting'][te] = np.average(probs, axis=0, weights=weights)
#         if not args.skip_stacking:
#             print('Training inner-CV stacking...', flush=True)
#             inner = StratifiedKFold(n_splits=args.stack_folds, shuffle=True, random_state=SEED)
#             stack = StackingClassifier(
#                 estimators=[(name, clone(est)) for name, est in models.items()],
#                 final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
#                 stack_method='predict_proba', cv=inner, n_jobs=1)
#             stack.fit(X[tr], y[tr])
#             oof['Stacking'][te] = stack.predict_proba(X[te])
#         for name in labels:
#             metrics, _ = evaluate(y[te], oof[name][te], n_classes)
#             rows.append({'Fold': fold, 'Model': name, **metrics})
#             print(f'{name:17s} accuracy={metrics["Accuracy"]:.5f} macro-F1={metrics["Macro F1"]:.5f}')
#         pd.DataFrame(rows).to_csv(out / 'fold_results.csv', index=False)
#     pd.DataFrame(weight_rows).to_csv(out / 'selected_voting_weights.csv', index=False)
#     summary = pd.DataFrame(rows).groupby('Model').agg({
#         key: ['mean', 'std'] for key in rows[0] if key not in ('Fold', 'Model')
#     })
#     summary.columns = ['_'.join(c) for c in summary.columns]
#     summary = summary.sort_values('Macro F1_mean', ascending=False)
#     summary.to_csv(out / 'model_comparison.csv')
#     print('\n===== FINAL COMPARISON =====\n', summary[['Accuracy_mean', 'Macro F1_mean', 'Balanced Accuracy_mean']])
#     for name in labels:
#         metrics, pred = evaluate(y, oof[name], n_classes)
#         save_confusion(y, pred, classes, out, name)
#         pd.DataFrame(classification_report(y, pred, labels=np.arange(n_classes),
#                   target_names=classes, output_dict=True, zero_division=0)).T.to_csv(out / f'{name}_classification_report.csv')
#         np.save(out / f'{name}_oof_probabilities.npy', oof[name])
#     ax = summary['Accuracy_mean'].sort_values().plot.barh(figsize=(10, 6), title='Mean CV Accuracy')
#     ax.set_xlabel('Accuracy')
#     plt.tight_layout(); plt.savefig(out / 'accuracy_comparison.png', dpi=200); plt.close()
#     ax = summary['Macro F1_mean'].sort_values().plot.barh(figsize=(10, 6), title='Mean CV Macro F1')
#     ax.set_xlabel('Macro F1')
#     plt.tight_layout(); plt.savefig(out / 'macro_f1_comparison.png', dpi=200); plt.close()
#     if not args.no_final_fit:
#         print('\nFitting final models on all data...', flush=True)
#         full_models = {name: clone(est).fit(X, y) for name, est in models.items()}
#         final_weights = choose_weights(models, X, y, args)
#         package = {'models': full_models, 'weights': final_weights,
#                    'label_encoder': encoder, 'feature_names': features.columns.tolist(),
#                    'method': 'WeightedVoting'}
#         joblib.dump(package, saved / 'weighted_voting_bundle.joblib')
#         if not args.skip_stacking:
#             inner = StratifiedKFold(n_splits=args.stack_folds, shuffle=True, random_state=SEED)
#             stack = StackingClassifier(estimators=[(name, clone(est)) for name, est in models.items()],
#                 final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
#                 stack_method='predict_proba', cv=inner, n_jobs=1)
#             stack.fit(X, y)
#             joblib.dump({'model': stack, 'label_encoder': encoder,
#                          'feature_names': features.columns.tolist()}, saved / 'stacking_bundle.joblib')
#     print(f'\nResults: {out}\nSaved models: {saved}')

# if __name__ == '__main__':
#     main()


