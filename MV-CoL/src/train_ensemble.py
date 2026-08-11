"""Leakage-safe Optuna model selection and stacking for MV-CoL."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

from src.evaluate import evaluate_predictions, save_metrics


PAPER_CANDIDATE_MODELS = (
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "extra_trees",
    "knn",
    "nusvc",
)


def _load_split(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = directory / f"{name}_fused.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Fused feature file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return data["features"], data["labels"].astype(np.int64), data["ids"]


def _number_of_cv_splits(y: np.ndarray, requested: int) -> int:
    counts = np.bincount(y)
    present_counts = counts[counts > 0]
    if len(present_counts) < 2:
        raise ValueError("stacking requires at least two classes")
    n_splits = min(int(requested), int(present_counts.min()))
    if n_splits < 2:
        raise ValueError("stacking CV requires at least two samples in every class")
    return n_splits


def _safe_pca_candidates(config: dict[str, Any], x: np.ndarray, y: np.ndarray) -> list[int]:
    """Create PCA choices feasible inside the smallest Stacking training fold."""
    n_splits = _number_of_cv_splits(y, int(config.get("cv_folds", 5)))
    smallest_fold_size = x.shape[0] - math.ceil(x.shape[0] / n_splits)
    upper = min(int(config["pca"]["max_components"]), x.shape[1], smallest_fold_size - 1)
    lower = min(int(config["pca"]["min_components"]), upper)
    step = int(config["pca"]["step"])
    if upper < 1 or step < 1:
        raise ValueError("PCA search range is infeasible for the training folds")
    candidates = list(range(lower, upper + 1, step))
    if candidates[-1] != upper:
        candidates.append(upper)
    return sorted(set(candidates))


def _safe_smote_choices(config: dict[str, Any], y: np.ndarray) -> list[int]:
    n_splits = _number_of_cv_splits(y, int(config.get("cv_folds", 5)))
    minimum_class_count = int(np.bincount(y)[np.bincount(y) > 0].min())
    minimum_fold_class_count = minimum_class_count - math.ceil(minimum_class_count / n_splits)
    choices = [
        int(value)
        for value in config["borderline_smote"]["k_neighbors_choices"]
        if 1 <= int(value) < minimum_fold_class_count
    ]
    if not choices:
        raise ValueError("no configured Borderline-SMOTE k is feasible inside CV folds")
    return choices


def _suggest_parameters(
    trial: Any,
    config: dict[str, Any],
    pca_choices: list[int],
    smote_choices: list[int],
) -> dict[str, Any]:
    space = config["search_space"]
    candidates = list(config["candidate_models"])
    unknown = sorted(set(candidates).difference(PAPER_CANDIDATE_MODELS))
    if unknown:
        raise ValueError(f"unknown candidate models: {unknown}")

    params: dict[str, Any] = {
        "pca_components": trial.suggest_categorical("pca_components", pca_choices),
        "smote_k": trial.suggest_categorical("smote_k", smote_choices),
        "meta_c": trial.suggest_float("meta_c", *map(float, space["meta_c"]), log=True),
    }
    selected: list[str] = []
    for name in candidates:
        enabled = trial.suggest_categorical(f"use_{name}", [True, False])
        params[f"use_{name}"] = enabled
        if enabled:
            selected.append(name)
    if len(selected) < int(config.get("minimum_selected_models", 3)):
        raise optuna.TrialPruned("too few base classifiers selected")

    if "xgboost" in selected:
        params["xgb_learning_rate"] = trial.suggest_float(
            "xgb_learning_rate", *map(float, space["xgb_learning_rate"]), log=True
        )
        params["xgb_max_depth"] = trial.suggest_int(
            "xgb_max_depth", *map(int, space["xgb_max_depth"])
        )
    if "lightgbm" in selected:
        params["lgbm_learning_rate"] = trial.suggest_float(
            "lgbm_learning_rate", *map(float, space["lgbm_learning_rate"]), log=True
        )
        params["lgbm_num_leaves"] = trial.suggest_int(
            "lgbm_num_leaves", *map(int, space["lgbm_num_leaves"])
        )
    if "catboost" in selected:
        params["catboost_learning_rate"] = trial.suggest_float(
            "catboost_learning_rate", *map(float, space["catboost_learning_rate"]), log=True
        )
        params["catboost_depth"] = trial.suggest_int(
            "catboost_depth", *map(int, space["catboost_depth"])
        )
    if "random_forest" in selected:
        params["rf_max_depth"] = trial.suggest_int(
            "rf_max_depth", *map(int, space["rf_max_depth"])
        )
    if "extra_trees" in selected:
        params["extra_trees_max_depth"] = trial.suggest_int(
            "extra_trees_max_depth", *map(int, space["extra_trees_max_depth"])
        )
    if "knn" in selected:
        params["knn_neighbors"] = trial.suggest_int(
            "knn_neighbors", *map(int, space["knn_neighbors"]), step=2
        )
    if "nusvc" in selected:
        params["nusvc_nu"] = trial.suggest_float(
            "nusvc_nu", *map(float, space["nusvc_nu"])
        )
    return params


def _selected_models(params: dict[str, Any], candidates: list[str]) -> list[str]:
    return [name for name in candidates if bool(params.get(f"use_{name}", False))]


def _build_classifier(
    name: str,
    params: dict[str, Any],
    num_classes: int,
    seed: int,
    defaults: dict[str, Any],
) -> Any:
    """Instantiate one model from the paper's candidate classifier pool.

    Estimator counts and other untuned constants come from the YAML's public
    reference defaults; they are not paper-final dataset-specific parameters.
    """
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(defaults["n_estimators"]),
            learning_rate=float(params["xgb_learning_rate"]),
            max_depth=int(params["xgb_max_depth"]),
            subsample=float(defaults["subsample"]),
            colsample_bytree=float(defaults["colsample_bytree"]),
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            random_state=seed,
            n_jobs=int(defaults["n_jobs"]),
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(defaults["n_estimators"]),
            learning_rate=float(params["lgbm_learning_rate"]),
            num_leaves=int(params["lgbm_num_leaves"]),
            class_weight=defaults["class_weight"],
            verbosity=-1,
            random_state=seed,
            n_jobs=int(defaults["n_jobs"]),
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(defaults["iterations"]),
            learning_rate=float(params["catboost_learning_rate"]),
            depth=int(params["catboost_depth"]),
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=int(defaults["thread_count"]),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(defaults["n_estimators"]),
            max_depth=int(params["rf_max_depth"]),
            class_weight=defaults["class_weight"],
            random_state=seed,
            n_jobs=int(defaults["n_jobs"]),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(defaults["n_estimators"]),
            max_depth=int(params["extra_trees_max_depth"]),
            class_weight=defaults["class_weight"],
            random_state=seed,
            n_jobs=int(defaults["n_jobs"]),
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params["knn_neighbors"]),
            weights=str(defaults["weights"]),
        )
    if name == "nusvc":
        return NuSVC(
            nu=float(params["nusvc_nu"]),
            probability=bool(defaults["probability"]),
            random_state=seed,
        )
    raise ValueError(f"unsupported candidate model: {name}")


def _fold_safe_pipeline(
    classifier: Any,
    pca_components: int,
    smote_k: int,
    seed: int,
    pca_config: dict[str, Any],
    smote_config: dict[str, Any],
) -> ImbalancedPipeline:
    """Fit scaler, PCA, and SMOTE only inside each estimator's training fold."""
    return ImbalancedPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=int(pca_components),
                    svd_solver=str(pca_config["svd_solver"]),
                    random_state=seed,
                ),
            ),
            (
                "smote",
                BorderlineSMOTE(
                    sampling_strategy=smote_config["sampling_strategy"],
                    k_neighbors=int(smote_k),
                    m_neighbors=int(smote_config["m_neighbors"]),
                    kind=str(smote_config["kind"]),
                    random_state=seed,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def _make_stacking(
    params: dict[str, Any],
    config: dict[str, Any],
    num_classes: int,
    seed: int,
    y: np.ndarray,
) -> StackingClassifier:
    n_splits = _number_of_cv_splits(y, int(config.get("cv_folds", 5)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = list(config["candidate_models"])
    classifier_defaults = config["classifier_defaults"]
    selected = _selected_models(params, candidates)
    if len(selected) < int(config.get("minimum_selected_models", 3)):
        raise ValueError("best configuration contains too few base classifiers")

    estimators = [
        (
            name,
            _fold_safe_pipeline(
                _build_classifier(
                    name,
                    params,
                    num_classes,
                    seed,
                    classifier_defaults[name],
                ),
                int(params["pca_components"]),
                int(params["smote_k"]),
                seed,
                config["pca"],
                config["borderline_smote"],
            ),
        )
        for name in selected
    ]
    meta_defaults = classifier_defaults["meta_logistic_regression"]
    meta_classifier = LogisticRegression(
        C=float(params["meta_c"]),
        class_weight=meta_defaults["class_weight"],
        max_iter=int(meta_defaults["max_iter"]),
        random_state=seed,
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_classifier,
        stack_method="predict_proba",
        passthrough=False,
        cv=cv,
        n_jobs=int(config.get("n_jobs", -1)),
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    data_cfg = config["data"]
    feature_dir = Path(config["features"]["fused_dir"])
    ensemble_cfg = config["ensemble"]
    output_dir = Path(ensemble_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(ensemble_cfg["n_trials"]) < 30:
        raise ValueError("ensemble.n_trials must be at least 30")

    # Only train and validation are loaded during hyperparameter selection.
    # The independent test split is deliberately not accessed until the final
    # development-set model has been selected and fitted.
    x_train, y_train, _ = _load_split(feature_dir, "train")
    x_validation, y_validation, _ = _load_split(feature_dir, "validation")
    pca_choices = _safe_pca_candidates(ensemble_cfg, x_train, y_train)
    smote_choices = _safe_smote_choices(ensemble_cfg, y_train)
    num_classes = len(data_cfg["labels"])

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_parameters(trial, ensemble_cfg, pca_choices, smote_choices)
        try:
            model = _make_stacking(params, ensemble_cfg, num_classes, seed, y_train)
            model.fit(x_train, y_train)
            validation_predictions = model.predict(x_validation)
            validation_accuracy = float(accuracy_score(y_validation, validation_predictions))
            validation_weighted_f1 = float(
                f1_score(y_validation, validation_predictions, average="weighted", zero_division=0)
            )
            trial.set_user_attr("validation_accuracy", validation_accuracy)
            print(
                f"Trial {trial.number}: weighted_f1={validation_weighted_f1:.4f}, "
                f"accuracy={validation_accuracy:.4f}"
            )
            # Weighted-F1 is the sole Optuna optimization target. Accuracy is
            # diagnostic only and never used for best-trial selection.
            return validation_weighted_f1
        except (ValueError, RuntimeError) as error:
            raise optuna.TrialPruned(str(error)) from error

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    timeout = ensemble_cfg.get("timeout_seconds")
    study.optimize(
        objective,
        n_trials=int(ensemble_cfg["n_trials"]),
        timeout=None if timeout is None else int(timeout),
    )
    if not study.best_trials:
        raise RuntimeError("Optuna did not complete any feasible trial")
    selected_configuration = dict(study.best_trial.params)
    print(f"Best validation Weighted-F1: {study.best_value:.4f}")
    print(f"Selected local-run configuration: {selected_configuration}")

    # Refit the selected stage-two classifier on train + validation. All
    # preprocessing and resampling remain inside the Stacking training folds.
    x_development = np.concatenate([x_train, x_validation], axis=0)
    y_development = np.concatenate([y_train, y_validation], axis=0)
    final_model = _make_stacking(
        selected_configuration,
        ensemble_cfg,
        num_classes,
        seed,
        y_development,
    )
    final_model.fit(x_development, y_development)

    # First and only access to the independent test split in this function.
    x_test, y_test, test_ids = _load_split(feature_dir, "test")
    predictions = final_model.predict(x_test).astype(np.int64)
    metrics = evaluate_predictions(y_test, predictions, data_cfg["labels"])
    metrics["validation_objective"] = "weighted_f1"
    metrics["best_validation_weighted_f1"] = float(study.best_value)
    save_metrics(metrics, output_dir / "test_metrics.json")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        ids=test_ids,
        labels=y_test,
        predictions=predictions,
    )

    # Disabled by default. If a user explicitly enables it, these files describe
    # only that user's local run; no paper-trained model or best parameters are
    # distributed with this reference repository.
    if bool(ensemble_cfg.get("save_local_search_artifacts", False)):
        import joblib

        with (output_dir / "local_best_configuration.json").open("w", encoding="utf-8") as handle:
            json.dump(selected_configuration, handle, indent=2)
        joblib.dump(final_model, output_dir / "local_stacking_model.joblib")
    return metrics
