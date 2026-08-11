"""Small CPU tests for the method-defining MV-CoL operations."""

from __future__ import annotations

import unittest

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from main import load_config, validate_config
from prompts.behavior_view import build_messages as build_behavior_messages
from prompts.common import PromptSpec
from prompts.definition_view import build_messages as build_definition_messages
from prompts.holistic_view import build_messages as build_holistic_messages
from src.feature_fusion import fuse_three_views
from src.losses import (
    PerViewJointLoss,
    batch_hard_triplet_loss,
    supervised_contrastive_loss,
)
from src.train_ensemble import _fold_safe_pipeline, _make_stacking, _suggest_parameters
from src.train_lora import pool_last_valid_token


class JointLossTests(unittest.TestCase):
    def test_supcon_excludes_self_and_uses_all_non_anchor_samples(self) -> None:
        projected = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, 0.2]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])
        temperature = 0.25
        normalized = F.normalize(projected, p=2, dim=1)
        similarities = normalized @ normalized.T / temperature
        anchor_losses = []
        for anchor in range(projected.shape[0]):
            non_anchor = [index for index in range(projected.shape[0]) if index != anchor]
            positives = [index for index in non_anchor if labels[index] == labels[anchor]]
            log_denominator = torch.logsumexp(similarities[anchor, non_anchor], dim=0)
            anchor_losses.append(
                -torch.stack(
                    [similarities[anchor, positive] - log_denominator for positive in positives]
                ).mean()
            )
        expected = torch.stack(anchor_losses).mean()

        actual = supervised_contrastive_loss(projected, labels, temperature)
        torch.testing.assert_close(actual, expected)

    def test_joint_loss_is_the_mean_of_three_per_view_losses(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        logits = [
            torch.tensor(
                [
                    [2.0 + view, -0.5],
                    [1.5 + view, 0.0],
                    [-0.5, 1.5 + view],
                    [0.0, 2.0 + view],
                ]
            )
            for view in (0.0, 0.2, 0.4)
        ]
        projections = [
            torch.tensor(
                [
                    [1.0, 0.1 + view],
                    [0.8, 0.2 + view],
                    [0.1 + view, 1.0],
                    [0.2 + view, 0.8],
                ]
            )
            for view in (0.0, 0.1, 0.2)
        ]
        lambda_1 = 0.7
        lambda_2 = 0.3
        temperature = 0.2
        margin = 0.5
        criterion = PerViewJointLoss(
            supcon_weight=lambda_1,
            triplet_weight=lambda_2,
            temperature=temperature,
            margin=margin,
            triplet_mining="batch_hard",
        )

        actual, components = criterion(logits, projections, labels)
        expected_ce = torch.stack([F.cross_entropy(value, labels) for value in logits]).mean()
        expected_supcon = torch.stack(
            [supervised_contrastive_loss(value, labels, temperature) for value in projections]
        ).mean()
        expected_triplet = torch.stack(
            [batch_hard_triplet_loss(value, labels, margin) for value in projections]
        ).mean()
        expected = expected_ce + lambda_1 * expected_supcon + lambda_2 * expected_triplet

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(components["ce"], expected_ce)
        torch.testing.assert_close(components["supcon"], expected_supcon)
        torch.testing.assert_close(components["triplet"], expected_triplet)

    def test_joint_loss_rejects_any_number_other_than_three_views(self) -> None:
        criterion = PerViewJointLoss(
            supcon_weight=0.7,
            triplet_weight=0.3,
            temperature=0.07,
            margin=1.0,
            triplet_mining="batch_hard",
        )
        labels = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "exactly three"):
            criterion(
                [torch.zeros(2, 2), torch.zeros(2, 2)],
                [torch.zeros(2, 2), torch.zeros(2, 2)],
                labels,
            )


class RepresentationTests(unittest.TestCase):
    def test_last_valid_token_supports_right_and_left_padding(self) -> None:
        hidden = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
        right_padded = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        left_padded = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

        torch.testing.assert_close(pool_last_valid_token(hidden, right_padded), hidden[[0, 1], [1, 2]])
        torch.testing.assert_close(pool_last_valid_token(hidden, left_padded), hidden[[0, 1], [3, 3]])

    def test_feature_fusion_uses_the_paper_order_without_normalization(self) -> None:
        h_a = np.array([[3.0, 4.0]], dtype=np.float32)
        h_b = np.array([[1.0, 2.0]], dtype=np.float32)
        h_c = np.array([[2.0, 6.0]], dtype=np.float32)
        actual = fuse_three_views(h_a, h_b, h_c)
        expected = np.concatenate(
            [
                h_a,
                h_b,
                h_c,
                np.abs(h_a - h_b),
                np.abs(h_a - h_c),
                np.abs(h_b - h_c),
                h_a * h_b,
                h_a * h_c,
                h_b * h_c,
                (h_a + h_b + h_c) / 3.0,
            ],
            axis=1,
        )

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, :2], h_a)


class LeakageBoundaryTests(unittest.TestCase):
    def test_infeasible_model_subset_raises_optuna_trial_pruned(self) -> None:
        class DisabledModelsTrial:
            def suggest_categorical(self, name: str, choices: list[object]) -> object:
                return False if name.startswith("use_") else choices[0]

            def suggest_float(self, name: str, low: float, high: float, **_: object) -> float:
                del name, high
                return low

        config = {
            "candidate_models": ["random_forest", "extra_trees", "knn"],
            "minimum_selected_models": 3,
            "search_space": {"meta_c": [0.1, 2.0]},
        }
        with self.assertRaises(optuna.TrialPruned):
            _suggest_parameters(DisabledModelsTrial(), config, [2], [1])

    def test_every_base_estimator_owns_its_fold_local_pipeline(self) -> None:
        classifier = LogisticRegression(max_iter=100)
        pipeline = _fold_safe_pipeline(
            classifier,
            pca_components=2,
            smote_k=1,
            seed=7,
            pca_config={"svd_solver": "randomized"},
            smote_config={
                "sampling_strategy": "auto",
                "m_neighbors": 3,
                "kind": "borderline-1",
            },
        )

        self.assertEqual(
            [name for name, _ in pipeline.steps],
            ["scaler", "pca", "smote", "classifier"],
        )
        self.assertEqual(pipeline.named_steps["pca"].n_components, 2)
        self.assertEqual(pipeline.named_steps["smote"].k_neighbors, 1)

    def test_stacking_uses_logistic_regression_as_meta_classifier(self) -> None:
        config = {
            "candidate_models": ["random_forest", "extra_trees", "knn"],
            "minimum_selected_models": 3,
            "cv_folds": 5,
            "n_jobs": 1,
            "pca": {"svd_solver": "randomized"},
            "borderline_smote": {
                "sampling_strategy": "auto",
                "m_neighbors": 3,
                "kind": "borderline-1",
            },
            "classifier_defaults": {
                "random_forest": {
                    "n_estimators": 10,
                    "class_weight": "balanced_subsample",
                    "n_jobs": 1,
                },
                "extra_trees": {
                    "n_estimators": 10,
                    "class_weight": "balanced",
                    "n_jobs": 1,
                },
                "knn": {"weights": "distance"},
                "meta_logistic_regression": {"class_weight": "balanced", "max_iter": 100},
            },
        }
        params = {
            "use_random_forest": True,
            "use_extra_trees": True,
            "use_knn": True,
            "rf_max_depth": 8,
            "extra_trees_max_depth": 8,
            "knn_neighbors": 3,
            "pca_components": 2,
            "smote_k": 1,
            "meta_c": 1.0,
        }
        labels = np.repeat(np.arange(3), 10)

        stacking = _make_stacking(params, config, num_classes=3, seed=7, y=labels)

        self.assertIsInstance(stacking.final_estimator, LogisticRegression)
        self.assertEqual(len(stacking.estimators), 3)
        for _, estimator in stacking.estimators:
            self.assertEqual(
                [name for name, _ in estimator.steps],
                ["scaler", "pca", "smote", "classifier"],
            )


class DatasetPromptConfigTests(unittest.TestCase):
    CONFIG_NAMES = ("question", "answer", "opinion", "urgency", "icap_psy", "coi_ast")

    def test_all_six_task_configs_are_valid_and_render_all_three_views(self) -> None:
        for config_name in self.CONFIG_NAMES:
            with self.subTest(config=config_name):
                config = load_config(f"configs/{config_name}.yaml")
                validate_config(config, require_data=False)
                spec = PromptSpec.from_dataset_config(config["data"])
                self.assertEqual(len(spec.labels), int(config["data"]["num_labels"]))
                for builder in (
                    build_definition_messages,
                    build_behavior_messages,
                    build_holistic_messages,
                ):
                    messages = builder("sample input", spec)
                    rendered = "\n".join(message["content"] for message in messages)
                    self.assertIn(config["data"]["task_description"], rendered)
                    for label in config["data"]["output_labels"]:
                        self.assertIn(label, rendered)

    def test_projection_and_triplet_defaults_are_explicit(self) -> None:
        config = load_config("configs/question.yaml")
        self.assertEqual(config["training"]["projection_head_mode"], "separate")
        self.assertEqual(config["training"]["triplet_mining"], "batch_hard")


if __name__ == "__main__":
    unittest.main()
