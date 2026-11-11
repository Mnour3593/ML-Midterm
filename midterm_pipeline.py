from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEEDS = [42, 52, 62]
REPRESENTATIONS = ["unscaled", "scaled"]
FIGURES_DIR = Path("figures")


def _classification_metrics(y_true, prediction, probabilities, dataset_name):
    result = {
        "accuracy": accuracy_score(y_true, prediction),
        "precision_macro": precision_score(
            y_true, prediction, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, prediction, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, prediction, average="macro", zero_division=0),
    }
    if dataset_name == "breast_cancer":
        malignant_target = (np.asarray(y_true) == 0).astype(int)
        result["roc_auc"] = roc_auc_score(malignant_target, probabilities[:, 0])
    else:
        result["roc_auc"] = roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="macro"
        )
    return result


def _fit_classifier(algorithm, parameter, seed):
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(max_depth=parameter, random_state=seed)
    if algorithm == "knn":
        return KNeighborsClassifier(
            n_neighbors=parameter, weights="uniform", metric="euclidean"
        )
    return GaussianNB()


def _parameter_label(algorithm, parameter):
    if algorithm == "decision_tree":
        return f"max_depth={parameter}"
    if algorithm == "knn":
        return f"n_neighbors={parameter}"
    return "default"


def _save_combined(filename, frame):
    path = Path(filename)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old[old["dataset"] != frame["dataset"].iloc[0]], frame])
    frame.to_csv(path, index=False)


def _plot_classifier_comparison(summary, dataset_name, algorithm):
    subset = summary[summary["algorithm"] == algorithm].copy()
    if subset.empty:
        return
    x_column = "parameter_value"
    plt.figure(figsize=(9, 5))
    for representation in REPRESENTATIONS:
        values = subset[subset["representation"] == representation]
        plt.errorbar(
            values[x_column],
            values["train_f1_macro_mean"],
            yerr=values["train_f1_macro_std"],
            marker="o",
            label=f"{representation} training",
        )
        plt.errorbar(
            values[x_column],
            values["val_f1_macro_mean"],
            yerr=values["val_f1_macro_std"],
            marker="s",
            linestyle="--",
            label=f"{representation} validation",
        )
    plt.xlabel("Tree max depth" if algorithm == "decision_tree" else "KNN neighbours")
    plt.ylabel("Macro-F1")
    plt.title(f"{dataset_name}: {algorithm.replace('_', ' ').title()} comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{dataset_name}_{algorithm}_comparison.png", dpi=150
    )
    plt.show()


def run_dataset(dataset_name, X, y, target_names, n_clusters, plot_features):
    """Run Sections 4-10 for one dataset and export the required results."""
    FIGURES_DIR.mkdir(exist_ok=True)
    X = X.copy()
    y = pd.Series(y, index=X.index, name="target")

    print(f"\n{'=' * 80}\n4-10. {dataset_name.upper()}\n{'=' * 80}")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    splits = {}
    split_rows = []
    preprocessing = {}
    for seed in SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.125,
            stratify=y_train_val,
            random_state=seed,
        )
        splits[seed] = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
        }
        for subset_name, subset_X, subset_y in (
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ):
            split_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "subset": subset_name,
                    "n_samples": len(subset_X),
                    "percentage_of_dataset": 100 * len(subset_X) / len(X),
                    "class_counts": dict(subset_y.value_counts().sort_index()),
                    "indices": list(subset_X.index),
                }
            )
        scaler = StandardScaler()
        scaler.fit(X_train)
        preprocessing[seed] = {
            "scaler": scaler,
            "X_train_scaled": pd.DataFrame(
                scaler.transform(X_train), index=X_train.index, columns=X.columns
            ),
            "X_val_scaled": pd.DataFrame(
                scaler.transform(X_val), index=X_val.index, columns=X.columns
            ),
            "X_test_scaled": pd.DataFrame(
                scaler.transform(X_test), index=X_test.index, columns=X.columns
            ),
        }

    split_table = pd.DataFrame(split_rows)
    print("\n4.2 Split summary, class counts and original indices")
    print(split_table.to_string(index=False))
    assert len(set(X_train_val.index) & set(X_test.index)) == 0
    assert all(
        len(set(item["X_train"].index) & set(item["X_val"].index)) == 0
        for item in splits.values()
    )
    assert all(
        set(item["X_train"].index) | set(item["X_val"].index)
        == set(X_train_val.index)
        for item in splits.values()
    )
    print(f"Fixed test indices: {len(X_test.index)} rows, unchanged for all seeds")

    print("\n5. Preprocessing")
    print("No missing values were found; no imputation was added.")
    print("StandardScaler is fitted separately on each training subset only.")
    seed_42_train = splits[42]["X_train"]
    before = seed_42_train.iloc[:5]
    after = preprocessing[42]["X_train_scaled"].iloc[:5]
    display_frame = pd.concat({"before": before, "after": after}, axis=1)
    print("Seed 42 first five training rows before and after scaling:")
    print(display_frame.to_string())

    classifier_rows = []
    fitted_classifiers = {}
    configurations = (
        [("decision_tree", depth) for depth in [2, 3, 5, 10, None]]
        + [("knn", neighbors) for neighbors in [1, 3, 5, 7, 11]]
        + [("gaussian_nb", "default")]
    )
    for seed in SEEDS:
        split = splits[seed]
        for representation in REPRESENTATIONS:
            if representation == "unscaled":
                X_train_rep, X_val_rep = split["X_train"], split["X_val"]
            else:
                X_train_rep = preprocessing[seed]["X_train_scaled"]
                X_val_rep = preprocessing[seed]["X_val_scaled"]
            for algorithm, parameter in configurations:
                model = _fit_classifier(algorithm, parameter, seed)
                model.fit(X_train_rep, split["y_train"])
                train_prediction = model.predict(X_train_rep)
                val_prediction = model.predict(X_val_rep)
                train_probability = model.predict_proba(X_train_rep)
                val_probability = model.predict_proba(X_val_rep)
                train_metrics = _classification_metrics(
                    split["y_train"],
                    train_prediction,
                    train_probability,
                    dataset_name,
                )
                val_metrics = _classification_metrics(
                    split["y_val"],
                    val_prediction,
                    val_probability,
                    dataset_name,
                )
                parameter_label = _parameter_label(algorithm, parameter)
                configuration_id = (
                    f"{dataset_name}_{algorithm}_{representation}_"
                    f"{str(parameter).replace(' ', '')}"
                )
                classifier_rows.append(
                    {
                        "dataset": dataset_name,
                        "experiment_id": (
                            f"{configuration_id}_s{seed}"
                        ),
                        "configuration_id": configuration_id,
                        "seed": seed,
                        "algorithm": algorithm,
                        "representation": representation,
                        "parameter": parameter_label,
                        "parameter_value": (
                            -1 if parameter is None else (
                                0 if parameter == "default" else parameter
                            )
                        ),
                        **{f"train_{key}": value for key, value in train_metrics.items()},
                        **{f"val_{key}": value for key, value in val_metrics.items()},
                    }
                )
                if seed == 42:
                    fitted_classifiers[(algorithm, representation, parameter_label)] = model

    classifier_runs = pd.DataFrame(classifier_rows)
    group_columns = [
        "dataset",
        "configuration_id",
        "algorithm",
        "representation",
        "parameter",
        "parameter_value",
    ]
    metric_columns = [
        "train_accuracy", "train_precision_macro", "train_recall_macro",
        "train_f1_macro", "train_roc_auc", "val_accuracy",
        "val_precision_macro", "val_recall_macro", "val_f1_macro", "val_roc_auc",
    ]
    classifier_summary = classifier_runs.groupby(group_columns, as_index=False)[
        metric_columns
    ].agg(["mean", "std"])
    classifier_summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in classifier_summary.columns
    ]
    classifier_summary = classifier_summary.reset_index(drop=True)
    selected_row = classifier_summary.sort_values(
        [
            "val_f1_macro_mean",
            "val_accuracy_mean",
            "val_f1_macro_std",
            "configuration_id",
        ],
        ascending=[False, False, True, True],
    ).iloc[0]
    selected_key = (
        selected_row["algorithm"],
        selected_row["representation"],
        selected_row["parameter"],
    )
    selected_model = fitted_classifiers[selected_key]
    selected_preprocessing = preprocessing[42]
    if selected_row["representation"] == "scaled":
        test_features = selected_preprocessing["X_test_scaled"]
    else:
        test_features = X_test
    test_prediction = selected_model.predict(test_features)
    test_probability = selected_model.predict_proba(test_features)
    test_metrics = _classification_metrics(
        y_test, test_prediction, test_probability, dataset_name
    )
    _plot_classifier_comparison(classifier_summary, dataset_name, "decision_tree")
    _plot_classifier_comparison(classifier_summary, dataset_name, "knn")
    classifier_test = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "selected_configuration_id": selected_row["configuration_id"],
                "selected_algorithm": selected_row["algorithm"],
                "selected_representation": selected_row["representation"],
                "selected_parameter": selected_row["parameter"],
                "evaluation_seed": 42,
                "validation_f1_mean": selected_row["val_f1_macro_mean"],
                "validation_f1_std": selected_row["val_f1_macro_std"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        ]
    )
    print("\n7-9. Classifier results")
    print(f"Classifier runs: {len(classifier_runs)}")
    print("Selected configuration:")
    print(classifier_test.T)
    print(classification_report(y_test, test_prediction, target_names=target_names, zero_division=0))
    error_indices = y_test.index[np.asarray(test_prediction) != np.asarray(y_test)]
    if len(error_indices):
        print("Actual misclassified test rows:")
        print(
            pd.DataFrame(
                {
                    "actual": y_test.loc[error_indices],
                    "predicted": pd.Series(
                        test_prediction, index=y_test.index
                    ).loc[error_indices],
                }
            )
        )
    else:
        print("No test misclassifications were made by the selected classifier.")

    cm = confusion_matrix(y_test, test_prediction, labels=list(range(len(target_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=target_names, yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{dataset_name}: selected classifier test confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{dataset_name}_classifier_confusion_matrix.png", dpi=150)
    plt.show()
    if dataset_name == "breast_cancer":
        binary_test_target = (np.asarray(y_test) == 0).astype(int)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_test_target, test_probability[:, 0]
        )
        plt.figure(figsize=(6, 5))
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"AUC = {test_metrics['roc_auc']:.3f}",
        )
        positive_name = target_names[0]
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name}: {positive_name} ROC curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{dataset_name}_classifier_roc.png", dpi=150)
        plt.show()
        true_positive = cm[0, 0]
        false_positive = cm[1, 0]
        false_negative = cm[0, 1]
        precision_manual = true_positive / (true_positive + false_positive)
        recall_manual = true_positive / (true_positive + false_negative)
        f1_manual = 2 * precision_manual * recall_manual / (
            precision_manual + recall_manual
        )
        print(
            "Manual malignant metrics:",
            precision_manual,
            recall_manual,
            f1_manual,
        )
    else:
        plt.figure(figsize=(7, 5))
        for class_value, class_name in enumerate(target_names):
            one_vs_rest = (np.asarray(y_test) == class_value).astype(int)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                one_vs_rest, test_probability[:, class_value]
            )
            class_auc = roc_auc_score(one_vs_rest, test_probability[:, class_value])
            plt.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{class_name} AUC = {class_auc:.3f}",
            )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{dataset_name}: one-versus-rest ROC curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{dataset_name}_classifier_roc.png", dpi=150)
        plt.show()
        selected_class = 0
        true_positive = cm[selected_class, selected_class]
        false_positive = cm[:, selected_class].sum() - true_positive
        false_negative = cm[selected_class, :].sum() - true_positive
        precision_manual = true_positive / (true_positive + false_positive)
        recall_manual = true_positive / (true_positive + false_negative)
        f1_manual = 2 * precision_manual * recall_manual / (
            precision_manual + recall_manual
        )
        print(
            f"Manual {target_names[selected_class]} metrics:",
            precision_manual,
            recall_manual,
            f1_manual,
        )

    kmeans_rows = []
    fitted_kmeans = {}
    for seed in SEEDS:
        split = splits[seed]
        for representation in REPRESENTATIONS:
            if representation == "unscaled":
                X_train_rep, X_val_rep = split["X_train"], split["X_val"]
            else:
                X_train_rep = preprocessing[seed]["X_train_scaled"]
                X_val_rep = preprocessing[seed]["X_val_scaled"]
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            model.fit(X_train_rep)
            val_clusters = model.predict(X_val_rep)
            val_silhouette = silhouette_score(X_val_rep, val_clusters)
            kmeans_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "representation": representation,
                    "n_clusters": n_clusters,
                    "train_inertia": model.inertia_,
                    "val_ari": adjusted_rand_score(split["y_val"], val_clusters),
                    "val_silhouette": val_silhouette,
                }
            )
            fitted_kmeans[(seed, representation)] = model
    kmeans_runs = pd.DataFrame(kmeans_rows)
    kmeans_summary = kmeans_runs.groupby(
        ["dataset", "representation"], as_index=False
    ).agg(
        train_inertia_mean=("train_inertia", "mean"),
        train_inertia_std=("train_inertia", "std"),
        val_ari_mean=("val_ari", "mean"),
        val_ari_std=("val_ari", "std"),
        val_silhouette_mean=("val_silhouette", "mean"),
        val_silhouette_std=("val_silhouette", "std"),
        valid_ari_count=("val_ari", "count"),
        valid_silhouette_count=("val_silhouette", "count"),
    )
    kmeans_test_rows = []
    for representation in REPRESENTATIONS:
        model = fitted_kmeans[(42, representation)]
        if representation == "unscaled":
            test_rep = X_test
        else:
            test_rep = preprocessing[42]["X_test_scaled"]
        test_clusters = model.predict(test_rep)
        test_silhouette = silhouette_score(test_rep, test_clusters)
        kmeans_test_rows.append(
            {
                "dataset": dataset_name,
                "representation": representation,
                "seed": 42,
                "n_clusters": n_clusters,
                "test_cluster_counts": dict(pd.Series(test_clusters).value_counts().sort_index()),
                "test_ari": adjusted_rand_score(y_test, test_clusters),
                "test_silhouette": test_silhouette,
            }
        )
        contingency = pd.crosstab(
            pd.Series(y_test, name="true_class"),
            pd.Series(test_clusters, index=y_test.index, name="cluster"),
        )
        print(f"\n{representation} test class/cluster contingency table:")
        print(contingency)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for class_value in sorted(y_test.unique()):
            points = X_test.loc[y_test == class_value, plot_features]
            axes[0].scatter(points.iloc[:, 0], points.iloc[:, 1], label=str(target_names[class_value]))
        for cluster_value in sorted(np.unique(test_clusters)):
            points = X_test.loc[test_clusters == cluster_value, plot_features]
            axes[1].scatter(points.iloc[:, 0], points.iloc[:, 1], label=f"cluster {cluster_value}")
        axes[0].set_title("True classes")
        axes[1].set_title("Predicted clusters")
        for axis in axes:
            axis.set_xlabel(plot_features[0])
            axis.set_ylabel(plot_features[1])
            axis.legend()
        fig.suptitle(f"{dataset_name}: {representation} K-Means test projection")
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"{dataset_name}_kmeans_{representation}.png", dpi=150
        )
        plt.show()
    kmeans_test = pd.DataFrame(kmeans_test_rows)

    _save_combined("classifier_runs.csv", classifier_runs)
    _save_combined("classifier_summary.csv", classifier_summary)
    _save_combined("classifier_test.csv", classifier_test)
    _save_combined("kmeans_runs.csv", kmeans_runs)
    _save_combined("kmeans_summary.csv", kmeans_summary)
    _save_combined("kmeans_test.csv", kmeans_test)

    print(f"\nSaved CSV outputs for {dataset_name}.")
    return {
        "split_summary": split_table,
        "classifier_runs": classifier_runs,
        "classifier_summary": classifier_summary,
        "classifier_test": classifier_test,
        "kmeans_runs": kmeans_runs,
        "kmeans_summary": kmeans_summary,
        "kmeans_test": kmeans_test,
    }
