"""
clustering.py

Standardizes the per-90 feature vectors and fits K-Means (baseline) and
a Gaussian Mixture Model (soft-assignment comparison), sweeping k over
config.K_RANGE to pick a cluster count via the elbow method (K-Means
training cost / within-cluster-sum-of-squares) and silhouette score.
"""
from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame

from features import FEATURE_COLUMNS


def assemble_and_scale(df: DataFrame):
    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features_raw")
    assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)
    return scaled, scaler_model


def sweep_kmeans_k(scaled_df: DataFrame, k_range, seed: int):
    """Returns a list of dicts: k, training_cost (elbow), silhouette."""
    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster", metricName="silhouette"
    )
    results = []
    for k in k_range:
        km = KMeans(featuresCol="features", predictionCol="cluster", k=k, seed=seed)
        model = km.fit(scaled_df)
        preds = model.transform(scaled_df)
        silhouette = evaluator.evaluate(preds)
        results.append({
            "k": k,
            "training_cost": model.summary.trainingCost,
            "silhouette": silhouette,
        })
        print(f"  k={k:2d}  training_cost={model.summary.trainingCost:10.2f}  silhouette={silhouette:.4f}")
    return results


def fit_kmeans(scaled_df: DataFrame, k: int, seed: int):
    km = KMeans(featuresCol="features", predictionCol="cluster_kmeans", k=k, seed=seed)
    model = km.fit(scaled_df)
    return model, model.transform(scaled_df)


def fit_gmm(scaled_df: DataFrame, k: int, seed: int):
    gmm = GaussianMixture(featuresCol="features", predictionCol="cluster_gmm", k=k, seed=seed)
    model = gmm.fit(scaled_df)
    return model, model.transform(scaled_df)


def silhouette_score(preds: DataFrame, prediction_col: str) -> float:
    """Silhouette for an already-clustered DataFrame, for comparing the
    final KMeans vs GMM fits on the same features."""
    return ClusteringEvaluator(
        featuresCol="features", predictionCol=prediction_col, metricName="silhouette"
    ).evaluate(preds)


def choose_k(sweep_results):
    """Pick k with the highest silhouette score as a simple default;
    inspect the printed elbow/silhouette table and override manually if
    the elbow plot suggests a more interpretable k."""
    return max(sweep_results, key=lambda r: r["silhouette"])["k"]
