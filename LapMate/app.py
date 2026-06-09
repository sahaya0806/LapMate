"""Flask API for laptop recommendations.

Supports two recommendation modes via the `mode` field in the request body
(or `?mode=` query param for GET-style testing):

  mode=rule  (default) – fast rule-based scoring (LaptopRecommender)
  mode=ml             – ML pipeline: KNN + cosine similarity + TF-IDF (MLLaptopRecommender)
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from recommendation_engine import LaptopRecommender, MLLaptopRecommender, DenStreamRecommender

app = Flask(__name__)
CORS(app)

# Instantiate all three engines once at startup so they are ready to serve requests.
rule_recommender       = LaptopRecommender()
ml_recommender         = MLLaptopRecommender()
denstream_recommender  = DenStreamRecommender()


@app.route("/")
def index():
    """Serve the frontend UI."""
    return render_template("index.html")


@app.route("/clusters")
def clusters_viz():
    """Serve the cluster visualization page."""
    return render_template("clusters.html")


@app.route("/api/brands", methods=["GET"])
def get_brands():
    """Return the list of available laptop brands (same for both engines)."""
    return jsonify(rule_recommender.get_brands())


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Return laptop recommendations.

    Request body (JSON):
        user_type   – e.g. "Student", "Employee", "Gamer"          (default: "Normal User")
        domain      – e.g. "AIML", "Gaming", "Design"              (default: "general")
        brand       – optional brand filter, e.g. "Dell"
        max_results – number of results to return                   (default: 12)
        mode        – "rule" for rule-based, "ml" for ML pipeline   (default: "rule")

    The `mode` can also be passed as a query-string parameter for quick testing:
        POST /api/recommend?mode=ml
    """
    data = request.get_json() or {}

    user_type   = data.get("user_type", "Normal User")
    domain      = data.get("domain")
    brand       = data.get("brand")
    max_results = int(data.get("max_results", 12))

    # Accept mode from JSON body first, fall back to query string, then default to "rule"
    mode = str(data.get("mode") or request.args.get("mode") or "rule").strip().lower()

    if mode == "ml":
        engine = ml_recommender
    elif mode == "denstream":
        engine = denstream_recommender
    else:
        engine = rule_recommender

    results = engine.get_recommendations(
        user_type=user_type,
        domain=domain,
        brand=brand,
        max_results=max_results,
    )

    # Tag each result with the engine that produced it so the frontend can show it
    for r in results:
        r["recommendation_mode"] = mode

    return jsonify({"mode": mode, "results": results})


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    """
    Return info about the DenStream micro-clusters formed from the laptop catalogue.
    Includes centroid values (inverse-transformed to human-readable units),
    and the count of laptops nearest to each centroid (for bubble sizing).
    """
    import numpy as np

    centroids  = denstream_recommender._centroids          # (n_clusters, 6) scaled
    scaler     = denstream_recommender.scaler
    df         = denstream_recommender.df
    feat_cols  = ["ram_gb", "ssd_gb", "gpu_vram_gb", "processor_tier", "gpu_tier", "price_norm"]

    # Assign every laptop to its nearest centroid
    laptop_vecs = df[feat_cols].fillna(0).values.astype(float)
    laptop_scaled = scaler.transform(laptop_vecs)           # (n_laptops, 6)
    # Euclidean distance from each laptop to each centroid
    diffs = laptop_scaled[:, None, :] - centroids[None, :, :]   # (n_laptops, n_clusters, 6)
    dist  = np.linalg.norm(diffs, axis=2)                        # (n_laptops, n_clusters)
    assignments = dist.argmin(axis=1)                            # (n_laptops,)

    # Count laptops per cluster
    from collections import Counter
    counts = Counter(assignments.tolist())

    cluster_list = []
    max_price = float(df["Price"].max()) or 1.0
    for i, c in enumerate(centroids):
        raw = scaler.inverse_transform(c.reshape(1, -1))[0]
        cluster_list.append({
            "cluster_id":     i,
            "ram_gb":         round(float(raw[0]), 1),
            "ssd_gb":         round(float(raw[1]), 1),
            "gpu_vram_gb":    round(float(raw[2]), 1),
            "processor_tier": round(float(raw[3]), 2),
            "gpu_tier":       round(float(raw[4]), 2),
            "price_approx":   round(float(raw[5]) * max_price, -2),
            "laptop_count":   counts.get(i, 0),
        })

    # Also send the raw scaled centroids so the frontend can do PCA/2-D projection
    centroids_scaled = centroids.tolist()

    return jsonify({
        "total_clusters":    len(cluster_list),
        "clusters":          cluster_list,
        "centroids_scaled":  centroids_scaled,
        "feature_names":     feat_cols,
    })


@app.route("/api/recommend/compare", methods=["POST"])
def recommend_compare():
    """
    Run both engines with the same parameters and return results side-by-side.
    Useful for A/B comparison during development.

    Request body: same fields as /api/recommend (mode is ignored here).
    """
    data = request.get_json() or {}

    user_type   = data.get("user_type", "Normal User")
    domain      = data.get("domain")
    brand       = data.get("brand")
    max_results = int(data.get("max_results", 12))

    rule_results = rule_recommender.get_recommendations(
        user_type=user_type, domain=domain, brand=brand, max_results=max_results
    )
    ml_results = ml_recommender.get_recommendations(
        user_type=user_type, domain=domain, brand=brand, max_results=max_results
    )
    denstream_results = denstream_recommender.get_recommendations(
        user_type=user_type, domain=domain, brand=brand, max_results=max_results
    )

    return jsonify({
        "rule":       rule_results,
        "ml":         ml_results,
        "denstream":  denstream_results,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")