"""Flask API for laptop recommendations."""
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from recommendation_engine import LaptopRecommender

app = Flask(__name__)
CORS(app)
recommender = LaptopRecommender()

@app.route("/")
def index():
    """Serve the main Laptop Finder page."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.route("/api/brands", methods=["GET"])
def get_brands():
    return jsonify(recommender.get_brands())

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    user_type = data.get("user_type", "Normal User")
    domain = data.get("domain")
    brand = data.get("brand")
    min_price = data.get("min_price")
    max_price = data.get("max_price")
    max_results = int(data.get("max_results", 24))
    if min_price is not None:
        min_price = int(min_price)
    if max_price is not None:
        max_price = int(max_price)

    results = recommender.get_recommendations(
        user_type=user_type,
        domain=domain,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        max_results=max_results,
    )
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")