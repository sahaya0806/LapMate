"""Laptop recommendation engine based on user type and domain."""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from river import cluster as river_cluster

DOMAIN_SPECS = {
    "student_aiml": (16, 512, 4, 3, 2),
    "student_gaming": (16, 512, 4, 2, 1),
    "student_arts_science": (8, 256, 0, 2, 0),
    "student_engineering": (16, 512, 2, 3, 1),
    "student_design": (16, 512, 2, 3, 1),
    "student_general": (8, 256, 0, 1, 0),
    "employee_software": (16, 512, 0, 3, 0),
    "employee_data_science": (16, 512, 4, 3, 2),
    "employee_design": (16, 512, 2, 3, 1),
    "employee_business": (8, 256, 0, 2, 0),
    "employee_general": (8, 256, 0, 2, 0),
    "gamer": (16, 512, 6, 3, 2),
    "video_editing": (16, 512, 4, 4, 2),
    "staff": (8, 256, 0, 1, 0),
    "normal_user": (8, 256, 0, 2, 0),
}

def extract_gb(value):
    if pd.isna(value) or value in ("", "NO SSD", "No HDD"):
        return 0
    val = str(value).upper()
    match = re.search(r"(\d+)\s*GB", val, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def classify_processor_tier(proc_name):
    if pd.isna(proc_name):
        return 1
    p = str(proc_name).upper()
    if "CELERON" in p or "PENTIUM" in p or "RYZEN 3" in p or "DUAL CORE" in p or "MEDIATEK" in p or "QUAD CORE" in p:
        return 1
    if "RYZEN 5" in p or "CORE I3" in p or "HEXA CORE" in p:
        return 2
    if "RYZEN 7" in p or "CORE I5" in p or "OCTA CORE" in p or "CORE ULTRA 5" in p:
        return 3
    if "RYZEN 9" in p or "CORE I7" in p or "CORE I9" in p or "CORE ULTRA 7" in p or "CORE ULTRA 9" in p or "M1" in p or "M2" in p or "M3" in p:
        return 4
    return 2

def classify_gpu_tier(gpu):
    if pd.isna(gpu):
        return 0
    g = str(gpu).upper()
    if "INTEGRATED" in g or "IRIS XE" in g or "IRIS" in g or "UHD" in g:
        return 0
    if "RADEON" in g and "RX" not in g and "RTX" not in g and "GTX" not in g:
        return 0
    if "GTX 1650" in g or "RTX 2050" in g or "RTX 3050" in g or "GTX" in g:
        return 1
    if "RTX 4050" in g or "RTX 3060" in g or "RTX 4060" in g or "RADEON RX" in g:
        return 2
    if "RTX 4070" in g or "RTX 4080" in g or "RTX 4090" in g or "RTX 3070" in g or "RTX 3080" in g:
        return 3
    if "RTX" in g or "GEFORCE" in g:
        return 1
    return 0

class LaptopRecommender:
    def __init__(self, csv_path=None):
        base = Path(__file__).parent
        self.csv_path = csv_path or str(base / "laptop.csv")
        self.df = None
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        self.df = pd.read_csv(self.csv_path)
        if len(self.df.columns) > 0 and (self.df.columns[0].startswith('Unnamed') or self.df.columns[0] == ''):
            self.df = self.df.iloc[:, 1:]
        self.df = self.df.dropna(subset=["Brand", "Name", "Price"])
        self.df["Price"] = pd.to_numeric(self.df["Price"], errors="coerce").fillna(0).astype(int)
        self.df = self.df[self.df["Price"] > 0]

        self.df["ram_gb"] = self.df["RAM"].apply(extract_gb)
        self.df["ssd_gb"] = self.df["SSD"].apply(
            lambda x: 0 if pd.isna(x) or "NO SSD" in str(x) else extract_gb(str(x))
        )
        self.df["gpu_vram_gb"] = self.df["GPU"].apply(
            lambda x: extract_gb(str(x)) if pd.notna(x) and "GB" in str(x) and "INTEGRATED" not in str(x).upper() and "UHD" not in str(x).upper() and "IRIS" not in str(x).upper() else (extract_gb(str(x)) if "RADEON RX" in str(x).upper() else 0)
        )
        self.df["processor_tier"] = self.df["Processor_Name"].apply(classify_processor_tier)
        self.df["gpu_tier"] = self.df["GPU"].apply(classify_gpu_tier)
        self.df["clean_name"] = self.df["Name"].apply(lambda x: str(x).split("::")[0].strip() if pd.notna(x) else "")

    def score_laptop(self, row, min_ram, min_ssd, min_gpu_vram, min_proc_tier, min_gpu_tier):
        score = 0.0
        ram, ssd, gpu_vram = row.get("ram_gb", 0), row.get("ssd_gb", 0), row.get("gpu_vram_gb", 0)
        proc_tier, gpu_tier = row.get("processor_tier", 1), row.get("gpu_tier", 0)

        if ram >= min_ram: score += 25
        elif ram >= min_ram // 2: score += 12
        if ssd >= min_ssd: score += 25
        elif ssd >= min_ssd // 2: score += 12
        if min_gpu_vram == 0 and min_gpu_tier == 0:
            score += 25
        elif gpu_vram >= min_gpu_vram or gpu_tier >= min_gpu_tier:
            score += 25
        elif gpu_tier >= max(0, min_gpu_tier - 1):
            score += 15
        if proc_tier >= min_proc_tier: score += 25
        elif proc_tier >= min_proc_tier - 1: score += 15

        price = row.get("Price", 0)
        if price > 0:
            score -= np.log1p(price) * 0.3
        return max(0, score)

    def get_recommendations(self, user_type, domain=None, brand=None, max_results=12):
        user_type = str(user_type).lower().replace(" ", "_")
        domain = (domain or "general").lower().replace(" ", "_").replace("&", "_").replace("and", "_")
        key = f"{user_type}_{domain}"
        if key not in DOMAIN_SPECS:
            key = f"{user_type}_general" if f"{user_type}_general" in DOMAIN_SPECS else "normal_user"
        min_ram, min_ssd, min_gpu_vram, min_proc_tier, min_gpu_tier = DOMAIN_SPECS.get(key, DOMAIN_SPECS["normal_user"])

        df = self.df.copy()
        if brand:
            df = df[df["Brand"].str.lower() == brand.lower()]

        df["score"] = df.apply(
            lambda r: self.score_laptop(r, min_ram, min_ssd, min_gpu_vram, min_proc_tier, min_gpu_tier), axis=1
        )
        df = df.sort_values("score", ascending=False).head(max_results)

        results = []
        for idx, row in df.iterrows():
            results.append({
                "Brand": row["Brand"],
                "Name": row["clean_name"],
                "Price": int(row["Price"]),
                "Processor_Name": str(row["Processor_Name"]) if pd.notna(row["Processor_Name"]) else "",
                "RAM": str(row["RAM"]) if pd.notna(row["RAM"]) else "",
                "SSD": str(row["SSD"]) if pd.notna(row["SSD"]) else "",
                "GPU": str(row["GPU"]) if pd.notna(row["GPU"]) else "",
                "Display": str(row["Display"]) if pd.notna(row["Display"]) else "",
                "Battery_Life": str(row["Battery_Life"]) if pd.notna(row["Battery_Life"]) else "",
                "image_url": f"https://picsum.photos/seed/{hash(str(row['Brand'])+str(idx)) % 100000}/400/300",
            })
        return results

    def get_brands(self):
        return sorted(self.df["Brand"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# ML-based recommender: Content-Based Filtering (cosine similarity) + KNN
# ---------------------------------------------------------------------------

# Weights applied to each numeric feature when building the feature matrix.
# Higher weight = that feature matters more when computing similarity.
FEATURE_WEIGHTS = {
    "ram_gb":         3.0,   # RAM is a primary differentiator
    "ssd_gb":         2.0,   # Storage matters but less than RAM
    "gpu_vram_gb":    2.5,   # GPU VRAM is critical for gaming/ML workloads
    "processor_tier": 3.0,   # CPU tier is a strong quality signal
    "gpu_tier":       2.5,   # GPU tier complements VRAM
    "price_norm":     1.5,   # Price included but down-weighted
}

# Maps (user_type, domain) → ideal spec vector matching FEATURE_WEIGHTS order
# (ram_gb, ssd_gb, gpu_vram_gb, processor_tier, gpu_tier)
# price_norm is derived from a budget hint per profile
PROFILE_SPECS = {
    "student_aiml":          (16, 512, 4, 3, 2),
    "student_gaming":        (16, 512, 4, 2, 1),
    "student_arts_science":  (8,  256, 0, 2, 0),
    "student_engineering":   (16, 512, 2, 3, 1),
    "student_design":        (16, 512, 2, 3, 1),
    "student_general":       (8,  256, 0, 1, 0),
    "employee_software":     (16, 512, 0, 3, 0),
    "employee_data_science": (16, 512, 4, 3, 2),
    "employee_design":       (16, 512, 2, 3, 1),
    "employee_business":     (8,  256, 0, 2, 0),
    "employee_general":      (8,  256, 0, 2, 0),
    "gamer":                 (16, 512, 6, 3, 2),
    "video_editing":         (16, 512, 4, 4, 2),
    "staff":                 (8,  256, 0, 1, 0),
    "normal_user":           (8,  256, 0, 2, 0),
}

# Rough mid-range price hint per profile (used to normalise price in query vector)
PROFILE_BUDGET_HINT = {
    "student_aiml":          80000,
    "student_gaming":        70000,
    "student_arts_science":  40000,
    "student_engineering":   60000,
    "student_design":        70000,
    "student_general":       35000,
    "employee_software":     70000,
    "employee_data_science": 90000,
    "employee_design":       80000,
    "employee_business":     50000,
    "employee_general":      45000,
    "gamer":                 100000,
    "video_editing":         90000,
    "staff":                 35000,
    "normal_user":           40000,
}


class MLLaptopRecommender:
    """
    Two-stage ML recommender:

    Stage 1 – Content-Based Filtering
        Build a combined feature matrix from:
        - Weighted numeric specs (RAM, SSD, GPU VRAM, processor tier, GPU tier, price)
        - TF-IDF vectors of laptop text (processor name + GPU name + brand)
        Compute cosine similarity between a user-profile query vector and all laptops.

    Stage 2 – KNN refinement
        Fit a NearestNeighbors model on the same feature matrix.
        Use it to retrieve the top-K nearest neighbours to the query vector,
        then re-rank by cosine similarity score.

    The two stages complement each other:
    - Cosine similarity gives a global relevance score across the whole catalogue.
    - KNN efficiently narrows the search space and handles edge cases where
      cosine similarity alone might surface outliers.
    """

    def __init__(self, csv_path=None):
        base = Path(__file__).parent
        self.csv_path = csv_path or str(base / "laptop.csv")
        self.df = None
        self.feature_matrix = None   # combined sparse matrix (numeric + tfidf)
        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._numeric_cols = list(FEATURE_WEIGHTS.keys())
        self._weights = np.array(list(FEATURE_WEIGHTS.values()), dtype=float)
        self._load_and_build()

    # ------------------------------------------------------------------
    # Data loading & feature engineering
    # ------------------------------------------------------------------

    def _load_and_build(self):
        """Load CSV, preprocess, build feature matrix, fit KNN."""
        df = pd.read_csv(self.csv_path)
        # Drop unnamed index column if present
        if len(df.columns) > 0 and (df.columns[0].startswith("Unnamed") or df.columns[0] == ""):
            df = df.iloc[:, 1:]

        df = df.dropna(subset=["Brand", "Name", "Price"])
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
        df = df[df["Price"] > 0].reset_index(drop=True)

        # Numeric feature extraction (reuse module-level helpers)
        df["ram_gb"]        = df["RAM"].apply(extract_gb)
        df["ssd_gb"]        = df["SSD"].apply(
            lambda x: 0 if pd.isna(x) or "NO SSD" in str(x) else extract_gb(str(x))
        )
        df["gpu_vram_gb"]   = df["GPU"].apply(self._extract_gpu_vram)
        df["processor_tier"] = df["Processor_Name"].apply(classify_processor_tier)
        df["gpu_tier"]      = df["GPU"].apply(classify_gpu_tier)
        df["clean_name"]    = df["Name"].apply(
            lambda x: str(x).split("::")[0].strip() if pd.notna(x) else ""
        )

        # Normalise price to [0, 1] across the catalogue
        max_price = df["Price"].max() or 1
        df["price_norm"] = df["Price"] / max_price

        self.df = df

        # Build feature matrix
        self.feature_matrix = self._build_feature_matrix(df)

        # Fit KNN on the full feature matrix
        self.knn.fit(self.feature_matrix)

    @staticmethod
    def _extract_gpu_vram(gpu):
        """Extract discrete GPU VRAM; returns 0 for integrated graphics."""
        if pd.isna(gpu):
            return 0
        g = str(gpu).upper()
        if any(k in g for k in ("INTEGRATED", "IRIS XE", "IRIS", "UHD")):
            return 0
        if "RADEON RX" in g or "GEFORCE" in g or "RTX" in g or "GTX" in g:
            return extract_gb(gpu)
        return 0

    def _build_feature_matrix(self, df):
        """
        Combine weighted numeric features with TF-IDF text features.
        Returns a sparse matrix of shape (n_laptops, n_features).
        """
        # --- Numeric block ---
        numeric = df[self._numeric_cols].fillna(0).values.astype(float)
        # Fit scaler on catalogue data (called once during init)
        numeric_scaled = self.scaler.fit_transform(numeric)
        # Apply per-feature weights
        numeric_weighted = numeric_scaled * self._weights  # broadcast
        numeric_sparse = csr_matrix(numeric_weighted)

        # --- Text block (TF-IDF) ---
        text_corpus = (
            df["Processor_Name"].fillna("") + " "
            + df["GPU"].fillna("") + " "
            + df["Brand"].fillna("")
        )
        tfidf_matrix = self.tfidf.fit_transform(text_corpus)

        # Concatenate horizontally
        return hstack([numeric_sparse, tfidf_matrix])

    # ------------------------------------------------------------------
    # Query vector construction
    # ------------------------------------------------------------------

    def _build_query_vector(self, profile_key: str, brand: str | None = None):
        """
        Build a single-row feature vector for the user's profile.
        Mirrors the structure of _build_feature_matrix.
        """
        ram, ssd, gpu_vram, proc_tier, gpu_tier = PROFILE_SPECS.get(
            profile_key, PROFILE_SPECS["normal_user"]
        )
        budget = PROFILE_BUDGET_HINT.get(profile_key, 50000)
        max_price = self.df["Price"].max() or 1
        price_norm = min(budget / max_price, 1.0)

        raw_numeric = np.array(
            [[ram, ssd, gpu_vram, proc_tier, gpu_tier, price_norm]], dtype=float
        )
        # Use the already-fitted scaler (transform only, not fit)
        numeric_scaled = self.scaler.transform(raw_numeric)
        numeric_weighted = numeric_scaled * self._weights
        numeric_sparse = csr_matrix(numeric_weighted)

        # Build a representative text query for the profile
        text_query = self._profile_text_query(profile_key, brand)
        tfidf_query = self.tfidf.transform([text_query])

        return hstack([numeric_sparse, tfidf_query])

    @staticmethod
    def _profile_text_query(profile_key: str, brand: str | None) -> str:
        """Generate a descriptive text string that matches likely laptop descriptions."""
        text_map = {
            "student_aiml":          "ryzen 7 core i7 rtx geforce nvidia gpu",
            "student_gaming":        "ryzen 5 core i5 gtx rtx geforce gaming",
            "student_arts_science":  "core i3 ryzen 5 integrated lightweight",
            "student_engineering":   "core i5 ryzen 5 iris xe engineering",
            "student_design":        "core i5 ryzen 7 rtx design display",
            "student_general":       "core i3 celeron lightweight budget",
            "employee_software":     "core i5 ryzen 7 iris xe developer",
            "employee_data_science": "ryzen 7 core i7 rtx nvidia data science",
            "employee_design":       "core i5 ryzen 7 rtx design creative",
            "employee_business":     "core i3 core i5 business lightweight",
            "employee_general":      "core i3 core i5 office productivity",
            "gamer":                 "ryzen 9 core i9 rtx 4070 4080 gaming nvidia",
            "video_editing":         "ryzen 9 core i9 rtx 4070 video editing nvidia",
            "staff":                 "celeron core i3 budget office",
            "normal_user":           "core i3 ryzen 5 everyday use",
        }
        base = text_map.get(profile_key, "core i5 ryzen 5")
        if brand:
            base = f"{brand} {base}"
        return base

    # ------------------------------------------------------------------
    # Recommendation pipeline
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        user_type: str,
        domain: str | None = None,
        brand: str | None = None,
        max_results: int = 12,
    ) -> list[dict]:
        """
        Return ranked laptop recommendations using cosine similarity + KNN.

        Pipeline:
        1. Resolve profile key from user_type + domain.
        2. Build query vector.
        3. KNN: retrieve top-K candidates (2× max_results for headroom).
        4. Cosine similarity: score each candidate against the query.
        5. Optional brand filter.
        6. Return top max_results sorted by similarity score.
        """
        profile_key = self._resolve_profile(user_type, domain)
        query_vec = self._build_query_vector(profile_key, brand)

        # --- Stage 1: KNN candidate retrieval ---
        # Fetch more candidates than needed so brand filtering doesn't starve results
        k = min(len(self.df), max(max_results * 4, 50))
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=k)
        candidate_indices = indices[0]

        # --- Stage 2: Cosine similarity re-ranking ---
        candidate_matrix = self.feature_matrix[candidate_indices]
        sim_scores = cosine_similarity(query_vec, candidate_matrix)[0]

        # Build result dataframe for easy filtering & sorting
        candidates = self.df.iloc[candidate_indices].copy()
        candidates["ml_score"] = sim_scores

        # Brand filter (applied after retrieval to preserve diversity)
        if brand:
            filtered = candidates[candidates["Brand"].str.lower() == brand.lower()]
            # Fall back to unfiltered if brand yields too few results
            candidates = filtered if len(filtered) >= max_results // 2 else candidates

        candidates = candidates.sort_values("ml_score", ascending=False).head(max_results)

        return self._format_results(candidates)

    def get_brands(self) -> list[str]:
        return sorted(self.df["Brand"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_profile(user_type: str, domain: str | None) -> str:
        user_type = str(user_type).lower().replace(" ", "_")
        domain = (domain or "general").lower().replace(" ", "_").replace("&", "_").replace("and", "_")
        key = f"{user_type}_{domain}"
        if key in PROFILE_SPECS:
            return key
        fallback = f"{user_type}_general"
        return fallback if fallback in PROFILE_SPECS else "normal_user"

    @staticmethod
    def _format_results(df: pd.DataFrame) -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            results.append({
                "Brand":          row["Brand"],
                "Name":           row["clean_name"],
                "Price":          int(row["Price"]),
                "Processor_Name": str(row["Processor_Name"]) if pd.notna(row.get("Processor_Name")) else "",
                "RAM":            str(row["RAM"])            if pd.notna(row.get("RAM"))            else "",
                "SSD":            str(row["SSD"])            if pd.notna(row.get("SSD"))            else "",
                "GPU":            str(row["GPU"])            if pd.notna(row.get("GPU"))            else "",
                "Display":        str(row["Display"])        if pd.notna(row.get("Display"))        else "",
                "Battery_Life":   str(row["Battery_Life"])   if pd.notna(row.get("Battery_Life"))   else "",
                "ml_score":       round(float(row["ml_score"]), 4),
                "image_url":      f"https://picsum.photos/seed/{hash(str(row['Brand'])+str(idx)) % 100000}/400/300",
            })
        return results


# ---------------------------------------------------------------------------
# DenStream-enhanced recommender
# ---------------------------------------------------------------------------
# How this works end-to-end:
#
#  STARTUP (once):
#   1. Load & preprocess the laptop CSV → extract 6 numeric features per laptop
#   2. Normalise features with MinMaxScaler (same scaler as MLLaptopRecommender)
#   3. Stream all laptop feature vectors through DenStream one by one
#      → DenStream builds micro-clusters (core + potential + outlier)
#      → Each micro-cluster has a centroid = weighted average of member points
#   4. Collect all core/potential micro-cluster centroids
#   5. Build the same 306-dim feature matrix (numeric + TF-IDF) and fit KNN
#
#  PER REQUEST:
#   1. Resolve user profile key  (e.g. "student_aiml")
#   2. Look up PROFILE_SPECS to get the ideal spec vector (hardcoded baseline)
#   3. Find the DenStream centroid NEAREST to that baseline in numeric space
#      → This centroid is the "enhanced query" — it reflects what laptops in
#        that segment actually look like rather than a hardcoded spec table
#   4. Build full 306-dim query vector from the centroid values + TF-IDF text
#   5. KNN: retrieve top-K candidate laptops (same as MLLaptopRecommender)
#   6. Cosine Similarity: re-rank candidates against the centroid query vector
#   7. Return top N results with ml_score + which cluster they came from
# ---------------------------------------------------------------------------

# DenStream hyper-parameters
# decaying_factor (λ): how fast old data fades. 0.15 = moderate forgetting
# beta: threshold weight for a micro-cluster to be "potential" vs "outlier"
# mu:   minimum weight for a micro-cluster to graduate to "core"
# epsilon: neighbourhood radius — controls how tightly packed a cluster must be
_DENSTREAM_PARAMS = dict(
    decaying_factor=0.15,
    beta=0.5,
    mu=3,
    epsilon=0.5,
)

# Feature columns fed into DenStream (raw numeric, before scaling)
_DS_FEATURE_COLS = ["ram_gb", "ssd_gb", "gpu_vram_gb", "processor_tier", "gpu_tier", "price_norm"]


class DenStreamRecommender:
    """
    Three-stage recommendation pipeline:

    Stage 0 – DenStream clustering (replaces hardcoded PROFILE_SPECS lookup)
        Stream all laptop feature vectors through DenStream at startup.
        On each request, find the cluster centroid nearest to the user's
        profile baseline. That centroid becomes the enhanced query vector.

    Stage 1 – KNN candidate retrieval
        Identical to MLLaptopRecommender: fetch top-K nearest laptops in the
        306-dim (weighted numeric + TF-IDF) feature space.

    Stage 2 – Cosine Similarity re-ranking
        Identical to MLLaptopRecommender: score and rank the K candidates.
    """

    def __init__(self, csv_path=None):
        base = Path(__file__).parent
        self.csv_path = csv_path or str(base / "laptop.csv")

        # Shared with MLLaptopRecommender logic
        self.df = None
        self.feature_matrix = None
        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._numeric_cols = list(FEATURE_WEIGHTS.keys())   # 6 cols incl. price_norm
        self._weights = np.array(list(FEATURE_WEIGHTS.values()), dtype=float)

        # DenStream state
        self._denstream = None          # river DenStream instance
        self._centroids = None          # np.ndarray shape (n_clusters, 6)
        self._cluster_labels = None     # list of string labels per centroid

        self._load_and_build()

    # ------------------------------------------------------------------
    # 1. Data loading & feature engineering (identical to MLLaptopRecommender)
    # ------------------------------------------------------------------

    def _load_and_build(self):
        df = pd.read_csv(self.csv_path)
        if len(df.columns) > 0 and (df.columns[0].startswith("Unnamed") or df.columns[0] == ""):
            df = df.iloc[:, 1:]
        df = df.dropna(subset=["Brand", "Name", "Price"])
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
        df = df[df["Price"] > 0].reset_index(drop=True)

        df["ram_gb"]         = df["RAM"].apply(extract_gb)
        df["ssd_gb"]         = df["SSD"].apply(
            lambda x: 0 if pd.isna(x) or "NO SSD" in str(x) else extract_gb(str(x))
        )
        df["gpu_vram_gb"]    = df["GPU"].apply(self._extract_gpu_vram)
        df["processor_tier"] = df["Processor_Name"].apply(classify_processor_tier)
        df["gpu_tier"]       = df["GPU"].apply(classify_gpu_tier)
        df["clean_name"]     = df["Name"].apply(
            lambda x: str(x).split("::")[0].strip() if pd.notna(x) else ""
        )
        max_price = df["Price"].max() or 1
        df["price_norm"] = df["Price"] / max_price

        self.df = df

        # Run DenStream FIRST — it fits the scaler as part of streaming
        self._run_denstream(df)

        # Build the full 306-dim feature matrix and fit KNN
        # (scaler is already fitted by _run_denstream, so we use transform)
        self.feature_matrix = self._build_feature_matrix(df)
        self.knn.fit(self.feature_matrix)

    # ------------------------------------------------------------------
    # 2. DenStream — stream all laptops and extract cluster centroids
    # ------------------------------------------------------------------

    def _run_denstream(self, df: pd.DataFrame):
        """
        Stream the normalised 6-dim numeric vectors through DenStream
        and collect the resulting micro-cluster centroids.

        river's DenStream works purely online: call .learn_one(x_dict)
        for each sample where x_dict is {feature_name: value}.

        After all samples are streamed:
        - p_micro_clusters → potential micro-clusters (building up density)
        - o_micro_clusters → outlier micro-clusters (sparse, possible noise)
        We collect centroids from potential clusters (enough weight to matter).
        Each micro-cluster's centroid is retrieved via .calc_center(timestamp).
        """
        # Normalise the 6 numeric features to [0,1] using the fitted scaler
        raw = df[_DS_FEATURE_COLS].fillna(0).values.astype(float)
        scaled = self.scaler.fit_transform(raw)   # fits scaler here (once)

        # DenStream needs n_samples_init samples before it initialises clusters
        # (default 1000 in river). With ~4000 laptops we are well above that.
        self._denstream = river_cluster.DenStream(**_DENSTREAM_PARAMS)

        col_names = _DS_FEATURE_COLS
        for i, row_vals in enumerate(scaled):
            x = {col_names[j]: float(row_vals[j]) for j in range(len(col_names))}
            self._denstream.learn_one(x)

        # ------------------------------------------------------------------
        # Extract centroids from potential micro-clusters.
        # river exposes .p_micro_clusters (potential) and .o_micro_clusters
        # (outliers). We use potential clusters as they represent real dense
        # regions in the laptop feature space.
        # calc_center(timestamp) returns a dict {feature: centroid_value}
        # ------------------------------------------------------------------
        ts = self._denstream.timestamp
        centroids = []

        for mc_id, mc in self._denstream.p_micro_clusters.items():
            center_dict = mc.calc_center(ts)  # dict {feat_name: scaled_value}
            c_vec = np.array([center_dict.get(f, 0.0) for f in col_names], dtype=float)
            centroids.append(c_vec)

        # Also include outlier micro-clusters as fallback material
        for mc_id, mc in self._denstream.o_micro_clusters.items():
            center_dict = mc.calc_center(ts)
            c_vec = np.array([center_dict.get(f, 0.0) for f in col_names], dtype=float)
            centroids.append(c_vec)

        if len(centroids) == 0:
            # Safety fallback — use global mean of scaled data
            centroids = [scaled.mean(axis=0)]

        self._centroids = np.array(centroids)   # shape (n_clusters, 6)
        print(f"[DenStream] {len(self._centroids)} micro-clusters formed "
              f"from {len(df)} laptops  "
              f"(p={len(self._denstream.p_micro_clusters)}, "
              f"o={len(self._denstream.o_micro_clusters)})")

    # ------------------------------------------------------------------
    # 3. Find nearest centroid to a profile baseline
    # ------------------------------------------------------------------

    def _nearest_centroid(self, profile_key: str) -> np.ndarray:
        """
        Given a profile key (e.g. "student_aiml"), build the baseline spec
        vector from PROFILE_SPECS, normalise it with the fitted scaler, then
        find the DenStream centroid with the smallest Euclidean distance to it.

        Returns the centroid as a 1-D numpy array of shape (6,) in SCALED space.
        """
        ram, ssd, gpu_vram, proc_tier, gpu_tier = PROFILE_SPECS.get(
            profile_key, PROFILE_SPECS["normal_user"]
        )
        budget    = PROFILE_BUDGET_HINT.get(profile_key, 50000)
        max_price = self.df["Price"].max() or 1
        price_norm = min(budget / max_price, 1.0)

        # Raw baseline vector (same 6 features, same order as _DS_FEATURE_COLS)
        baseline_raw = np.array(
            [[ram, ssd, gpu_vram, proc_tier, gpu_tier, price_norm]], dtype=float
        )

        # Scale using the already-fitted scaler (transform only)
        baseline_scaled = self.scaler.transform(baseline_raw)[0]  # shape (6,)

        # Euclidean distances from baseline to every centroid
        diffs = self._centroids - baseline_scaled          # (n_clusters, 6)
        distances = np.linalg.norm(diffs, axis=1)          # (n_clusters,)
        nearest_idx = int(np.argmin(distances))

        return self._centroids[nearest_idx]   # shape (6,) in scaled space

    # ------------------------------------------------------------------
    # 4. Build the 306-dim query vector from the centroid
    # ------------------------------------------------------------------

    def _build_query_vector_from_centroid(
        self, centroid_scaled: np.ndarray, profile_key: str, brand: str | None
    ):
        """
        Convert the 6-dim scaled centroid into the full 306-dim query vector:
          - Apply feature weights to the scaled centroid
          - Append TF-IDF text vector for the profile
        This is the enhanced query vector that replaces the hardcoded one.
        """
        # Apply weights (element-wise multiply)
        numeric_weighted = centroid_scaled * self._weights   # shape (6,)
        numeric_sparse   = csr_matrix(numeric_weighted.reshape(1, -1))

        # TF-IDF text part (same as MLLaptopRecommender)
        text_query = MLLaptopRecommender._profile_text_query(profile_key, brand)
        tfidf_query = self.tfidf.transform([text_query])

        return hstack([numeric_sparse, tfidf_query])

    # ------------------------------------------------------------------
    # 5. Main recommendation pipeline
    # ------------------------------------------------------------------

    def get_recommendations(
        self,
        user_type: str,
        domain: str | None = None,
        brand: str | None = None,
        max_results: int = 12,
    ) -> list[dict]:
        """
        DenStream → KNN → Cosine Similarity pipeline:

        1. Resolve profile key from user_type + domain
        2. Find the nearest DenStream centroid to the profile baseline
        3. Build 306-dim query vector from the centroid (enhanced query)
        4. KNN: retrieve top-K candidate laptops
        5. Cosine Similarity: re-rank candidates
        6. Optional brand filter
        7. Return top max_results with ml_score and cluster_info
        """
        # Step 1 – profile key
        profile_key = MLLaptopRecommender._resolve_profile(user_type, domain)

        # Step 2 – nearest DenStream centroid (in scaled 6-dim space)
        centroid = self._nearest_centroid(profile_key)

        # Step 3 – build full 306-dim query vector from centroid
        query_vec = self._build_query_vector_from_centroid(centroid, profile_key, brand)

        # Step 4 – KNN candidate retrieval
        k = min(len(self.df), max(max_results * 4, 50))
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=k)
        candidate_indices = indices[0]

        # Step 5 – Cosine Similarity re-ranking
        candidate_matrix = self.feature_matrix[candidate_indices]
        sim_scores = cosine_similarity(query_vec, candidate_matrix)[0]

        candidates = self.df.iloc[candidate_indices].copy()
        candidates["ml_score"] = sim_scores

        # Step 6 – optional brand filter
        if brand:
            filtered = candidates[candidates["Brand"].str.lower() == brand.lower()]
            candidates = filtered if len(filtered) >= max_results // 2 else candidates

        candidates = candidates.sort_values("ml_score", ascending=False).head(max_results)

        # Build cluster info string for transparency
        cluster_info = (
            f"RAM≈{centroid[0]*self.scaler.data_max_[0]:.0f}GB  "
            f"SSD≈{centroid[1]*self.scaler.data_max_[1]:.0f}GB  "
            f"GPU_VRAM≈{centroid[2]*self.scaler.data_max_[2]:.1f}GB  "
            f"CPU_tier≈{centroid[3]*self.scaler.data_max_[3]:.1f}  "
            f"GPU_tier≈{centroid[4]*self.scaler.data_max_[4]:.1f}"
        )

        return self._format_results(candidates, cluster_info)

    def get_brands(self) -> list[str]:
        return sorted(self.df["Brand"].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Helpers (shared with MLLaptopRecommender)
    # ------------------------------------------------------------------

    def _build_feature_matrix(self, df: pd.DataFrame):
        """Same as MLLaptopRecommender — weighted numeric + TF-IDF."""
        numeric = df[self._numeric_cols].fillna(0).values.astype(float)
        # scaler already fitted by _run_denstream — transform only
        numeric_scaled   = self.scaler.transform(numeric)
        numeric_weighted = numeric_scaled * self._weights
        numeric_sparse   = csr_matrix(numeric_weighted)

        text_corpus = (
            df["Processor_Name"].fillna("") + " "
            + df["GPU"].fillna("") + " "
            + df["Brand"].fillna("")
        )
        tfidf_matrix = self.tfidf.fit_transform(text_corpus)
        return hstack([numeric_sparse, tfidf_matrix])

    @staticmethod
    def _extract_gpu_vram(gpu):
        if pd.isna(gpu):
            return 0
        g = str(gpu).upper()
        if any(k in g for k in ("INTEGRATED", "IRIS XE", "IRIS", "UHD")):
            return 0
        if "RADEON RX" in g or "GEFORCE" in g or "RTX" in g or "GTX" in g:
            return extract_gb(gpu)
        return 0

    @staticmethod
    def _format_results(df: pd.DataFrame, cluster_info: str = "") -> list[dict]:
        results = []
        for idx, row in df.iterrows():
            results.append({
                "Brand":          row["Brand"],
                "Name":           row["clean_name"],
                "Price":          int(row["Price"]),
                "Processor_Name": str(row["Processor_Name"]) if pd.notna(row.get("Processor_Name")) else "",
                "RAM":            str(row["RAM"])             if pd.notna(row.get("RAM"))             else "",
                "SSD":            str(row["SSD"])             if pd.notna(row.get("SSD"))             else "",
                "GPU":            str(row["GPU"])             if pd.notna(row.get("GPU"))             else "",
                "Display":        str(row["Display"])         if pd.notna(row.get("Display"))         else "",
                "Battery_Life":   str(row["Battery_Life"])    if pd.notna(row.get("Battery_Life"))    else "",
                "ml_score":       round(float(row["ml_score"]), 4),
                "cluster_info":   cluster_info,
                "image_url":      f"https://picsum.photos/seed/{hash(str(row['Brand'])+str(idx)) % 100000}/400/300",
            })
        return results
