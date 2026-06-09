"""Laptop recommendation engine based on user type and domain."""
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path

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
    "gamer_casual_gaming": (16, 512, 4, 2, 1),
    "gamer_esports": (16, 512, 6, 3, 2),
    "gamer_streaming": (16, 512, 6, 3, 2),
    "gamer_general": (16, 512, 6, 3, 2),
    "video_editing": (16, 512, 4, 4, 2),
    "video_editing_content_creation": (16, 512, 4, 4, 2),
    "video_editing_professional": (16, 512, 4, 4, 2),
    "video_editing_hobbyist": (16, 512, 2, 3, 1),
    "video_editing_general": (16, 512, 4, 4, 2),
    "staff": (8, 256, 0, 1, 0),
    "staff_office_work": (8, 256, 0, 1, 0),
    "staff_administration": (8, 256, 0, 1, 0),
    "staff_general": (8, 256, 0, 1, 0),
    "normal_user": (8, 256, 0, 2, 0),
    "normal_user_everyday_use": (8, 256, 0, 2, 0),
    "normal_user_entertainment": (8, 256, 0, 2, 0),
    "normal_user_productivity": (8, 256, 0, 2, 0),
    "normal_user_general": (8, 256, 0, 2, 0),
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

    def get_recommendations(self, user_type, domain=None, brand=None, min_price=None, max_price=None, max_results=24):
        user_type = str(user_type).lower().replace(" ", "_")
        domain = (domain or "general").lower().replace(" ", "_").replace("&", "_").replace("and", "_")
        key = f"{user_type}_{domain}"
        if key not in DOMAIN_SPECS:
            key = (
                f"{user_type}_general" if f"{user_type}_general" in DOMAIN_SPECS
                else user_type if user_type in DOMAIN_SPECS
                else "normal_user"
            )
        min_ram, min_ssd, min_gpu_vram, min_proc_tier, min_gpu_tier = DOMAIN_SPECS.get(key, DOMAIN_SPECS["normal_user"])

        df = self.df.copy()
        if min_price is not None and min_price > 0:
            df = df[df["Price"] >= min_price]
        if max_price is not None and max_price > 0:
            df = df[df["Price"] <= max_price]
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