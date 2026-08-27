import sys
import os
import json
import pandas as pd
import hashlib

sys.path.append(str(Path(__file__).parent))

from datetime import datetime
from pathlib import Path
from data.loader import load_and_clean_data
from models.physics import physics_engine, validate_physiology
from models.westgard import analyze_westgard_control, plot_westgard
from models.risk_engine import train_risk_engine_ml, predict_risk_engine_ml
from analysis.stats import analyze_dataset
from visualization.plots import scatter_hct_albumin
from utils.export import export_stats_to_excel

def full_pipeline():
    print("DBD Risk Engine - Full Pipeline")
    print("=" * 50)
    
    # 1. Data Loading & Cleaning
    print("\n1.Loading & Cleaning Data...")
    df = load_and_clean_data()
    
    # 2. Statistical Analysis
    print("\n2.Statistical Analysis...")
    stats = analyze_dataset(df)
    
    # 3. Physics Engine Validation
    print("\n3.Physics Engine...")
    violations = physics_engine(df)
    if violations:
        print("Violations:", violations)
    
    # Sample leakage analysis
    sample = df.iloc[0]
    phys_result = validate_physiology(sample['Albumin_g/dL'], sample['HCT_%'])
    print(f"Sample leakage: {phys_result}")
    
    # 4. Westgard QC (Albumin)
    print("\n4.Westgard Quality Control...")
    try:
        mean_alb, sd_alb, w_results = analyze_westgard_control('Albumin_g/dL', df)
        plot_westgard(df['Albumin_g/dL'].dropna().tolist(), mean_alb, sd_alb, w_results)
    except Exception as e:
        print(f"Westgard skipped: {e}")
    
    # 5. Visualizations
    print("\n5.Visualizations...")
    scatter_hct_albumin(df)
    
    # 6. SINGLE ML Training
    print("\n6.Training Risk Engine ML...")
    feature_cols = ['Umur_Tahun', 'HCT_%', 'Albumin_g/dL']  # Add more as available
    pipeline, le = train_risk_engine_ml(df, feature_cols)
    
    # 7. Demo Prediction
    print("\n7.Demo Prediction...")
    sample_dict = sample.to_dict()
    pred_label, probs = predict_risk_engine_ml(pipeline, le, sample_dict)
    print(f"Patient {sample.get('ID_Pasien', 'Unknown')}: {pred_label}")
    print(f"Probabilities: { {k: f'{v:.1%}' for k,v in probs.items()} }")
    
    # 8. Export Results
    print("\n8.Export Results...")
    export_stats_to_excel(stats, 'results_stats.xlsx')
    
    print("\n Pipeline COMPLETE!")
    print("Files generated:")
    print("- westgard.png")
    print("- scatter_plot.png") 
    print("- risk_engine.onnx / .joblib")
    print("- results_stats.xlsx")

if __name__ == "__main__":
    full_pipeline()



# MODULE DATA VERSIONING ENGINE
class DataVersioningEngine:
    """
    Engine untuk versioning dataset:
    - Save version
    - Metadata tracking
    - Version tagging
    - Load version
    """

    def __init__(self, base_path="data_versions"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # HASH GENERATOR
    def _generate_hash(self, df):
        data_bytes = df.to_csv(index=False).encode()
        return hashlib.md5(data_bytes).hexdigest()

    # SAVE VERSION
    def save_version(self, df, metadata=None, tag="experimental"):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_hash = self._generate_hash(df)

        version_name = f"dataset_{timestamp}_{data_hash[:8]}"
        version_path = os.path.join(self.base_path, version_name)

        os.makedirs(version_path, exist_ok=True)

        # Save data
        data_file = os.path.join(version_path, "data.csv")
        df.to_csv(data_file, index=False)

        # Metadata
        metadata_dict = {
            "version_name": version_name,
            "timestamp": timestamp,
            "hash": data_hash,
            "rows": len(df),
            "columns": list(df.columns),
            "tag": tag,
            "user_metadata": metadata if metadata else {}
        }

        metadata_file = os.path.join(version_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata_dict, f, indent=4)

        return version_name

    # LIST VERSIONS
    def list_versions(self):

        versions = []

        for folder in os.listdir(self.base_path):
            meta_path = os.path.join(self.base_path, folder, "metadata.json")

            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)

                versions.append({
                    "version": meta["version_name"],
                    "timestamp": meta["timestamp"],
                    "rows": meta["rows"],
                    "tag": meta["tag"]
                })

        return pd.DataFrame(versions)

    # LOAD VERSION
    def load_version(self, version_name):

        path = os.path.join(self.base_path, version_name, "data.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Version {version_name} tidak ditemukan")

        return pd.read_csv(path)

    # COMPARE VERSION (SIMPLE)
    def compare_versions(self, version1, version2):

        df1 = self.load_version(version1)
        df2 = self.load_version(version2)

        return {
            "shape_v1": df1.shape,
            "shape_v2": df2.shape,
            "row_diff": len(df2) - len(df1),
            "column_diff": list(set(df2.columns) - set(df1.columns))
        }
    
# orchestrator
def step_auto_versioning(self):

    version_engine = DataVersioningEngine()

    version_name = version_engine.save_version(
        self.df,
        metadata={"step": "final_output"},
        tag="production"
    )

    return {"version": version_name}

def _safe_execute(self, step_name, func):
    try:
        result = func()
        self.results[step_name] = result
        print(f"{step_name} berhasil")
    except Exception as e:
        print(f"{step_name} error: {e}")
        self.results[step_name] = {"error": str(e)}