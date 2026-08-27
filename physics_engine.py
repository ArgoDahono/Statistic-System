import pandas as pd
import numpy as np
import joblib

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Define Leakage / physiology helpers
def leakage_index(albumin, hematokrit):
    if albumin in (0, 0.0) or pd.isna(albumin) or pd.isna(hematokrit):
        return None
    try:
        return float(hematokrit) / float(albumin)
    except Exception:
        return None

def validate_physiology(albumin, hematokrit):
    idx = leakage_index(albumin, hematokrit)
    if idx is None:
        return {'status': 'Invalid', 'index': None}
    if idx < 1.5:
        status = 'Normal'
    elif idx < 2.0:
        status = 'Risiko Kebocoran'
    else:
        status = 'Kebocoran Plasma'
    return {'status': status, 'index': idx}

# Define Rule-based risk engine: derive label from clinical columns
def risk_engine_from_row(row):
    def truth(r, keys):
        for k in keys:
            if k in r and pd.notna(r[k]):
                val = r[k]
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in ('yes', 'y', '1', 'true', 'positif', 'positive'):
                        return True
                    if v in ('no', 'n', '0', 'false', 'negatif'):
                        return False
                else:
                    try:
                        if float(val) != 0:
                            return True
                    except:
                        pass
        return False

    demam = truth(row, ['Demam','fever','Demam_flag'])
    rumple = truth(row, ['Rumple_Leede_Positive','Rumple_Leede','Tourniquet_Positive'])
    pendarahan = truth(row, ['Pendarahan_Spontan','Pendarahan','Bleeding_spontan'])
    nadi_cepat_lemah = truth(row, ['Nadi_Rapid_Weak','Nadi_cepat_lemah','Pulse_Rapid_Weak'])
    hipotensi = truth(row, ['Hipotensi','Hypotension'])
    kulit_dingin = truth(row, ['Kulit_Dingin_Lembab','Kulit_Dingin','Cold_Damp_Skin'])
    gelisah = truth(row, ['Gelisah','Agitated'])
    nadi_tidak_teraba = truth(row, ['Nadi_Tidak_Teraba','Pulse_None'])
    tekanan_tidak_terukur = truth(row, ['Tekanan_Darah_Tidak_Terukur','BP_Unmeasurable'])

    tekanan_nadi = None
    for col in ['Tekanan_Nadi','Pulse_Pressure','Tekanan_nadi']:
        if col in row and pd.notna(row[col]):
            try:
                tekanan_nadi = float(row[col])
                break
            except:
                pass

    if nadi_tidak_teraba or tekanan_tidak_terukur:
        return 'IV'
    if nadi_cepat_lemah and ((tekanan_nadi is not None and tekanan_nadi <= 20) or hipotensi or kulit_dingin or gelisah):
        return 'III'
    if demam and pendarahan:
        return 'II'
    if demam and rumple:
        return 'I'
    return None

def create_derajat_column(df, target_col='Derajat_DBD'):
    df = df.copy()
    df[target_col] = df.apply(lambda r: risk_engine_from_row(r), axis=1)
    df[target_col] = df[target_col].where(df[target_col].notna(), np.nan)
    return df

# Define Machine learning pipeline
def train_risk_engine_ml(df, feature_cols, label_col='Derajat_DBD', save_path='risk_engine_pipeline.joblib'):
    if label_col not in df.columns:
        df = create_derajat_column(df, target_col=label_col)
    df_train = df.dropna(subset=feature_cols + [label_col]).copy()
    if df_train.empty:
        raise ValueError("Tidak ada data untuk training setelah dropna.")
    le = LabelEncoder()
    y = le.fit_transform(df_train[label_col].astype(str))
    X = df_train[feature_cols]

    # safe numeric/categorical detection
    numeric_feats = [c for c in feature_cols if c in df_train.columns and pd.api.types.is_numeric_dtype(df_train[c])]
    categorical_feats = [c for c in feature_cols if c in df_train.columns and c not in numeric_feats]

    numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preproc = ColumnTransformer([('num', numeric_pipeline, numeric_feats), ('cat', categorical_pipeline, categorical_feats)], remainder='drop')

    pipeline = Pipeline([('preproc', preproc), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Laporan Klasifikasi Risk Engine ML ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump({'pipeline': pipeline, 'label_encoder': le}, save_path)
    return pipeline, le, X_test, y_test, y_pred

def predict_risk_engine_ml(pipeline, le, data_row, feature_cols):
    if isinstance(data_row, pd.Series):
        data_row = data_row.to_dict()
    X = pd.DataFrame([{c: data_row.get(c, np.nan) for c in feature_cols}])
    probs = pipeline.predict_proba(X)[0]
    pred_idx = int(pipeline.predict(X)[0])
    label_text = le.inverse_transform([pred_idx])[0]
    proba_dict = {le.classes_[i]: float(probs[i]) for i in range(len(probs))}
    interp_map = {
        'I': "Derajat I: Demam dengan manifestasi perdarahan hanya Rumple-Leede positif.",
        'II': "Derajat II: Demam dan perdarahan spontan (petekie/mimisan/gusi berdarah).",
        'III': "Derajat III (DSS): Kegagalan sirkulasi: nadi cepat/lemah, tekanan nadi ≤20 mmHg atau hipotensi, kulit dingin/lembab, gelisah.",
        'IV': "Derajat IV (DSS Berat): Syok berat, nadi tidak teraba, tekanan tidak terukur."
    }
    return label_text, interp_map.get(str(label_text), ""), proba_dict

def generate_bab_iv(df, feature_cols, label_col, pipeline=None, le=None, X_test=None, y_test=None, y_pred=None):
    lines = []
    lines.append("BAB IV\nHASIL DAN PEMBAHASAN\n")
    if label_col in df.columns:
        distrib = Counter(df[label_col].dropna().astype(str))
        lines.append("A. Distribusi Derajat DBD pada dataset:")
        for k, v in distrib.items():
            lines.append(f"- Derajat {k}: {v} kasus")
    lines.append("\nB. Statistik Deskriptif Fitur Utama:")
    for c in feature_cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            lines.append(f"- {c}: n={len(s)}, mean={s.mean():.2f}, sd={s.std():.2f}, median={s.median():.2f}")
    if pipeline is not None and le is not None and X_test is not None and y_test is not None and y_pred is not None:
        lines.append("\nC. Evaluasi Model Machine Learning:")
        lines.append(classification_report(y_test, y_pred, target_names=le.classes_))
        lines.append("\nConfusion Matrix:")
        lines.append(str(confusion_matrix(y_test, y_pred)))
        try:
            clf = pipeline.named_steps['clf']
            preproc = pipeline.named_steps['preproc']
            feat_names = []
            for name, trans, cols in preproc.transformers_:
                if name == 'num':
                    feat_names.extend(list(cols))
                elif name == 'cat':
                    ohe = trans.named_steps.get('onehot')
                    if ohe is not None:
                        cats = ohe.get_feature_names_out(cols)
                        feat_names.extend(list(cats))
            importances = clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            lines.append("\nTop Feature Importance:")
            for i in top_idx:
                if i < len(feat_names):
                    lines.append(f"- {feat_names[i]}: {importances[i]:.4f}")
        except Exception:
            lines.append("\n(Failed to extract feature importance)")
    lines.append("\nD. Pembahasan Fisiologis dan Patofisiologis:")
    lines.append("Albumin dan Hematokrit adalah indikator status cairan dan kebocoran plasma. Peningkatan permeabilitas kapiler pada DBD menyebabkan kebocoran plasma sehingga hematokrit relatif meningkat dan albumin cenderung menurun. Kombinasi parameter klinis dan laboratoris diperlukan untuk stratifikasi risiko.")
    lines.append("\nE. Implikasi Klinis:")
    lines.append("Model prediktif membantu triase namun perlu validasi dan integrasi klinis.")
    return "\n".join(lines)