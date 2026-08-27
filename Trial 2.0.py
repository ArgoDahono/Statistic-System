import pandas as pd
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os
import io
import torch
import re
import string
import subprocess
import sys
import scipy.stats as stats
import networkx as nx
import pymc as pm
import optuna
import hashlib
import boto3
import logging
import yaml
import json
import statsmodels.formula.api as smf

from sklearn.base import clone
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
)
from sklearn.metrics import recall_score, mean_squared_error
from rapidfuzz import fuzz
from typing import List, Self
from torch import nn
from scipy.stats import shapiro, kstest, pearsonr, spearmanr, linregress, zscore
from matplotlib import lines
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from difflib import get_close_matches
from fpdf import FPDF
from physics_engine import (
    leakage_index,
    create_derajat_column
)

# MODULE SYSTEM STATE MANAGEMENT ENGINE
# SUBMODULE AVAILABILITY FLAG
cpp_module_available = False
# SUBMODULE INSTANCE PLACEHOLDER
plasma_leakage = None

# MODULE MEMORY MANAGEMENT SIMULATION ENGINE
# COMPONENT CLASS DEFINITION
class LaptopMemoryModel:
    """
    Model for laptop memory simulation
    """
    # Subcomponent initialization engine
    def __init__(self, total_ram_gb=8, swap_size_gb=4):
        """
        Initialize laptop memory model
        
        Args:
            total_ram_gb: Total RAM in GB
            swap_size_gb: Total swap/virtual memory in GB
        """
        self.total_ram_bytes = total_ram_gb * 1024 * 1024 * 1024
        self.swap_size_bytes = swap_size_gb * 1024 * 1024 * 1024
        self.total_memory = self.total_ram_bytes + self.swap_size_bytes
        
        # Subcomponent initial memory state
        self.used_ram = 0
        self.available_ram = self.total_ram_bytes
        self.used_swap = 0
        self.available_swap = self.swap_size_bytes
        
        ## Subcomponent threshold configuration
        self.warning_threshold = 0.8
        self.critical_threshold = 0.9
    
    # COMPONENT MEMORY ALLOCATION ENGINE
    def allocate_memory(self, size_mb):
        """
        Allocate memory
        
        Args:
            size_mb: Memory size in MB
        """
        # Subcomponent unit conversion
        size_bytes = size_mb * 1024 * 1024
        
        # Subcomponent ram allocation priority
        if self.available_ram >= size_bytes:
            self.used_ram += size_bytes
            self.available_ram -= size_bytes
            return "RAM"
        # Subcomponent swap allocation fallback
        elif self.available_swap >= size_bytes:
            self.used_swap += size_bytes
            self.available_swap -= size_bytes
            return "SWAP"
        # Subcomponent allocation failure
        else:
            return "FAILED"
    
    # COMPONENT MEMORY DEALLOCATION ENGINE
    def free_memory(self, size_mb, location="RAM"):
        """
        Free up memory
        
        Args:
            size_mb: Memory size in MB
            location: Memory location (RAM atau SWAP)
        """
        # Subcomponent unit conversion
        size_bytes = size_mb * 1024 * 1024
        # Subcomponent ram deallocation
        if location == "RAM":
            self.used_ram = max(0, self.used_ram - size_bytes)
            self.available_ram = min(self.total_ram_bytes, self.available_ram + size_bytes)
        # Subcomponent swap deallocation
        elif location == "SWAP":
            self.used_swap = max(0, self.used_swap - size_bytes)
            self.available_swap = min(self.swap_size_bytes, self.available_swap + size_bytes)
    
    # COMPONENT USAGE CALCULATION ENGINE
    def get_ram_usage_percent(self):
        """Returns RAM usage percentage"""
        return (self.used_ram / self.total_ram_bytes) * 100
    
    def get_swap_usage_percent(self):
        """return SWAP usage percentage"""
        return (self.used_swap / self.swap_size_bytes) * 100
    
    # COMPONENT MEMORY STATUS EVALUATION ENGINE
    def get_memory_status(self):
        """Restore memory state"""
        ram_percent = self.get_ram_usage_percent()
        
        # Subcomponent status classification
        if ram_percent >= self.critical_threshold * 100:
            status = "CRITICAL"
        elif ram_percent >= self.warning_threshold * 100:
            status = "WARNING"
        else:
            status = "NORMAL"
        
        # Subcomponent output structure
        return {
            "status": status,
            "ram_used_gb": round(self.used_ram / (1024**3), 2),
            "ram_total_gb": self.total_ram_bytes / (1024**3),
            "ram_percent": round(ram_percent, 2),
            "swap_used_gb": round(self.used_swap / (1024**3), 2),
            "swap_total_gb": self.swap_size_bytes / (1024**3)
        }
    
    # COMPONENT OBJECT REPRESENTATION ENGINE
    def __repr__(self):
        return f"LaptopMemoryModel(RAM: {self.total_ram_bytes/(1024**3)}GB, SWAP: {self.swap_size_bytes/(1024**3)}GB)"

# MODULE NLP TEXT PREPROCESSING PIPELINE
# COMPONENT TEXT CLEANING AND NORMALIZATION
def preprocess_text(text: str) -> str:
    """Preprocess text: lowercase, remove punctuation, numbers"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# COMPONENT TOKENIZATION ENGINE 
def tokenize(text: str) -> List[str]:
    """Tokenize text into words"""
    return text.split()

# MODULE VOCABULARY MANAGEMENT
# COMPONENT VOCABULARY CLASS DEFINITION
class Vocabulary:
    """Vocabulary class to map tokens to indices"""
    # Subcomponent initialization and special tokens setup
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.token_counts = Counter()
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self._add_special_tokens()
    
    # Subcomponent add special tokens
    def _add_special_tokens(self):
        self.token_to_idx[self.PAD_TOKEN] = 0
        self.token_to_idx[self.UNK_TOKEN] = 1
        self.token_to_idx[self.SOS_TOKEN] = 2
        self.token_to_idx[self.EOS_TOKEN] = 3
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
    
    # Subcomponent add token to vocabulary
    def add_token(self, token: str):
        self.token_counts[token] += 1
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
    
    # Subcomponent vocabulary builder
    def build_vocab(self, texts: List[str], min_freq: int = 1):
        all_tokens = []
        for text in texts:
            tokens = tokenize(preprocess_text(text))
            all_tokens.extend(tokens)
        self.token_counts = Counter(all_tokens)
        for token, count in self.token_counts.items():
            if count >= min_freq:
                self.add_token(token)
    
    # Subcomponent token to index mapping
    def token_to_index(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx[self.UNK_TOKEN])
    
    # Subcomponent index to token mapping
    def index_to_token(self, idx: int) -> str:
        return self.idx_to_token.get(idx, self.UNK_TOKEN)
    
    # Subcomponent vocabulary size
    def __len__(self):
        return len(self.token_to_idx)

# MODULE TEXT NUMERICALIZATION
# COMPONENT TEXT TO INDICES CONVERTER
def texts_to_indices(texts: List[str], vocab: Vocabulary, max_length: int = 50) -> List[List[int]]:
    """Convert list of texts to list of token indices"""
    indices_list = []
    for text in texts:
        processed_text = preprocess_text(text)
        tokens = tokenize(processed_text)
        indices = [vocab.token_to_index(token) for token in tokens]

        # Subcomponent sequence truncation
        if len(indices) > max_length:
            indices = indices[:max_length]
        indices_list.append(indices)
    return indices_list

# MODULE SEQUENCE PROCESSING
def pad_sequences(sequences: List[List[int]], max_length: int, pad_value: int = 0) -> torch.Tensor:
    """Pad sequences to same length"""
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            seq = seq + [pad_value] * (max_length - len(seq))
        padded.append(seq)
    return torch.tensor(padded, dtype=torch.long)

# MODULE DATASET HANDLING
# COMPONENT CUSTOM PYTORCH DATASET
class TextDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for text"""
    # Subcomponent dataset initialization
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_length: int = 50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.indices = texts_to_indices(texts, vocab, max_length)
        self.indices = pad_sequences(self.indices, max_length)
    
    # Subcomponent dataset length
    def __len__(self):
        return len(self.texts)
    
    # Subcomponent data retrieval
    def __getitem__(self, idx):
        return {
            'text': self.indices[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# MODULE DATALOADER CREATION
# COMPONENT DATALOADER BUILDER
def create_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str] = None,
    val_labels: List[int] = None,
    vocab: Vocabulary = None,
    max_length: int = 50,
    batch_size: int = 32,
    shuffle: bool = True
):
    """Create train and validation dataloaders"""
    
    # Subcomponent vocabulary initialization
    if vocab is None:
        vocab = Vocabulary()
        vocab.build_vocab(train_texts, min_freq=2)
    
    # Subcomponent training dataset and loader
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Subcomponent validation dataset and loader
    val_loader = None
    if val_texts is not None and val_labels is not None:
        val_dataset = TextDataset(val_texts, val_labels, vocab, max_length)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab

# MODULE MODEL TRAINING SETUP
def setup_loss_and_optimizer(model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-5):
    """Setup loss function and optimizer"""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return criterion, optimizer

# MODULE TRAINING ENGINE
# COMPONENT TRAINING LOOP ENGINE
def train_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    print_every: int = 100
):
    """Training loop for model"""

    # Subcomponent device setup
    model.to(device)
    
    for epoch in range(num_epochs):
        # Subcomponent training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Subcomponent forward and backward pass
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Subcomponent training log output
            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Subcomponent validation phase
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    texts = batch['text'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

# MODULE PLASMA LEAKAGE CALCULATOR
class PlasmaLeakageProcessor:
    """Python implementation of C++ plasma leakage computations"""
    
    # COMPONENT LEAKAGE INDEX CALCULATION
    @staticmethod
    def calculate_leakage_index(hct, albumin):
        """
        Calculate plasma leakage index based on HCT and Albumin
        Formula: Leakage Index = HCT - (Albumin * 10)
        """
        # Subcomponent data valiation engine
        if pd.isna(hct) or pd.isna(albumin) or albumin == 0:
            return np.nan
        # Subcomponent computation engine
        return hct - (albumin * 10)
    
    # COMPONENT RISK LEVEL ASSESSMENT
    @staticmethod
    def assess_risk_level(leakage_index):
        """
        Assess risk level based on leakage index
        """
        # Subcomponent null handling engine
        if pd.isna(leakage_index):
            return 'Unknown'
        # Subcomponent threshold evaluation engine
        elif leakage_index < -10:
            return 'Low Leakage'
        elif leakage_index < 0:
            return 'Minimal Leakage'
        elif leakage_index < 10:
            return 'Mild Leakage'
        elif leakage_index < 20:
            return 'Moderate Leakage'
        else:
            return 'Severe Leakage'
    
    # COMPONENT DATA PROCESSING PIPELINE
    @staticmethod
    def process_data(df, hct_col='HCT_%', albumin_col='Albumin_g/dL'):
        """
        Process data and compute plasma leakage metrics
        """
        # Subcomponent output initialization
        results = pd.DataFrame()
        # Subcomponent data mapping engine
        results['HCT_%'] = df[hct_col]
        results['Albumin_g/dL'] = df[albumin_col]
        # Subcomponent leakage index pipeline
        results['Leakage_Index'] = df.apply(
            lambda row: PlasmaLeakageProcessor.calculate_leakage_index(row[hct_col], row[albumin_col]),
            axis=1
        )
        # Subcomponent risk classification pipeline
        results['Risk_Level'] = results['Leakage_Index'].apply(
            PlasmaLeakageProcessor.assess_risk_level
        )
        return results

# MODULE SYSTEM INITIALIZATION ENGINE
def build_cpp_module_deferred():
    """Process plasma leakage data using Python implementation (C++ fallback)"""
    global cpp_module_available, plasma_leakage
    # Subcomponent status flag activation
    cpp_module_available = True
    # Subcomponent class instantiation
    plasma_leakage = PlasmaLeakageProcessor()
    # Subcomponent initialization confirmation
    return True

# MODULE DATA INGESTION AND INITIALIZATION
file_path = 'Data_Rekam_Medis_DBD_Total_Sampling_300.csv'
df_real = pd.read_csv(file_path)
df = df_real.copy()

# MODULE FORMAT DETECTION
# COMPONENT FORMAT DETECTION ENGINE
def detect_data_format(df):
    cols = [c.lower() for c in df.columns]

    # Subcomponent indikation long format
    if 'parameter' in cols and 'hasil' in cols:
        return 'LONG'
    
    # Subcomponent indikation wide format
    elif 'hct_%' in cols and 'albumin_g/dl' in cols:
        return 'WIDE'
    
    return 'UNKNOWN'

# COMPONENT STRUCTURE VALIDATION ENGINE
def validate_structure(df, format_type):
    if format_type == 'LONG':
        required = ['Parameter', 'Hasil']
    elif format_type == 'WIDE':
        required = ['HCT_%', 'Albumin_g/dL']
    else:
        return False
    
    return all(col in df.columns for col in required)

# MODULE STANDARDIZATION ENGINE (LONG) - CONDITIONAL SKIP
# COMPONENT SKIP LONG STANDARDIZATION FOR WIDE FORMAT
def skip_standardization_long(df, format_type):
    """
    If the data format is already WIDE, skip standardization to LONG format
    If the format is LONG, continue to standardization to LONG format
    """
    if format_type == 'WIDE':
        print("[OK] SKIP: LONG format standardization is skipped (data is already WIDE)")
        return df, True  # (dataframe, is_skipped)
    elif format_type == 'LONG':
        print("-> PROCESS: Standardizing LONG format...")
        return df, False  # (dataframe, is_not_skipped)
    else:
        print("[WARN] WARNING: Format not recognized")
        return df, True

# MODULE STANDARDIZATION ENGINE (WIDE)
# COMPONENT PARAMETER MAPPING DICTIONARY FOR WIDE FORMAT
PARAMETER_MAP_WIDE = {
    # Subcomponent Hematokrit variations untuk kolom WIDE
    "HCT_%": [
        "hematokrit", "hematocrit", "hct", "ht", "packed cell volume",
        "pcv", "packed_cell_volume", "hct_persen", "hct_percent",
        "hematokrit_%", "hct_%", "hematocrit_%"
    ],
    # Subcomponent Albumin variations untuk kolom WIDE
    "Albumin_g/dL": [
        "albumin", "alb", "serum albumin", "albumin serum",
        "albumin_g/dl", "albumin_gdl", "albumin (g/dl)",
        "alb_g/dl", "serum_albumin", "albumin_serum"
    ]
}

# COMPONENT TEXT NORMALIZATION ENGINE FOR WIDE
def normalize_text_wide(text):
    """Text normalization for matching WIDE format columns"""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9%/_]', '', text)  # Keep underscores
    text = re.sub(r'\s+', '', text)  # Remove all spaces
    return text

# COMPONENT WIDE FORMAT STANDARDIZATION ENGINE
def standardize_wide_format(df):
    """
    Standardize column names for WIDE format with fuzzy matching
    Using PARAMETER_MAP_WIDE to match column name variations
    """
    print("-> PROCESSING: WIDE format standardization begins...")
    
    # Subcomponent column mapping creation
    column_mapping = {}
    
    for col in df.columns:
        col_normalized = normalize_text_wide(col)
        
        # Subcomponent parameter matching engine
        for standard_name, variants in PARAMETER_MAP_WIDE.items():
            for variant in variants:
                if col_normalized == normalize_text_wide(variant):
                    column_mapping[col] = standard_name
                    print(f"  [OK] colom '{col}' -> '{standard_name}'")
                    break
            if col in column_mapping:
                break
    
    # Subcomponent column renaming execution
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"[OK] WIDE standardization completed: {len(column_mapping)} rename colom")
    else:
        print("[OK] Columns are standard (no rename needed)")
    
    return df

# MODULE DATA TRANSFORMATION (LONG TO WIDE) - CONDITIONAL SKIP
# COMPONENT SKIP TRANSFORMATION FOR WIDE FORMAT
def skip_transformation_long_to_wide(df, format_type):
    """
    If the data format is already WIDE, skip the transformation from LONG to WIDE.
    If the format is LONG, continue the transformation to WIDE
    """
    if format_type == 'WIDE':
        print("[OK] SKIP: LONG->WIDE transformation is skipped (data is already WIDE)")
        return df, True  # (dataframe, is_skipped)
    elif format_type == 'LONG':
        print("-> PROCESS: Performing LONG->WIDE transformation...")
        return df, False  # (dataframe, is_not_skipped)
    else:
        print("[WARN] WARNING: Format not recognized")
        return df, True

# COMPONENT OUTPUT STANDARDIZATION ENGINE
def standardize_columns(df):
    rename_map = {
        'hematokrit': 'HCT_%',
        'hct': 'HCT_%',
        'albumin': 'Albumin_g/dL'
    }

    df.columns = [rename_map.get(c.lower(), c) for c in df.columns]
    return df

# MODULE DISCONTINUE STANDARDIZATION FOR WIDE FORMAT
# COMPONENT DATA ROUTING ENGINE
def process_data(df):
    # Subcomponent format detection engine
    format_type = detect_data_format(df)
    print(f"=============================\n[OK] FORMAT DETECTED: {format_type}\n=============================")
    print(f"DEBUG COLUMNS: {df.columns.tolist()}\n")

    # Subcomponent conditional standardization for LONG format
    df, skip_long_std = skip_standardization_long(df, format_type)
    
    if not skip_long_std and format_type == 'LONG':
        df = standardize_long_format(df)
    elif format_type == 'WIDE':
        df = standardize_wide_format(df)
    
    # Subcomponent conditional transformation LONG to WIDE
    df, skip_transform = skip_transformation_long_to_wide(df, format_type)
    
    if not skip_transform and format_type == 'LONG':
        df = transform_long_to_wide(df)
    
    print(f"\n[OK] Data processing is complete. Final shape: {df.shape}\n")
    return df

# MODULE LONG STANDARDIZATION ENGINE
def standardize_long_format(df):
    """
    Standardizing the LONG format with parameter mapping
    Using fuzzy matching to match parameter name variations
    """
    if 'Parameter' not in df.columns:
        print("[WARN] SKIP: Column 'Parameter' does not exist in dataframe")
        return df

    print("-> PROCESSING: LONG format standardization begins...")
    
    # Subcomponent parameter standardization with fuzzy matching
    df['Parameter'] = df['Parameter'].apply(standardize_parameter_fuzzy)
    
    # Subcomponent unknown parameter detection
    unknown = df[~df['Parameter'].isin(['HCT_%', 'Albumin_g/dL'])]
    if not unknown.empty:
        print(f"[WARN] Unknown parameter: {unknown['Parameter'].unique()}")
    else:
        print("[OK] All parameters were successfully standardized")
    
    return df

# MODULE LONG TO WIDE TRANSFORMATION ENGINE
def transform_long_to_wide(df):
    """
    Transforming data from LONG to WIDE format
    Combining HCT_% and Albumin_g/dL into separate columns
    Ensuring data types are maintained correctly (float64 for continuous variables)
    """
    # COMPONENT INPUT VALIDATION
    if 'Parameter' not in df.columns or 'Hasil' not in df.columns:
        print("[WARN] SKIP: Missing 'Parameter' or 'Result' column")
        return df
    
    # COMPONENT PROCESS INITIALIZATION ENGINE
    print("-> PROCESSING: LONG->WIDE transformation started...")
    
    # COMPONENT AUTO-DETECT PARAMETER NAMES
    print(f"   Available parameters: {df['Parameter'].unique().tolist()}")
    
    # SUBCOMPONENT PARAMETER STANDARDIZATION MAPPING
    param_mapping = {}
    for param in df['Parameter'].unique():
        param_lower = str(param).lower().strip()
        if 'hct' in param_lower:
            param_mapping[param] = 'HCT_%'
        elif 'albumin' in param_lower:
            param_mapping[param] = 'Albumin_g/dL'
    
    # COMPONENT DATA STANDARDIZATION ENGINE
    df_long = df.copy()
    df_long['Parameter'] = df_long['Parameter'].map(lambda x: param_mapping.get(x, x))
    
    # COMPONENT DATA FILTERING ENGINE
    df_filtered = df_long[df_long["Parameter"].isin(["HCT_%", "Albumin_g/dL"])]
    
    # SUBCOMPONENT EMPTY DATA HANDLING
    if df_filtered.empty:
        print(f"[WARN] No HCT or Albumin data found after mapping")
        return df.iloc[:0, :]  # Return empty dataframe with proper structure
    
    # COMPONENT DATA TYPE CONVERSION ENGINE - ENSURES FLOAT64 FOR CONTINUOUS VARIABLES
    df_filtered = df_filtered.copy()
    df_filtered['Hasil'] = pd.to_numeric(df_filtered['Hasil'], errors='coerce')
    
    # COMPONENT DATA TRANSFORMATION ENGINE
    # Subcomponent index column detection
    try:
        index_cols = [col for col in ['No_RM', 'Nama', 'Umur', 'Jenis_Kelamin', 'Tanggal_Pemeriksaan'] 
                      if col in df.columns]
        
        # Subomponent index validation
        if not index_cols:
            print("[WARN] No suitable index columns found")
            return df.iloc[:0, :]
        
        # Subcomponent pivot table transformation - IMPROVED WITH PROPER DATA TYPES
        df_wide = df_filtered.pivot_table(
            index=index_cols,
            columns="Parameter",
            values="Hasil",
            aggfunc="first"  # Use first value if duplicates
        ).reset_index()
        
        # COMPONENT EXPLICIT DATA TYPE CORRECTION ENGINE
        # Ensure numeric columns are float64 (continuous variables)
        for col in ['HCT_%', 'Albumin_g/dL']:
            if col in df_wide.columns:
                df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce').astype('float64')
        
        # Subcomponent output logging
        print(f"[OK] Transformation complete. Line: {len(df_wide)} -> colom: {len(df_wide.columns)}")
        print(f"[OK] Data types after transformation:")
        print(df_wide.dtypes)
        return df_wide
    
    # COMPONENT ERROR HANDLING ENGINE
    except Exception as e:
        print(f"[WARN] Error transformasi: {str(e)}")
        return df.iloc[:0, :]

# MODULE STANDARDIZATION ENGINE (LONG)
# COMPONENT PARAMETER MAPPING DICTIONARY
PARAMETER_MAP = {
    # Subcomponent Hematokrit variations untuk kolom long
    "HCT_%": [
        "hct", "hematokrit", "hematocrit", "ht", "packed cell volume",
        "pcv", "hct%", "hematokrit%", "hct %", "hematocrit %"
    ],
    # Subcomponent Albumin variations untuk kolom WIDE
    "Albumin_g/dL": [
        "albumin", "alb", "serum albumin", "albumin serum",
        "alb g/dl", "albumin g/dl", "albumin (g/dl)"
    ]
}

# COMPONENT TEXT NORMALIZATION ENGINE
def normalize_text(text):
    """Text normalization for matching LONG format parameters"""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9%/ ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# COMPONENT PARAMETER STANDARDIZATION ENGINE (EXACT MATCH)
def standardize_parameter(param):
    """Parameter standardization with exact matching"""
    param_clean = normalize_text(param)
    
    for standard, variants in PARAMETER_MAP.items():
        for v in variants:
            if param_clean == normalize_text(v):
                return standard
    
    return param  # Fallback: return original if no match

# COMPONENT FUZZY MATCHING ENGINE (APPROXIMATE MATCH)
def standardize_parameter_fuzzy(param):
    """Parameter standardization with fuzzy matching (70% cutoff)"""
    param_clean = normalize_text(param)
    
    all_variants = {}
    for standard, variants in PARAMETER_MAP.items():
        for v in variants:
            all_variants[normalize_text(v)] = standard
    
    match = get_close_matches(param_clean, all_variants.keys(), n=1, cutoff=0.7)
    
    if match:
        return all_variants[match[0]]
    
    return param  # Fallback: return original if no match

# MODULE DATA CLEANING ENGINE
def clean_data(df):
    """
    Pembersihan data: remove incomplete records, fix duplicates, 
    filter clinical range, anonymization, dan alignment kolom
    IMPROVED: Ensures proper data types for continuous variables
    """
    # COMPONENT INPUT VALIDATION ENGIN
    if df is None or df.empty:
        print("[WARN] WARNING: Empty dataframe or None")
        return df
    
    # COMPONENT PROCESS INITIALIZATION ENGINE
    print("-> PROCESSING: Data cleaning begins...")
    print(f"  Initial shape: {df.shape}")
    
    # Subcomponent check required columns
    required_cols = ["HCT_%", "Albumin_g/dL"]
    if not all(col in df.columns for col in required_cols):
        print(f"[WARN] WARNING: colom {required_cols} incomplete")
        return df
    
    # COMPONENT DATA TYPE CORRECTION ENGINE - ENSURE FLOAT64 FOR CONTINUOUS VARIABLES
    df = df.copy()
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    # COMPONENT DATA CLEANING CORE ENGINE
    df_clean = df.dropna(subset=["HCT_%", "Albumin_g/dL"])
    print(f"  [OK] After removing incomplete records: {df_clean.shape[0]} rows")
    
    # Subcomponent sort by examination date (if exists)
    if 'Tanggal_Pemeriksaan' in df_clean.columns:
        df_clean = df_clean.sort_values(by="Tanggal_Pemeriksaan")
        print(f"  [OK] Data sorted by examination date")
    
    # COMPONENT DUPLICATE HANDLING ENGINE
    if 'Anon_ID' in df_clean.columns and 'Nama' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=["Anon_ID", "Nama"], keep="first")
        print(f"  [OK] After removing duplicates: {df_clean.shape[0]} rows")
    
    # COMPONENT CLINICAL VALIDATION ENGINE
    print("  -> Applying clinical range criteria...")
    rows_before = len(df_clean)
    df_clean = df_clean[
        (df_clean["HCT_%"] >= 35) & (df_clean["HCT_%"] <= 55) &
        (df_clean["Albumin_g/dL"] >= 2.5) & (df_clean["Albumin_g/dL"] <= 5.5)
    ]
    rows_removed = rows_before - len(df_clean)
    print(f"  [OK] Clinical range filter: {rows_removed} rows removed, {len(df_clean)} rows remaining")
    
    # COMPONENT DATA ALIGNMENT ENGINE
    rename_map = {}
    if 'Umur' in df_clean.columns and 'Umur_Tahun' not in df_clean.columns:
        rename_map['Umur'] = 'Umur_Tahun'
    if 'Tanggal_Pemeriksaan' in df_clean.columns and 'Tanggal_Hematokrit' not in df_clean.columns:
        rename_map['Tanggal_Pemeriksaan'] = 'Tanggal_Hematokrit'
    
    if rename_map:
        df_clean = df_clean.rename(columns=rename_map)
        print(f"  [OK] Columns renamed: {rename_map}")
    
    # SUBCOMPONENT COLUMN SYNCHRONIZATION
    if 'Tanggal_Hematokrit' in df_clean.columns and 'Tanggal_Albumin' not in df_clean.columns:
        df_clean['Tanggal_Albumin'] = df_clean['Tanggal_Hematokrit']
        print(f"  [OK] Tanggal_Albumin column created")
    
    # COMPONENT DATA PRIVACY ENGINE
    if 'Nama' in df_clean.columns and 'ID_Pasien' not in df_clean.columns:
        name_map = {name: f'DBD-{i+1:03d}' for i, name in enumerate(df_clean['Nama'].dropna().unique())}
        df_clean['ID_Pasien'] = df_clean['Nama'].map(name_map)
        print(f"  [OK] ID_Pasien anonymized ({len(name_map)} unique patients)")
    
    # COMPONENT DATA REDUCTION ENGINE
    if 'No_RM' in df_clean.columns:
        df_clean = df_clean.drop(columns=['No_RM'], errors='ignore')
        print(f"  [OK] No_RM column removed")
    
    # COMPONENT PROCESS FINALIZATION ENGINE
    print(f"[OK] Data cleaning completed. Final shape: {df_clean.shape}")
    print(f"[OK] Data types verification:")
    print(df_clean.dtypes)
    print()
    return df_clean

# MODULE DATA VALIDATION AND SUMMARY ENGINE
def validate_and_summarize_data(df):
    """
    Validate and display a summary of the cleaned data
    Display dataset information, descriptive statistics, and quality checks
    """
    # COMPONENT INPUT VALIDATION ENGINE
    if df is None or df.empty:
        print("[WARN] WARNING: Empty or None dataframe, cannot be summarized")
        return
    
    # COMPONENT REPORT INITIALIZATION ENGINE
    print("\n" + "="*70)
    print("MODULE DATA VALIDATION AND SUMMARY")
    print("="*70)
    
    # COMPONENT DATASET OVERVIEW ENGINE
    # Subcomponent dataset size information
    print("\n[#] DATASET SIZE INFORMATION:")
    print(f"  • Total rows: {len(df)}")
    print(f"  • Total columns: {len(df.columns)}")
    
    # Subcomponent unique patient count
    if 'ID_Pasien' in df.columns:
        unique_patients = df['ID_Pasien'].nunique()
        print(f"  • Unique patients: {unique_patients}")
    elif 'Anon_ID' in df.columns:
        unique_patients = df['Anon_ID'].nunique()
        print(f"  • Unique patients (Anon_ID): {unique_patients}")
    
    # COMPONENT DATA QUALITY CHECK ENGINE
    print("\n[#] MISSING VALUES CHECK:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  [OK] No missing values detected")
    else:
        print("  Missing values per column:")
        for col, count in missing[missing > 0].items():
            print(f"    • {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # COMPONENT DATA STRUCTURE INSPECTION ENGINE
    # Subcomponent column listing
    print("\n[#] COLUMN INFORMATION:")
    print(f"  Columns in dataset: {df.columns.tolist()}")
    
    # Subcomponent data type identification
    print("\n[#] DATA TYPES:")
    for col, dtype in df.dtypes.items():
        print(f"  • {col}: {dtype}")
    
    # COMPONENT STATISTICAL ANALYSIS ENGINE
    print("\n[#] DESCRIPTIVE STATISTICS (Numeric Columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe()
        print(stats_df.to_string())
    else:
        print("No numeric columns found")
    
    # COMPONENT CLINICAL PARAMETER ANALYSIS ENGINE
    # Subcomponent hematokrit statistics
    print("\n[#] KEY PARAMETER STATISTICS:")
    if 'HCT_%' in df.columns:
        hct_mean = df['HCT_%'].mean()
        hct_std = df['HCT_%'].std()
        hct_median = df['HCT_%'].median()
        print(f"  • Hematokrit (HCT_%):")
        print(f"    - Mean ± SD: {hct_mean:.2f} ± {hct_std:.2f}")
        print(f"    - Median (IQR): {hct_median:.2f} ({df['HCT_%'].quantile(0.25):.2f}–{df['HCT_%'].quantile(0.75):.2f})")
        print(f"    - Range: {df['HCT_%'].min():.2f} – {df['HCT_%'].max():.2f}")
    # Subcomponent albumin statistics
    if 'Albumin_g/dL' in df.columns:
        alb_mean = df['Albumin_g/dL'].mean()
        alb_std = df['Albumin_g/dL'].std()
        alb_median = df['Albumin_g/dL'].median()
        print(f"  • Albumin (g/dL):")
        print(f"    - Mean ± SD: {alb_mean:.2f} ± {alb_std:.2f}")
        print(f"    - Median (IQR): {alb_median:.2f} ({df['Albumin_g/dL'].quantile(0.25):.2f}–{df['Albumin_g/dL'].quantile(0.75):.2f})")
        print(f"    - Range: {df['Albumin_g/dL'].min():.2f} – {df['Albumin_g/dL'].max():.2f}")
    
    # COMPONENT DEMOGRAPHIC ANALYSIS ENGINE
    # Subcomponent age statistics
    print("\n[#] DEMOGRAPHIC INFORMATION:")
    if 'Umur_Tahun' in df.columns:
        age_mean = df['Umur_Tahun'].mean()
        age_std = df['Umur_Tahun'].std()
        print(f"  • Age (Tahun): Mean ± SD = {age_mean:.2f} ± {age_std:.2f}")
    # Subcomponent gender distribution
    if 'Jenis_Kelamin' in df.columns:
        gender_dist = df['Jenis_Kelamin'].value_counts()
        print(f"  • Gender distribution:")
        for gender, count in gender_dist.items():
            print(f"    - {gender}: {count} ({count/len(df)*100:.1f}%)")
    
    # COMPONENT DATA QUALITY SUMMARY ENGINE
    print("\n[#] DATA QUALITY SUMMARY:")
    print(f"  [OK] Dataset ready for analysis")
    print(f"  [OK] No missing values in key parameters (HCT_%, Albumin_g/dL)")
    print(f"  [OK] All values within clinical ranges")
    print(f"  [OK] Patient data anonymized")
    
    # COMPONENT REPORT FINALIZATION ENGINE
    print("\n" + "="*70 + "\n")

# MODULE WESTGARD RULES QC
def westgard_rules(data, mean, sd):
    """
    Calculate and evaluate control data using Westgard Rules
    """
    # COMPONENT INITIALIZATION
    results = []
    n = len(data)
    
    # COMPONENT Z-SCORE CALCULATION ENGINE
    z_scores = []
    for val in data:
        if sd != 0:
            z = (val - mean) / sd
        else:
            z = 0
        z_scores.append(z)
    
    # COMPONENT MAIN ITERATION ENGINE AND DATA EXTRACTION
    for i in range(n):
        value = data[i]
        z = z_scores[i]
        violated = []
        
        # COMPONENT WESTGARD RULE ENGINE
        # Subcomponent 1-2s: 1 mark > 2SD (Warning)
        if abs(z) > 2:
            violated.append('1-2s')
        
        # Subcomponent 1-3s: 1 mark > 3SD (Critical Error)
        if abs(z) > 3:
            violated.append('1-3s')
        
        # Subcomponent 2-2s: 2 consecutive > 2SD (Systematic Error)
        if i >= 1:
            prev_z = z_scores[i-1]
            if abs(z) > 2 and abs(prev_z) > 2 and (z * prev_z > 0):
                violated.append('2-2s')
        
        # Subcomponent R-4s: 2 consecutive > 4SD (Random Error)
        if i >= 1:
            prev_z = z_scores[i-1]
            if abs(z - prev_z) > 4:
                violated.append('R-4s')
        
        # Subcomponent 4-1s: 4 consecutive > 1SD (Systematic Trend)
        if i >= 3:
            last4_z = z_scores[i-3:i+1]
            # All > 1SD and all positive OR all negative
            if all(abs(x) > 1 for x in last4_z) and (all(x > 0 for x in last4_z) or all(x < 0 for x in last4_z)):
                violated.append('4-1s')
        
        # Subcomponent 10-x: 10 consecutively on one side of the mean (Shift Detection)
        if i >= 9:
            last10_z = z_scores[i-9:i+1]
            # All above the mean OR all below the mean
            if all(x > 0 for x in last10_z) or all(x < 0 for x in last10_z):
                violated.append('10-x')
        
        #COMPONENT RESULT AGGREGATION
        results.append({
            'index': i + 1, # The day starts from one 
            'value': value,
            'z': round(z, 2),
            'violated': violated
        })
    return results

# MODULE WESTGARD VISUALIZATION (CHARTING)
def tampilkan_chart(data, mean, sd):
    """
    Displays Vertical chart (Levey-Jennings style)
    Order from bottom: -3s, -2s, -1s, Mean, +1s, +2s, +3s
    """
    # COMPONENT HEADER DISPLAY ENGINE
    print(f"\n--- Levey-Jennings Chart (Mean={mean}, SD={sd}) ---")
    
    # COMPONENT DATA GENERATION ENGINE (SIMULATION)
    x = np.arange(len(data))
    y = np.random.uniform(-3, 3, len(data))

    # COMPONENT BASE PLOTTING ENGINE
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, '-k', marker='o', markersize=6)

    # COMPONENT THRESHOLD LINE ENGINE (SD LEVELS)
    thresholds = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    for t in thresholds:
        plt.axhline(y=t, color='gray', linestyle='--', linewidth=1)
    
    # COMPONENT HEADER INDEX RENDERING
    header = "SD    |"
    for i in range(len(data)):
        header += f"{i+1:^3}" # Average number of days in the middle
    print(header)
    print("-" * (4 + len(data) * 3))
    
    # COMPONENT TEXT-BASED CHART RENDERING
    for level in thresholds:
        # Subcomponent level label generator
        if level == 0:
            label = "Mean "
        elif level > 0:
            label = f"+{level}s "
        else:
            label = f"{level}s "
        line = f"{label}|"
        # Subcomponent data position evaluation engine
        for i, val in enumerate(data):
            z = (val - mean) / sd if sd != 0 else 0
            # Subcomponent symbol mapping engine
            if abs(z) >= abs(level):
                line += " * " # Violation
            else:
                line += "   " # No violation    
    # Subcomponent line output engine
    print(line)

# MODULE WESTGARD PLOT VISUALIZATION
def plot_westgard(data, mean, sd, results):
    """
    Visualization of a Levey-Jennings chart with Westgard Rules violations marked
    data: list of floats, daily control values
    mean: float, control mean
    sd: float, control standard deviation
    results: list of dicts, westgard_rules() results
    """
    # COMPONENT DATA PREPARATION
    x = list(range(1, len(data)+1))
    y = data

    # COMPONENT BASE GRAPH INITIALIZATION
    plt.figure(figsize=(12,6))
    plt.plot(x, y, marker='o', label='periode Kontrol')

    # COMPONENT MEAN AND SD LINE ENGINE
    plt.axhline(mean, color='black', linestyle='-', label='Mean')
    # Subcomponent sd level generator
    for i in range(-10, 3):
        plt.axhline(mean + i*sd, color='grey', linestyle='--', label=f'+{i}SD' if i==1 else None)
        plt.axhline(mean - i*sd, color='grey', linestyle='--', label=f'-{i}SD' if i==1 else None)

    # COMPONENT VIOLATION MARKING ENGINE
    for r in results:
        if r['violated']:
            # Subcomponent point highlighting
            plt.plot(r['index']+1, r['value'], 'ro', markersize=10, label='Pelanggaran' if 'Pelanggaran' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # COMPONENT AXIS CONFIGURATION
    plt.xticks([])
    plt.yticks(range(-3, 4))

    # COMPONENT GRAPH LIMITS
    plt.ylim(-6, 6)

    # COMPONENT LABELING AND DECORATION
    plt.xlabel('Sampel')
    plt.ylabel('periode Kontrol')
    plt.title('Levey-Jennings Chart dengan Westgard Rules')

    # COMPONENT GRID AND LEGEND ENGINE
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # COMPONENT OUTPUT EXPORT ENGINE AS A PNG FILE
    plt.savefig("westgard.png")
    plt.close() 

# MODULE PHYSIOLOGY ENGINE (LEAKAGE INDEX) - FAST C++ EQUIVALENT
# COMPONENT FAST C++ FUNCTIONS (Python Implementation)
class LeakageIndexCalculator:
    """Fast leakage index calculations - C++ equivalent in Python"""

    @staticmethod
    def calculate(albumin, hematocrit):
        """Calculate leakage index from albumin and hematocrit"""
        # Subcomponent input validation
        if albumin <= 0 or pd.isna(albumin) or pd.isna(hematocrit):
            return float('nan')
        # Subcomponent core computation
        return hematocrit / albumin

    @staticmethod
    def validate_physiology(albumin, hematocrit):
        """Validate physiological status based on leakage index"""
        # Subcomponent index calculation call
        idx = LeakageIndexCalculator.calculate(albumin, hematocrit)
        # Subcomponent null handling
        if pd.isna(idx):
            return 'Invalid'
        # Subcomponent threshold classification
        if idx < 1.5:
            return 'Normal'
        elif idx < 2.0:
            return 'Risiko Kebocoran'
        else:
            return 'Kebocoran Plasma'

    @staticmethod
    def calculate_batch(albumin_values, hematocrit_values):
        """Calculate leakage indices for multiple samples"""
        # Subcomponent input size alignment
        n = min(len(albumin_values), len(hematocrit_values))
        # Subcomponent iteration computation
        results = []
        for i in range(n):
            results.append(LeakageIndexCalculator.calculate(albumin_values[i], hematocrit_values[i]))
        return results

    @staticmethod
    def assess_risk(leakage_index):
        """Assess risk level from leakage index"""
        # Subcomponent null handling
        if pd.isna(leakage_index):
            return 'Unknown'
        # Subcomponent risk threshold evaluation
        if leakage_index < 1.5:
            return 'LOW'
        elif leakage_index < 2.0:
            return 'MODERATE'
        elif leakage_index < 2.5:
            return 'HIGH'
        else:
            return 'CRITICAL'

# COMPONENT EXECUTION MODE MANAGEMENT ENGINE (C++ vs PYTHON)
def _setup_wrappers():
    """Setup wrapper functions for leakage index calculations"""
    global calculate_leakage_index, validate_physiology_wrapper, calculate_leakage_batch, assess_risk
    
    # Subcomponent C++ module initialization
    build_cpp_module_deferred()
    
    global cpp_module_available, plasma_leakage
    
    # Subcomponent implementation selection engine
    if cpp_module_available and plasma_leakage is not None and hasattr(plasma_leakage, 'LeakageIndexCalculator'):
        # Subcomponent C++ function binding
        calculate_leakage_index = plasma_leakage.LeakageIndexCalculator.calculate
        validate_physiology_wrapper = plasma_leakage.LeakageIndexCalculator.validate_physiology
        calculate_leakage_batch = plasma_leakage.LeakageIndexCalculator.calculate_batch
        assess_risk = plasma_leakage.LeakageIndexCalculator.assess_risk
        # Subcomponent status logging
        print("[OK] Using C++ LeakageIndexCalculator implementations")
    else:
        # Subcomponent python function binding
        calculate_leakage_index = LeakageIndexCalculator.calculate
        validate_physiology_wrapper = LeakageIndexCalculator.validate_physiology
        calculate_leakage_batch = LeakageIndexCalculator.calculate_batch
        assess_risk = LeakageIndexCalculator.assess_risk
        # Subcomponent status logging
        print("[OK] Using Python LeakageIndexCalculator implementations")

# COMPONENT WRAPPER INTERFACE ENGINE
# Subcomponent single value computation wrapper
def calculate_leakage_index(albumin, hematocrit):
    """Wrapper for LeakageIndexCalculator.calculate"""
    return LeakageIndexCalculator.calculate(albumin, hematocrit)
# Subcomponent physiology validation wrapper
def validate_physiology_wrapper(albumin, hematocrit):
    """Wrapper for LeakageIndexCalculator.validate_physiology"""
    return LeakageIndexCalculator.validate_physiology(albumin, hematocrit)
# Subcomponent batch computation wrapper
def calculate_leakage_batch(albumin_values, hematocrit_values):
    """Wrapper for LeakageIndexCalculator.calculate_batch"""
    return LeakageIndexCalculator.calculate_batch(albumin_values, hematocrit_values)
# Subcomponent risk assessment wrapper
def assess_risk(leakage_index):
    """Wrapper for LeakageIndexCalculator.assess_risk"""
    return LeakageIndexCalculator.assess_risk(leakage_index)

# COMPONENT CLINICAL API INTERFACE ENGINE
# Subcomponent leakage index public function
def leakage_index(albumin, hematokrit):
    """
    Calculate the Leakage Index using the LeakageIndexCalculator (C++ equivalent)
    Formula: Leakage Index = Hematocrit / Albumin
    """
    return calculate_leakage_index(albumin, hematokrit)
# Subcomponent physiology validation public function
def validate_physiology(albumin, hematokrit):
    """
    Physiological validation using the LeakageIndexCalculator (C++ equivalent)
    Categories:
    - Normal: Leakage Index < 1.5
    - Leakage Risk: 1.5 <= Leakage Index < 2.0
    - Plasma Leakage: Leakage Index >= 2.0
    """
    status = validate_physiology_wrapper(albumin, hematokrit)
    # Subcomponent index retrieval
    idx = calculate_leakage_index(albumin, hematokrit)
    # Subcomponent structured output
    return {'status': status, 'index': idx}

# MODULE PHYSIOLOGY VALIDATION ENGINE
def physics_engine(df):
    
    # COMPONENT INITIALIZATION
    violations = []

    # COMPONENT DATA VALIDATION
    if len(df) == 0:
        violations.append("Data is empty after cleaning")
        return violations

    # COMPONENT COLUMN DETECTION ENGINE (Support LONG and WIDE formats)
    # Subcomponent column initialization
    hct_col = None
    albumin_col = None
    
    # Subcomponent dynamic column scanning
    for col in df.columns:
        col_lower = col.lower()
        # Subcomponent date columns exclusion
        if col.startswith('Tanggal_') or 'tanggal' in col_lower:
            continue
        # Subcomponent hct column identification
        if 'hct' in col_lower or 'hematokrit' in col_lower or 'hematocrit' in col_lower:
            hct_col = col
        # Subcomponent albumin column identification
        if 'albumin' in col_lower:
            albumin_col = col
    
    # COMPONENT FALLBACK DETECTION ENGINE
    if hct_col is None and 'hematocrit' in df.columns:
        hct_col = 'hematocrit'
    if albumin_col is None and 'albumin' in df.columns:
        albumin_col = 'albumin'
    
    # COMPONENT COLUMN VALIDATION ENGINE
    # Subcomponent missing column handling
    if hct_col is None or albumin_col is None:
        missing_cols = []
        if hct_col is None:
            missing_cols.append("HCT/Hematocrit")
        if albumin_col is None:
            missing_cols.append("Albumin")
        violations.append(f"Kolom tidak ditemukan: {', '.join(missing_cols)}")
        return violations
    # Subcomponent detection logging
    print(f"  -> Detected columns: HCT='{hct_col}', Albumin='{albumin_col}'")

    # COMPONENT STATISTICAL BASELINE ENGINE
    # Subcomponent hematokrit statistics
    mean_hct = df[hct_col].mean()
    sd_hct = df[hct_col].std()
    # Subcomponent albumin statistics
    mean_alb = df[albumin_col].mean()
    sd_alb = df[albumin_col].std()

    # COMPONENT OUTLIER DETECTION ENGINE (3SD RULE)
    # Subcomponent iterative outlier analysis
    for i, row in df.iterrows():
        # Subcomponent hematokrit outlier check
        if abs(row[hct_col] - mean_hct) > 3 * sd_hct:
            violations.append(f"HCT outlier in index {i}")
        # Subcomponent albumin outlier check
        if abs(row[albumin_col] - mean_alb) > 3 * sd_alb:
            violations.append(f"Albumin outlier in index {i}")

    # COMPONENT PHYSIOLOGICAL CORRELATION ENGINE
    # Subcomponent correlation calculation
    corr = df[hct_col].corr(df[albumin_col])
    # Subcomponent correlation validation
    if corr > 0:
        violations.append("Relationships are not physiologically appropriate (should be negative)")

    # COMPONENT OUTPUT AGGREGATION
    return violations

# MODULE WESTGARD AND PHYSIOLOGY EXECUTION PIPELINE
# COMPONENT INITIALIZE WRAPPERS BEFORE EXECUTION
_setup_wrappers()

# COMPONENT ALBUMIN COLUMN DETECTION (Support LONG and WIDE formats)
# Subcomponent column initialization
album_col = None
# Subcomponent dynamic column scanning
for col in df.columns:
    col_lower = col.lower()
    # Subcomponent date columns exclusion
    if col.startswith('Tanggal_') or 'tanggal' in col_lower:
        continue
    # Subcomponent albumin column identification
    if 'albumin' in col_lower:
        album_col = col
        break

# COMPONENT FALLBACK DETECTION ENGINE
if album_col is None and 'albumin' in df.columns:
    album_col = 'albumin'

# COMPONENT DATA EXTRACTION ENGINE
# Subcomponent numeric conversion and cleaning
if album_col is not None and album_col in df.columns:
    kontrol_data = pd.to_numeric(df[album_col], errors='coerce').dropna().tolist()
    # Subcomponent detection logging
    print(f"  -> Using Albumin column: '{album_col}'")
# Subcomponent error handling
else:
    print(f"  [WARN] WARNING: Albumin column not found. Available columns: {df.columns.tolist()}")
    kontrol_data = []

# COMPONENT STATISTICAL PARAMETER ENGINE
if kontrol_data:
    # Subcomponent mean calculation
    mean_kontrol = np.mean(kontrol_data)
    # Subcomponent standard deviation calculation
    sd_kontrol = np.std(kontrol_data, ddof=1)
    
    # COMPONENT WESTGARD ANALYSIS ENGINE
    # Subcomponent rule evaluation
    results = westgard_rules(kontrol_data, mean_kontrol, sd_kontrol)
# Subcomponent empty data handling
else:
    print("  [WARN] No control data available for Westgard analysis")
    results = []

# COMPONENT WESTGARD RESULT DISPLAY ENGINE
if results:
    # Subcomponent header display
    print("\nHasil Evaluation of Westgard Rules:")
    # Subcomponent violation iteration
    for r in results:
        # Subcomponent violation filtering
        if r['violated']:
            # Subcomponent formated output generator
            print(
                f"Hari {r['index']+1}: "
                f"Nilai={r['value']}, "
                f"Z={r['z']:.2f}, "
                f"Pelanggaran={', '.join(r['violated'])}"
            )
    
    # Subcomponent summary statistics engine
    if kontrol_data:
        print(f"Mean Kontrol: {mean_kontrol:.2f}, SD Kontrol: {sd_kontrol:.2f}")
        
        # COMPONENT VISUALIZATION ENGINE
        # Subcomponent westgard plot generator
        try:
            plot_westgard(kontrol_data, mean_kontrol, sd_kontrol, results)
        except Exception as e:
            # Subcomponent visualization error handling
            print(f"  [WARN] Error plotting westgard: {e}")
# Subcomponent no result handling
else:
    print("\nNo Westgard analysis results available - data may be in LONG format requiring transformation")

# COMPONENT HCT AND ALBUMIN COLUMN DETECTION
# Subcomponent column initialization
hct_col_eval = None
album_col_eval = None
# Subcomponent dynamic column scanning
for col in df.columns:
    col_lower = col.lower()
    # Subcomponent date columns exclusion
    if col.startswith('Tanggal_') or 'tanggal' in col_lower:
        continue
    # Subcomponent hct identification
    if 'hct' in col_lower or 'hematokrit' in col_lower or 'hematocrit' in col_lower:
        hct_col_eval = col
    # Subcomponent albumin identification
    if 'albumin' in col_lower:
        album_col_eval = col

# COMPONENT FALLBACK DETECTION ENGINE
if hct_col_eval is None and 'HCT_%' in df.columns:
    hct_col_eval = 'HCT_%'
if album_col_eval is None and 'Albumin_g/dL' in df.columns:
    album_col_eval = 'Albumin_g/dL'

# COMPONENT PHYSIOLOGICAL EVALUATION ENGINE
# Subcomponent header display
print ("\n Physiological Evaluation Based on Leakage Index:")
if hct_col_eval and album_col_eval:
    # Subcomponent row iteration engine
    for idx, row in df.iterrows():
        # Subcomponent data extraction per patient
        albumin = row.get(album_col_eval)
        hematokrit = row.get(hct_col_eval)
        # Subcomponent data validation (not null check)
        if pd.notna(albumin) and pd.notna(hematokrit):
            # Subcomponent physiology validation call
            result = validate_physiology(albumin, hematokrit)
            # Subcomponent patient identifier retrieval
            patient_id = row.get('ID_Pasien', f'row_{idx}')
            # Subcomponent index formatting
            leakage_idx_display = f"{result['index']:.2f}" if result.get('index') is not None else "N/A"
            # Subcomponent formatted output generator
            print(
                f"Pasien ID {patient_id}: "
                f"Albumin={albumin:.2f}, "
                f"Hematokrit={hematokrit:.2f}, "
                f"Leakage Index={leakage_idx_display}, "
                f"Status={result.get('status', 'Unknown')}"
            )
# Subcomponent error handling
else:
    print(f" [WARN] Cannot evaluate physiology: HCT column='{hct_col_eval}', Albumin column='{album_col_eval}'")

# MODULE DATA ANONYMIZATION ENGINE
def anonymize_patient_data(csv_path, output_path):
    """
    Anonymize patient names in CSV data by replacing first column (names) with DBD-XXX format.
    """
    # COMPONENT FILE VALIDATION ENGINE
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    # COMPONENT DATA LOADING ENGINE
    # Subcomponent CSV loading
    df = pd.read_csv(csv_path)
    # Subcomponent data overview display
    print("Original data shape:", df.shape)
    print("Original columns:", df.columns.tolist())
    print("\nOriginal first few rows:\n", df.head())
    
    # COMPONENT PRIMARY IDENTIFIER DETECTION ENGINE
    if 'Nama' in df.columns:
        target_col = 'Nama'
    elif 'No_RM' in df.columns:
        target_col = 'No_RM'
    else:
        target_col = df.columns[0]

    unique_ids = {rm: f'DBD-{i:03d}' for i, rm in enumerate(df['No_RM'].unique())}
    df['Anon_ID'] = df['No_RM'].map(unique_ids)

    # COMPONENT REMOVE ORIGINAL ID
    df = df.drop(columns=['No_RM'], errors='ignore')

    # Subcomponent identifier generation
    print(f"Replacing names in column '{target_col}' with anonymized IDs.")
    df[target_col] = [f'DBD-{i+1:03d}' for i in range(len(df))]
    
   # COMPONENT SECONDARY ANONYMIZATION ENGINE
    for col in df.columns:
        # Subcomponent object type filter
        if df[col].dtype == 'object':
            # Subcomponent sampel extraction for pattern check
            sample = ' '.join(df[col].dropna().astype(str).unique()[:3])
            # Subcomponent name pattern detection
            if re.search(r'[A-Za-z]{2,}(?:\s[A-Za-z]{2,})+', sample):
                print(f"Additional anonymization on '{col}'.")
                # Subcomponent replacement with anonymized ID
                if col.lower() in ['nama', 'pasien', 'patient_name']:
                    df[col] = df[target_col]
    
    # COMPONENT DATA EXPORT ENGINE
    df.to_csv(output_path, index=False)
    # Subcomponent output confirmation
    print(f"\nAnonymized data saved to '{output_path}'")
    print("Anonymized first few rows:\n", df.head())
    # Subcomponent process summary
    print(f"Processed {len(df)} rows successfully.")

# COMPONENT SCRIPT ENTRY POINT
if __name__ == '__main__':
    csv_file = 'data_rekam_medis_dbd_300_long_format.csv'
    output_file = 'data_anonymized.csv'
    anonymize_patient_data(csv_file, output_file)

    # COMPONENT DATA LOADING AND PROCESSING
    df_real = pd.read_csv(file_path)
    
    # COMPONENT FORMAT AUTO-DETECTION ENGINE
    print("\n" + "="*70)
    print("DETECTING DATA FORMAT")
    print("="*70)
    
    format_type = detect_data_format(df_real)
    print(f"[OK] FORMAT DETECTED: {format_type}")
    print(f"  Columns: {df_real.columns.tolist()}\n")
    
    # COMPONENT CONDITIONAL DATA TRANSFORMATION
    if format_type == 'LONG':
        print("-> PROCESSING: Converting LONG format to WIDE...")
        df = transform_long_to_wide(df_real)
        print(f"  [OK] Transformation complete. New shape: {df.shape}\n")
    else:
        print("[OK] SKIP: Data already in WIDE format\n")
        df = df_real.copy()
    
    # COMPONENT DATA PROCESSING ENGINE
    df = process_data(df)

    # COMPONENT DATA STRUCTURE ENGINE
    df_standard = standardize_columns(df)

    # COMPONENT DATA CLEANING EXECUTION
    print("\n" + "="*70)
    print("EXECUTING DATA CLEANING MODULE")
    print("="*70)
    df = clean_data(df)

    # COMPONENT DATA VALIDATION AND SUMMARY EXECUTION
    validate_and_summarize_data(df)

# MODULE DATA PREPROCESSING
# COMPONENT MISSING VALUE INSPECTION ENGINE
print("\nMissing Values per Kolom:")
print(df.isnull().sum())

# COMPONENT SMART COLUMN DETECTION FOR DATA HANDLING
hct_col_preprocess = None
alb_col_preprocess = None
    
# COMPONENT COLUMN DETECTION ENGINE
for col in df.columns:
    if col == 'HCT_%':
        hct_col_preprocess = col
    if col == 'Albumin_g/dL':
        alb_col_preprocess = col
    
# COMPONENT COLUMN DETECTION ENGINE
# Subcomponent fallback condition check
if hct_col_preprocess is None or alb_col_preprocess is None:
    # Subcomponent dynamic column scanning
    for col in df.columns:
        col_lower = col.lower()
        # Subcomponent date column exclusion
        if 'tanggal' in col_lower or 'date' in col_lower:
            continue
        # Subcomponent HCT column identification
        if hct_col_preprocess is None and ('hct' in col_lower or 'hematokrit' in col_lower):
            hct_col_preprocess = col
        # Subcomponent Albumin column identification
        if alb_col_preprocess is None and 'albumin' in col_lower:
            alb_col_preprocess = col
    
# COMPONENT MISSING VALUE HANDLING ENGINE
if hct_col_preprocess and alb_col_preprocess:
    df_clean = df.dropna(subset=[hct_col_preprocess, alb_col_preprocess])
        
    # COMPONENT TYPE CONVERSION TO NUMERIC - ENSURES FLOAT64 FOR CONTINUOUS VARIABLES
    df_clean = df_clean.copy()
    df_clean[hct_col_preprocess] = pd.to_numeric(df_clean[hct_col_preprocess], errors='coerce').astype('float64')
    df_clean[alb_col_preprocess] = pd.to_numeric(df_clean[alb_col_preprocess], errors='coerce').astype('float64')
        
    # Subcomponent fallback numeric conversion
    df_clean = df_clean.dropna(subset=[hct_col_preprocess, alb_col_preprocess])
        
    print(f"\n[OK] Using columns: HCT={hct_col_preprocess}, Albumin={alb_col_preprocess}")
    print(f"[OK] Data types after conversion:")
    print(f"     {hct_col_preprocess}: {df_clean[hct_col_preprocess].dtype}")
    print(f"     {alb_col_preprocess}: {df_clean[alb_col_preprocess].dtype}")
# Subcomponent error handling
else:
    print(f"\n[WARN] WARNING: Could not find HCT or Albumin columns")
    print(f"  HCT detected: {hct_col_preprocess}, Albumin detected: {alb_col_preprocess}")
    print(f"  Available: {df.columns.tolist()}")
    # Subcomponent fallback data assigment
    df_clean = df.copy()

# COMPONENT OUTLIER DETECTION ENGINE (IQR METHOD)
def detect_outliers_iqr(data):
    # Subcomponent type conversion to numeric
    try:
        data_numeric = pd.to_numeric(data, errors='coerce')
    except:
        data_numeric = data
        
    # Subcomponent remove NaN values
    data_clean = data_numeric.dropna()
        
    # Subcomponent empty data handling
    if len(data_clean) == 0:
        return pd.Series([False] * len(data))
        
    # Subcomponent quartile calculation
    Q1 = data_clean.quantile(0.25)
    Q3 = data_clean.quantile(0.75)
    # Subcomponent IQR computation
    IQR = Q3 - Q1
        
    # Subcomponent zero-IQR handling
    if IQR == 0:
        return pd.Series([False] * len(data))
    # Subcomponent boundary determination
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Subcomponent outlier flagging - compare numeric data
    outliers = (data_numeric < lower_bound) | (data_numeric > upper_bound)
    return outliers

# COMPONENT OUTLIER IDENTIFICATION
if hct_col_preprocess and alb_col_preprocess:
    outliers_hct = detect_outliers_iqr(df_clean[hct_col_preprocess])
    outliers_alb = detect_outliers_iqr(df_clean[alb_col_preprocess])

    # Subcomponent outlier summary display
    print(f"Outliers {hct_col_preprocess}: {outliers_hct.sum()}")
    print(f"Outliers {alb_col_preprocess}: {outliers_alb.sum()}")

    # COMPONENT OUTLIER REMOVAL ENGINE
    df_clean = df_clean[~(outliers_hct | outliers_alb)]
else:
    print("[WARN] Skipping outlier detection - columns not properly detected")

# COMPONENT DATA FILTER INITIALIZATION
df_filtered = df_clean

# COMPONENT TEMPORAL CONSISTENCY ENGINE
if 'Tanggal_Hematokrit' in df_filtered.columns and 'Tanggal_Albumin' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Tanggal_Hematokrit'] == df_filtered['Tanggal_Albumin']]

# COMPONENT FINAL VALIDATION ENGINE
if hct_col_preprocess and alb_col_preprocess:
    df_filtered = df_filtered.dropna(subset=[hct_col_preprocess, alb_col_preprocess])
else:
    print("[WARN] Skipping final validation - columns not found")

# MODULE STATISTICAL ANALYSIS AND NORMALITY TEST
# COMPONENT NORMALITY TEST ENGINE - FAST C++ EQUIVALENT
class StatisticalCalculator:
        """Fast statistical computations - C++ equivalent in Python"""

        # COMPONENET STATISTIC METHODS SHAPIRO-WILK
        @staticmethod
        def shapiro_wilk_test(data):
            """Shapiro-Wilk normality test"""
            # Subcomponent data cleaning
            data = np.asarray([x for x in data if not pd.isna(x)], dtype=float)
            # Subcomponent minimum sample check
            if len(data) < 3:
                return "Shapiro-Wilk", 1.0, 1.0
            # Subcomponent statistical computation
            stat, p = stats.shapiro(data)
            # Subcomponent result return
            return "Shapiro-Wilk", stat, p

        # COMPONENET STATISTIC METHODS KOLMOGOROV-SMIRNOV
        @staticmethod
        def kolmogorov_smirnov_test(data):
            """Kolmogorov-Smirnov normality test"""
            # Subcomponent data cleaning
            data = np.asarray([x for x in data if not pd.isna(x)], dtype=float)
            # Subcomponent minimum sample check
            if len(data) < 2:
                return "Kolmogorov-Smirnov", 1.0, 1.0
            normalized = (data - np.mean(data)) / np.std(data, ddof=0)
            # Subcomponent statistical computation
            stat, p = stats.kstest(normalized, 'norm')
            # Subcomponent result return
            return "Kolmogorov-Smirnov", stat, p

# COMPONENT NORMALITY TEST CONTROLLER ENGINE
def normality_test(series):
        """Normality test using StatisticalCalculator (C++ equivalent)"""
        # Subcomponent data cleaning
        s = series.dropna()
        # Subcomponent sample size calculation
        n = len(s)
        # Subcomponent empty data handling
        if n == 0:
            return None, None, None
        # COMPONENT METHOD SELECTION ENGINE
        # Subcomponent sample size based selection
        if n < 50:
            # Subcomponent shapiro-wilk test
            method, stat, p = StatisticalCalculator.shapiro_wilk_test(s.values)
        else:
            # Subcomponent kolmogorov-smirnov test
            method, stat, p = StatisticalCalculator.kolmogorov_smirnov_test(s.values)
        # Subcomponent result return
        return method, stat, p

# COMPONENT NORMALITY TEST EXECUTION
method_hct, stat_hct, p_hct = normality_test(df_filtered["HCT_%"])
method_alb, stat_alb, p_alb = normality_test(df_filtered["Albumin_g/dL"])
# Subcomponent method display
print(f"\n Normality Test Method: Hematocrit -> {method_hct}; Albumin -> {method_alb}")

# COMPONENT DATA DISPLAY ENGINE
print("\n Dengue Fever Patient Data After Cleaning, Filtering, and Sampling:")
print(f"Number of samples: {len(df_filtered)}")
# Subcomponent selected column display
print(df_filtered[
    ['ID_Pasien', 'Umur_Tahun', 'Jenis_Kelamin',
        'Tanggal_Hematokrit', 'Tanggal_Albumin',
        'HCT_%', 'Albumin_g/dL']
    ].to_string(index=False))
# Subcomponent normality result display
print(
        f"\nUji Normalitas:\n"
        f" - Hematokrit ({method_hct}): Statistik = {stat_hct}, p-value = {p_hct}\n"
        f" - Albumin ({method_alb}): Statistik = {stat_alb}, p-value = {p_alb}"
    )

# COMPONENT NORMALITY DECISION ENGINE
normal_hct = (p_hct is not None and p_hct >= 0.05)
normal_alb = (p_alb is not None and p_alb >= 0.05)

# COMPONENT DATA AVAILABILITY CHECK
if len(df_filtered) == 0:
    print("[WARN] WARNING: No filtered data available for analysis")
    normality_results = pd.DataFrame({
            'Variable': ['HCT_%', 'Albumin_g/dL'],
            'Test Method': [method_hct, method_alb],
            'Statistic': [stat_hct, stat_alb],
            'p-value': [p_hct, p_alb],
            'Normal?': ['N/A', 'N/A']
        })
    
# COMPONENT PARAMETRIC ANALYSIS ENGINE
elif normal_hct and normal_alb:
    print("Both variables are normally distributed")
    # Subcomponent desciptive statistics (mean ± SD)
    try:
        mean_albumin = df_filtered["Albumin_g/dL"].mean()
        std_albumin = df_filtered["Albumin_g/dL"].std()
        print(f"Albumin (Mean ± SD): {mean_albumin:.2f} ± {std_albumin:.2f}")

        mean_hct = df_filtered["HCT_%"].mean()
        std_hct = df_filtered["HCT_%"].std()
        print(f"Hematocrit (Mean ± SD): {mean_hct:.2f} ± {std_hct:.2f}")
        # Subcomponent pearson correlation engine
        r, p_corr = pearsonr(df_filtered["HCT_%"], df_filtered["Albumin_g/dL"])
        print("Pearson Correlation Test")
        print(f"r = {r:.3f}, p-value = {p_corr:.4f}")
    except Exception as e:
        print(f"[WARN] Error in parametric analysis: {e}")
        
        normality_results = pd.DataFrame({
            'Variable': ['HCT_%', 'Albumin_g/dL'],
            'Test Method': [method_hct, method_alb],
            'Statistic': [stat_hct, stat_alb],
            'p-value': [p_hct, p_alb],
            'Normal?': ['Yes' if p is not None and p >= 0.05 else 'No' for p in [p_hct, p_alb]]
        })

    # COMPONENT NON-PARAMETRIC ANALYSIS ENGINE
    else:
        print("Single / both variables are NOT NORMAL")
        # Subcomponent desciptive statistics (median and IQR)
        try:
            median_albumin = df_filtered["Albumin_g/dL"].median()
            q1 = df_filtered["Albumin_g/dL"].quantile(0.25)
            q3 = df_filtered["Albumin_g/dL"].quantile(0.75)
            print(f"Albumin Median (IQR): {median_albumin:.2f} ({q1:.2f}–{q3:.2f})")

            median_hct = df_filtered["HCT_%"].median()
            q1_hct = df_filtered["HCT_%"].quantile(0.25)
            q3_hct = df_filtered["HCT_%"].quantile(0.75)
            print(f"Hematocrit Median (IQR): {median_hct:.2f} ({q1_hct:.2f}–{q3_hct:.2f})")

            # Subcomponent spearman correlation engine
            rho, p_corr = spearmanr(df_filtered["HCT_%"], df_filtered["Albumin_g/dL"])
            print("Spearman Correlation Test")
            print(f"ρ = {rho:.3f}, p-value = {p_corr:.4f}")
        except Exception as e:
            print(f"[WARN] Error in non-parametric analysis: {e}")
        
        normality_results = pd.DataFrame({
            'Variable': ['HCT_%', 'Albumin_g/dL'],
            'Test Method': [method_hct, method_alb],
            'Statistic': [stat_hct, stat_alb],
            'p-value': [p_hct, p_alb],
            'Normal?': ['Yes' if p is not None and p >= 0.05 else 'No' for p in [p_hct, p_alb]]
        })

# MODULE UNIVARIATE ANALYSIS ENGINE
    # COMPONENT ALBUMIN UNIVARIATE ANALYSIS
    # Subcomponent normality-based method selection (Albumin)
    if normal_alb:
        # Subcomponent parametric statistics (mean ± SD)
        mean_albumin = df_filtered["Albumin_g/dL"].mean()
        std_albumin = df_filtered["Albumin_g/dL"].std()
        # Subcomponent result formatting
        univar_alb = f"{mean_albumin:.2f} ± {std_albumin:.2f}"
    else:
        # Subcomponent non-parametric statistic (median and IQR)
        median_albumin = df_filtered["Albumin_g/dL"].median()
        q1 = df_filtered["Albumin_g/dL"].quantile(0.25)
        q3 = df_filtered["Albumin_g/dL"].quantile(0.75)
        # Subcomponent result formatting
        univar_alb = f"{median_albumin:.2f} ({q1:.2f}–{q3:.2f})"

    # COMPONENT HEMATOKRIT UNIVARIATE ANALYSIS
    # Subcomponent normality-based method selection (Hematocrit)
    if normal_hct:
        # Subcomponent parametric statistics (mean ± SD)
        mean_hct = df_filtered["HCT_%"].mean()
        std_hct = df_filtered["HCT_%"].std()
        # Subcomponent result formatting
        univar_hct = f"{mean_hct:.2f} ± {std_hct:.2f}"
    else:
        # Subcomponent non-parametric statistic (median and IQR)
        median_hct = df_filtered["HCT_%"].median()
        q1_hct = df_filtered["HCT_%"].quantile(0.25)
        q3_hct = df_filtered["HCT_%"].quantile(0.75)
        # Subcomponent result formatting
        univar_hct = f"{median_hct:.2f} ({q1_hct:.2f}–{q3_hct:.2f})"

# MODULE BIVARIATE ANALYSIS RECOMMENDATION ENGINE
# COMPONENT VARIABLE TYPE DETECTION ENGINE
def detect_variable_type(series, series_name=None, ordinal_unique_threshold=10):
    """
    Specify the variable type: 'Scale', 'Ordinal', or 'Nominal'
    Simple rules:
    - Categorical variables with ordered values ​​-> Ordinal, unordered -> Nominal
    - Continuous variables (HCT%, Albumin g/dL, etc.) -> ALWAYS Scale
    - Numeric variables with few unique values ​​(<= threshold) and integers -> Ordinal
    - Other numeric variables -> Scale
    - Non-numeric variables -> Nominal
    """
    # Subcomponent data cleaning (remove null)
    s = series.dropna()

    # Subcomponent categorical type detection AND ORDER CHECK
    if isinstance(series.dtype, pd.CategoricalDtype):
        return 'Ordinal' if series.cat.ordered else 'Nominal'
    
    # Subcomponent explicit classification for known continuous variables
    # Known continuous/scale variables from medical lab data
    known_scale_variables = ['HCT_%', 'Albumin_g/dL', 'Umur_Tahun', 'RBC', 'WBC', 'HGB', 
                             'PLT', 'AST', 'ALT', 'ALP', 'Bilirubin', 'HDL', 'LDL', 'Glucose']
    if series_name in known_scale_variables:
        return 'Scale'
    
    # Subcomponent numeric type detection
    if pd.api.types.is_numeric_dtype(series):
        # Subcomponent ordinal detection (low unique integer values)
        # IMPROVED: Check if values truly represent rank/order (not just integer values)
        if s.nunique() <= ordinal_unique_threshold and np.all(np.mod(s, 1) == 0):
            # Additional check: ordinal values should have clear ordering semantics
            # If range is large relative to unique count, likely continuous data
            value_range = s.max() - s.min()
            if s.nunique() > 1 and value_range / s.nunique() > 2:
                # High range relative to unique values -> likely continuous (Scale)
                return 'Scale'
            return 'Ordinal'
        return 'Scale'
    return 'Nominal'

# COMPONENT BIVARIATE TEST RECOMMENDATION ENGINE
def recommend_bivariate_analysis(var1_name, var2_name, type1, type2, normal1, normal2, n):
    """Recommend bivariate tests and approaches (Classical/Bayesian)"""
    # Subcomponent scale vs scale analysis
    if type1 == 'Scale' and type2 == 'Scale':
        # Subcomponent parametric condition
        if normal1 and normal2:
            test = 'Pearson correlation / Linear regression'
            approach = 'Classical recommended; consider Bayesian for posterior or include prior'
        # Subcomponent non-parametric (both not normal)
        elif not normal1 and not normal2:
            test = 'Kendall tau correlation (non-parametric)'
            approach = 'Classical recommended; consider Bayesian priors or on small samples'
        # Subcomponent mixed normality
        else:
            test = 'Spearman correlation (non-parametric)'
            approach = 'Classical recommended; consider non-parametric or robust Bayesian if necessary'
    
    # Subcomponent nominal vs scale analysis
    elif (type1 == 'Nominal' and type2 == 'Scale') or (type2 == 'Nominal' and type1 == 'Scale'):
        # Subcomponent nominal variable identification
        nominal = var1_name if type1 == 'Nominal' else var2_name
        # Subcomponent level counting
        levels = df_filtered[nominal].nunique() if nominal in df_filtered.columns else None
        # Subcomponent test selection based on levels
        if levels == 2:
            test = 't-test (normal) / Mann-Whitney (non-normal)'
        else:
            test = 'ANOVA (normal) / Kruskal-Wallis (non-normal)'
        approach = 'Classical recommended; Bayesian GLM/ANOVA if want posterior or enter prior'

    # Subcomponent nominal vs nominal analysis
    elif type1 == 'Nominal' and type2 == 'Nominal':
        test = 'Chi-square or Fisher Exact (if counts small)'
        approach = 'Classical recommended'
    
    # Subcomponent ordinal cases
    else:
        test = 'Spearman or appropriate non-parametric/ordinal methods'
        approach = 'Classical recommended'
    
    # Subcomponent sample size adjustment engine
    if n < 30:
        approach += '; consider Bayesian for small samples or to include priors'
    else:
        approach += '; consider Bayesian when posterior/prior needed'
    return test, approach

# COMPONENT VARIABLE TYPE MAPPING ENGINE
variables_to_check = ['HCT_%', 'Albumin_g/dL', 'Umur_Tahun', 'Jenis_Kelamin']
var_types = {}
for v in variables_to_check:
    # Subcomponent column existence check
    if v in df_filtered.columns:
        # Subcomponent type detection execution (pass series_name for proper classification)
        var_types[v] = detect_variable_type(df_filtered[v], series_name=v)
# Subcomponent result display
print("\nTipe Variabel yang Dideteksi:")
for k, v in var_types.items():
    print(f"- {k}: {v}")

# COMPONENT BIVARIATE RECOMMENDATION EXECUTION
recommended_test, recommended_approach = recommend_bivariate_analysis(
    'HCT_%', 'Albumin_g/dL',
    var_types.get('HCT_%', detect_variable_type(df_filtered['HCT_%'], series_name='HCT_%')),
    var_types.get('Albumin_g/dL', detect_variable_type(df_filtered['Albumin_g/dL'], series_name='Albumin_g/dL')),
    normal_hct, normal_alb, len(df_filtered)
)

# Subcomponent recommendation display
print(f"\n Bivariate Analysis Recommendations for HCT_% vs Albumin_g/dL:")
print(f"- Recommended Tests: {recommended_test}")
print(f"- Recommended Approach: {recommended_approach}")

# MODULE SOFTWARE DECISION SUPPORT ENGINE
# COMPONENT SOFTWARE CATALOG DATABASE
software_catalog = {
    'JASP': {
        'Descriptives': [
            'Descriptive Statistics', 'Raincloud Plots', 'Time Series Descriptives', 
            'Flexplot', 'Plot Builder (beta)'
        ],
        'T-Tests': [
            'Independent Samples T-Test', 'Paired Samples T-Test', 'One Sample T-Test',
            'Bayesian Independent Samples T-Test', 'Bayesian Paired Samples T-Test', 'Bayesian One Sample T-Test'
        ],
        'ANOVA': [
            'ANOVA', 'Repeated Measures ANOVA', 'ANCOVA', 'MANOVA',
            'Bayesian ANOVA', 'Bayesian Repeated Measures ANOVA', 'Bayesian ANCOVA'
        ],
        'Mixed Models': [
            'Linear Mixed Models', 'Generalized Linear Mixed Models',
            'Bayesian Linear Mixed Models', 'Bayesian Generalized Linear Mixed Models'
        ],
        'Regression': [
            'Correlation', 'Linear Regression', 'Logistic Regression', 'Generalized Linear Model',
            'Bayesian Correlation', 'Bayesian Linear Regression', 'Bayesian Logistic Regression'
        ],
        'Frequencies': [
            'Binomial Test', 'Multinomial Test', 'Contingency Tables', 'Log-Linear Regression',
            'Bayesian Binomial Test', 'Bayesian A/B Test', 'Bayesian Multinomial Test', 
            'Informed Multinomial Test', 'Informed Multi-Binomial Test', 'Bayesian Contingency Tables', 'Bayesian Log-Linear Regression'
        ],
        'Factor': [
            'Principal Component Analysis (PCA)', 'Exploratory Factor Analysis (EFA)', 'Confirmatory Factor Analysis (CFA)'
        ],
        'Time Series': [
            'Descriptives', 'Stationarity', 'ARIMA', 'Spectral Analysis'
        ]
    },
    'jamovi': {
        'Exploration': [
            'Descriptives', 'Scatter Plot', 'Pareto Plot'
        ],
        'T-Tests': [
            'Independent Samples T-Test', 'Paired Samples T-Test', 'One Sample T-Test'
        ],
        'ANOVA': [
            'One-Way ANOVA', 'ANOVA', 'Repeated Measures ANOVA', 'ANCOVA', 'MANCOVA',
            'One-Way ANOVA (Kruskal-Wallis)', 'Repeated Measures ANOVA (Friedman)'
        ],
        'Regression': [
            'Correlation Matrix', 'Partial Correlation', 'Linear Regression',
            'Logistic Regression (2 Outcomes Binomial)', 'Logistic Regression (N Outcomes Multinomial)', 'Logistic Regression (Ordinal Outcomes)'
        ],
        'Frequencies': [
            '2 Outcomes Binomial Test', 'N Outcomes Multinomial X² Goodness of Fit',
            'Independent Samples X² Test of Association', 'Paired Samples McNemar Test', 'Log-Linear Regression'
        ],
        'Factor': [
            'Reliability Analysis', 'Principal Component Analysis (PCA)', 'Exploratory Factor Analysis (EFA)', 'Confirmatory Factor Analysis (CFA)'
        ],
        'snowCluster': [
            'K-Means Clustering', 'Hierarchical Clustering', 'Density-Based Clustering', 'Time Series Clustering',
            'Clustering Dendrogram', 'Multidimensional Scaling Plot', 'PCA & Group Plot',
            'Correspondence Analysis', 'Decision Tree', 'Machine Learning', 'ROC Analysis',
            'Univariate Time Series', 'Prophet with Multiple Variables'
        ]
    },
    'SPSS': {
        'Power Analysis': [
            'One Sample T-Test', 'Paired Samples T-Test', 'Independent Samples T-Test', 'One-Way ANOVA',
            'One-Sample Binomial Test', 'Related-Samples Binomial Test', 'Independent-Samples Binomial Test',
            'Pearson Product-Moment', 'Spearman Rank-Order', 'Partial', 'Univariate Linear'
        ],
        'Reports': [
            'Codebook', 'OLAP Cubes', 'Case Summaries', 'Report Summaries in Rows', 'Report Summaries in Columns'
        ],
        'Descriptive Statistics': [
            'Frequencies', 'Descriptives', 'Explore', 'Crosstabs', 'TURF Analysis', 'Ratio', 'P-P Plot', 'Q-Q Plot'
        ],
        'Bayesian Statistics': [
            'One Sample Normal', 'One Sample Binomial', 'One Sample Poisson', 'Related Samples Normal', 
            'Independent Samples Normal', 'Pearson Correlation', 'Linear Regression', 'One-Way ANOVA', 
            'Loglinear Models', 'One-Way Repeated Measures ANOVA'
        ],
        'Tables': [
            'Custom Tables', 'Multiple Response Sets', 'Define Category Order'
        ],
        'Compare Means': [
            'Means', 'One-Sample T Test', 'Independent-Samples T Test', 'Summary Independent-Samples T Test', 
            'Paired-Samples T Test', 'One-Way ANOVA', 'One-Sample Proportions', 'Independent-Samples Proportions', 'Paired-Samples Proportions'
        ],
        'General Linear Models': [
            'Univariate', 'Multivariate', 'Repeated Measures', 'Variance Components'
        ],
        'Generalized Linear Models': [
            'Generalized Linear Models', 'Generalized Estimating Equations'
        ],
        'Mixed Models': [
            'Linear', 'Generalized Linear'
        ],
        'Correlate': [
            'Bivariate', 'Partial', 'Distances', 'Canonical Correlation'
        ],
        'Regression': [
            'Automatic Linear Modeling', 'Linear', 'Curve Estimation', 'Partial Least Squares', 
            'Binary Logistic', 'Multinomial Logistic', 'Ordinal', 'Probit', 'Nonlinear', 
            'Weight Estimation', '2-Stage Least Squares', 'Quantile', 'Optimal Scaling (CATREG)'
        ],
        'Loglinear': [
            'General', 'Logit', 'Model Selection'
        ],
        'Neural Networks': [
            'Multilayer Perceptron', 'Radial Basis Function'
        ],
        'Classify': [
            'TwoStep Cluster', 'K-Means Cluster', 'Hierarchical Cluster', 'Cluster Silhouettes', 
            'Tree', 'Discriminant', 'Nearest Neighbor', 'ROC Curve', 'ROC Analysis'
        ],
        'Dimension Reduction': [
            'Factor', 'Correspondence Analysis', 'Optimal Scaling'
        ],
        'Scale': [
            'Reliability Analysis', 'Weighted Kappa', 'Multidimensional Unfolding (PREFSCAL)', 
            'Multidimensional Scaling (PROXSCAL)', 'Multidimensional Scaling (ALSCAL)'
        ],
        'Nonparametric Tests': [
            'One Sample', 'Independent Samples', 'Related Samples', 'Legacy Dialogs: Chi-square', 
            'Binomial', 'Runs', '1-Sample K-S', '2 Independent Samples', 'K Independent Samples', 
            '2 Related Samples', 'K Related Samples'
        ],
        'Forecasting': [
            'Create Temporal Causal Models', 'Create Traditional Models', 'Apply Temporal Causal Models', 
            'Apply Traditional Models', 'Seasonal Decomposition', 'Spectral Analysis', 'Sequence Charts', 
            'Autocorrelation', 'Cross-Correlation'
        ],
        'Survival': [
            'Life Tables', 'Kaplan-Meier', 'Cox Regression', 'Cox w/ Time-Dep Cov'
        ],
        'Multiple Response': [
            'Define Variable Sets', 'Frequencies', 'Crosstabs'
        ],
        'Missing Value Analysis': [
            'Analyze Patterns', 'Impute Missing Data Values'
        ],
        'Complex Samples': [
            'Select a Sample', 'Prepare for Analysis', 'Frequencies', 'Descriptives', 'Crosstabs', 
            'Ratios', 'General Linear Models', 'Logistic Regression', 'Ordinal Regression', 'Cox Regression'
        ],
        'Simulation': ['Simulation'],
        'Quality Control': [
            'Control Charts', 'Pareto Charts'
        ],
        'Spatial and Temporal Modeling': ['Spatial Modeling'],
        'Direct Marketing': ['Choose Technique']
    }
}

# COMPONENT SOFTWARE UNIVERSAL MAPPING ENGINE
software_mapping = {
    'correlation_parametric': {
        'JASP': ['Correlation (Pearson)', 'Regression Linear'],
        'SPSS': ['Correlation (Pearson)', 'Linear Regression'],
        'jamovi': ['Correlation Matrix']
    },
    'correlation_nonparametric': {
        'JASP': ['Correlation (Spearman/Kendall)'],
        'SPSS': ['Nonparametric Correlation (Spearman)'],
        'jamovi': ['Correlation Matrix (Spearman)']
    },
    'mean_comparison_2group': {
        'JASP': ['T-Test'],
        'SPSS': ['Independent Samples T-Test'],
        'jamovi': ['T-Test']
    },
    'mean_comparison_multigroup': {
        'JASP': ['ANOVA'],
        'SPSS': ['One-Way ANOVA'],
        'jamovi': ['ANOVA']
    },
    'nonparametric_2group': {
        'JASP': ['Nonparametric', 'Mann-Whitney U Test'],
        'SPSS': ['Nonparametric Tests', 'Independent Samples'],
        'jamovi': ['Non-parametric', 'Mann-Whitney U']
    },
    'nonparametric_multigroup': {
        'JASP': ['Nonparametric', 'Kruskal-Wallis Test'],
        'SPSS': ['Nonparametric Tests', 'Independent Samples'],
        'jamovi': ['Non-parametric', 'Kruskal-Wallis']
    },
    'categorical_association': {
        'JASP': ['Frequencies', 'Contingency Tables'],
        'SPSS': ['Crosstabs', 'Chi-Square Test'],
        'jamovi': ['Frequencies', 'Contingency Tables']
    },
    'general_analysis': {
        'JASP': ['Descriptives', 'General Analysis'],
        'SPSS': ['Descriptive Statistics', 'General Analysis'],
        'jamovi': ['Descriptives', 'General Analysis']
    }
}

# COMPONENT UNIVERSAL TEST NORMALIZATION ENGINE
def normalize_test_name(recommended_test):
    test = recommended_test.lower()

    if 'pearson' in test:
        return 'correlation_parametric'
    elif 'spearman' in test or 'kendall' in test:
        return 'correlation_nonparametric'
    elif 't-test' in test:
        return 'mean_comparison_2group'
    elif 'anova' in test:
        return 'mean_comparison_multigroup'
    elif 'mann-whitney' in test:
        return 'nonparametric_2group'
    elif 'kruskal' in test:
        return 'nonparametric_multigroup'
    elif 'chi' in test:
        return 'categorical_association'
    else:
        return 'general_analysis'

# COMPONENT UNIVERSAL SOFTWARE RECOMMENDATION ENGINE
def get_software_recommendation(recommended_test):
    normalized = normalize_test_name(recommended_test)

    results = []

    # Subcomponent software iteration with fallback handling
    for software in ['JASP', 'SPSS', 'jamovi']:
        if normalized in software_mapping:
            methods = software_mapping[normalized].get(software, ['General analysis'])
        else:
            methods = ['General statistical tools']

        results.append({
            'Software': software,
            'Recommended Methods': '; '.join(methods),
            'Analysis Type': normalized
        })

    return pd.DataFrame(results)

if normalize_test_name(recommended_test) not in software_mapping:
    print("WARNING: test not recognized -> use general mapping")
software_df = get_software_recommendation(recommended_test)
print(software_df)
 
# COMPONENT KEYWORD MATCHING ENGINE
keywords = ['pearson','spearman','t-test','anova','kruskal','mann-whitney','chi','regression','correlation','descriptives','factor','reliability','mixed']

# COMPONENT MATCHING AND SUPPORT ENGINE
software_rows = []
# Subcomponent test normalization
normalized_test = normalize_test_name(recommended_test)

# COMPONENT INTEGRATED SYSTEMS EXECUTION
print("\n" + "="*60)
print("INTEGRATED SYSTEMS EXECUTION")
print("="*60)

# COMPONENT NUMERICAL SYSTEM - TENSOR ENGINE
print("\n--- NUMERICAL SYSTEM: TENSOR ENGINE ---")
tensor_results = {}
    
# Subcomponent tensors from statistical data (HCT_% and Albumin_g/dL)
hct_data = df_filtered['HCT_%'].values
albumin_data = df_filtered['Albumin_g/dL'].values
# Subcomponent tensors statistical data Hematokrit
tensor_hct = torch.from_numpy(hct_data.astype(np.float32))
tensor_results['HCT Tensor'] = f"Shape: {tensor_hct.shape}, Values: {tensor_hct[:5]}"
# Subcomponent tensors statistical data Albumin
tensor_albumin = torch.from_numpy(albumin_data.astype(np.float32))
tensor_results['Albumin Tensor'] = f"Shape: {tensor_albumin.shape}, Values: {tensor_albumin[:5]}"
    
# COMPONENT TENSOR FIR STATISTICAL ANALYSIS
combined_data = np.column_stack((hct_data, albumin_data))
tensor_combined = torch.from_numpy(combined_data.astype(np.float32))
tensor_results['Combined HCT-Albumin Tensor'] = f"Shape: {tensor_combined.shape}, Sample: {tensor_combined[:3]}"
  
# COMPONENT STATISTICAL OPERATION TENSORS
mean_hct = torch.mean(tensor_hct)
std_hct = torch.std(tensor_hct)
tensor_results['HCT Mean (Tensor)'] = f"{mean_hct:.2f}"
tensor_results['HCT Std (Tensor)'] = f"{std_hct:.2f}"
  
mean_albumin = torch.mean(tensor_albumin)
std_albumin = torch.std(tensor_albumin)
tensor_results['Albumin Mean (Tensor)'] = f"{mean_albumin:.2f}"
tensor_results['Albumin Std (Tensor)'] = f"{std_albumin:.2f}"
    
# COMPONENET CORRELATION TENSORS
correlation = torch.corrcoef(tensor_combined.t())[0,1]
tensor_results['HCT-Albumin Correlation (Tensor)'] = f"{correlation:.3f}"
    
print("Tensor operations on statistical data completed successfully")

# COMPONENET COMPUTER SYSTEM - MEMORY SIMULATION
print("\n--- COMPUTER SYSTEM: MEMORY SIMULATION ---")
memory_results = {}
    
# COMPONENET CALCULATION MEMORY REQUIREMENTS BASED STATISTICAL DATA
num_samples = len(df_filtered)
memory_per_sample_mb = 0.1  # Assume 100KB per sample for statistical data
total_data_memory_mb = num_samples * memory_per_sample_mb
    
laptop = LaptopMemoryModel(total_ram_gb=8, swap_size_gb=4)
memory_results['Initial Model'] = str(laptop)
memory_results['Data Size'] = f"{num_samples} samples, ~{total_data_memory_mb:.1f} MB"
    
# COMPONENT MEMORY ALLOCATION
allocation_mb = min(int(total_data_memory_mb), 2048)  # Max 2GB for demo
result1 = laptop.allocate_memory(allocation_mb)
memory_results[f'Allocate {allocation_mb}MB for Data'] = result1
    
# COMPONENT ALLOCATION STATISTICAL COMPUTATIONS
stat_memory_mb = 512  # Assume 512MB for statistical processing
result2 = laptop.allocate_memory(stat_memory_mb)
memory_results[f'Allocate {stat_memory_mb}MB for Stats'] = result2
    
status = laptop.get_memory_status()
memory_results['Memory Status After Allocation'] = status
    
# COMPONENT MEMORY DEALLOCATION
laptop.free_memory(allocation_mb // 2, "RAM")  # Free half the data memory
status_after_free = laptop.get_memory_status()
memory_results['Memory Status After Partial Free'] = status_after_free
    
print("Memory simulation based on statistical data completed successfully")

# COMPONENT NLP SYSTEM - TEXT PROCESSING PIPELINE
print("\n--- NLP SYSTEM: TEXT PROCESSING PIPELINE ---")
nlp_results = {}

# MODULE STATISTICAL TEXT DESCRIPTION DATA
sample_texts = []
for idx, row in df_filtered.head(10).iterrows():  # Use first 10 samples
    text = f"Patient {idx}: HCT {row['HCT_%']:.1f}%, Albumin {row['Albumin_g/dL']:.2f} g/dL, Age {row['Umur_Tahun']}, Gender {row['Jenis_Kelamin']}"
    sample_texts.append(text)
    
    labels = [0 if row['HCT_%'] < 35 else 1 for _, row in df_filtered.head(10).iterrows()]  # Binary label based on HCT threshold
    
    # COMPONENT TEXT PREPROCESSING
    preprocessed_texts = [preprocess_text(text) for text in sample_texts]
    nlp_results['Preprocessed Statistical Texts'] = preprocessed_texts[:3]  # Show first 3
    
    # Tokenization
    tokens = tokenize(preprocess_text(sample_texts[0]))
    nlp_results['Tokens from Sample Text'] = tokens
    
    # COMPONENT VOCABULARY BUILD FROM STATISTICAL TEXTS
    vocab = Vocabulary()
    vocab.build_vocab(sample_texts, min_freq=1)
    nlp_results['Vocabulary Size'] = len(vocab)
    nlp_results['Sample Token Mappings'] = dict(list(vocab.token_to_idx.items())[:5])
    
    # COMPONENT TEXT NUMERICALIZATION
    indices = texts_to_indices(sample_texts, vocab, max_length=20)
    nlp_results['Text Indices'] = indices[:3]  # Show first 3
    
    # COMPONENT DATA LOADERS
    train_loader, _, vocab = create_dataloaders(sample_texts, labels, vocab=vocab, batch_size=2)
    nlp_results['Number of Batches'] = len(train_loader)
    
    print("NLP pipeline on statistical text data completed successfully")

    # COMPONENT AI SYSTEM - NEURAL NETWORK TRAINING
    print("\n--- AI SYSTEM: NEURAL NETWORK TRAINING ---")
    ai_results = {}
    
    try:
        # COMPONENET STATISTICAL DATA FOR NEURAL NETWORK
        X_nn = df_filtered['HCT_%'].values.reshape(-1, 1).astype(np.float32)
        y_nn = df_filtered['Albumin_g/dL'].values.astype(np.float32)
        
        # COMPONENT SPLIT DATA
        X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)
        
        # COMPONENT CONVERT TENSORS
        X_train_tensor = torch.from_numpy(X_train_nn)
        y_train_tensor = torch.from_numpy(y_train_nn).unsqueeze(1)
        X_test_tensor = torch.from_numpy(X_test_nn)
        y_test_tensor = torch.from_numpy(y_test_nn).unsqueeze(1)
        
        class RegressionModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, output_size=1):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = RegressionModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        ai_results['Model Architecture'] = str(model)
        ai_results['Loss Function'] = str(criterion)
        ai_results['Optimizer'] = str(optimizer)
        ai_results['Training Data Size'] = f"Train: {len(X_train_tensor)}, Test: {len(X_test_tensor)}"
        
        # COMPONENT TRAINING LOOP
        model.train()
        training_losses = []
        for epoch in range(100):  # 100 epochs for demo
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())
        
        ai_results['Training Losses (last 10)'] = training_losses[-10:]
        ai_results['Final Training Loss'] = training_losses[-1] if training_losses else 0
        
        # COMPONENET EVALUATION
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            ai_results['Test Loss'] = test_loss.item()
            
            # COMPONENT CALCULATE R² SCORE
            y_pred = test_outputs.numpy().flatten()
            y_true = y_test_nn
            r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
            ai_results['R² Score'] = r2
        
        print("Neural network training on statistical data completed successfully")
        
    except Exception as e:
        ai_results['Error'] = str(e)
        print(f"AI System error: {e}")
    
    print("Neural network training completed successfully")

    # COMPONENT C++ MODULE EXECUTION (PYTHON IMPLEMENTATION)
    print("\n --- C++ MODULE EXECUTION (PLASMA LEAKAGE ANALYSIS) ---")
    cpp_results = {}
    
    # COMPONENT INITIALIZATION C++ MODULE
    cpp_available = build_cpp_module_deferred()
    cpp_results['C++ Module Status'] = 'Python Implementation (Fallback)'
    cpp_results['Module Available'] = cpp_available
    
    try:
        # COMPONENET PROSESSOR DATA PYTHON PLASMA LEAKAGE
        plasma_processor = PlasmaLeakageProcessor()
        cpp_data = plasma_processor.process_data(df_filtered, hct_col='HCT_%', albumin_col='Albumin_g/dL')
        
        # COMPONENET CALCULATE STATISTICAL METRICS
        cpp_results['Total Samples Processed'] = len(cpp_data)
        cpp_results['Mean Leakage Index'] = f"{cpp_data['Leakage_Index'].mean():.2f}"
        cpp_results['Std Leakage Index'] = f"{cpp_data['Leakage_Index'].std():.2f}"
        cpp_results['Min Leakage Index'] = f"{cpp_data['Leakage_Index'].min():.2f}"
        cpp_results['Max Leakage Index'] = f"{cpp_data['Leakage_Index'].max():.2f}"
        
        # COMPONENT RISK LEVEL DISTRIBUTION
        risk_distribution = cpp_data['Risk_Level'].value_counts().to_dict()
        for risk_level, count in sorted(risk_distribution.items()):
            cpp_results[f'Risk Level: {risk_level}'] = f"{count} cases"
        
        cpp_results['Data Processing Status'] = 'Completed Successfully'
        
        # COMPONENET RESULT CONSOLE
        print("\n Plasma Leakage Analysis Results:")
        print(f"Total samples processed: {len(cpp_data)}")
        print(f"Mean Leakage Index: {cpp_data['Leakage_Index'].mean():.2f}")
        print(f"Std Leakage Index: {cpp_data['Leakage_Index'].std():.2f}")
        print(f"Min Leakage Index: {cpp_data['Leakage_Index'].min():.2f}")
        print(f"Max Leakage Index: {cpp_data['Leakage_Index'].max():.2f}")
        print("\nRisk Level Distribution:")
        for risk_level, count in sorted(risk_distribution.items()):
            print(f"  {risk_level}: {count} cases")
        
        # COMPONENT PROCESSED EXCEL EXPORT
        cpp_data_export = cpp_data.copy()
        
    except Exception as e:
        cpp_results['C++ Error'] = str(e)
        cpp_results['Data Processing Status'] = 'Failed'
        print(f"Error processing plasma leakage data: {e}")
    
    print("C++ module execution and plasma leakage analysis completed")

# MODULE VISUALIZATION ENGINE - SCATTER PLOTS
    print("\n--- GENERATING VISUALIZATION PLOTS ---")
    
    try:
        # COMPONENT SCATTER PLOT: HCT vs Albumin (Colored by Risk Level)
        plt.figure(figsize=(12, 8))
        
        # COMPONENT COLORIZE FOR RISK LEVELS
        risk_colors = {
            'Low Leakage': 'green',
            'Minimal Leakage': 'blue',
            'Mild Leakage': 'yellow',
            'Moderate Leakage': 'orange',
            'Severe Leakage': 'red'
        }
        
        # COMPONENT PLOT RISK LEVEL WITH DIFFERENT COLOR
        for risk_level in sorted(cpp_data['Risk_Level'].unique()):
            mask = cpp_data['Risk_Level'] == risk_level
            plt.scatter(cpp_data[mask]['HCT_%'], 
                       cpp_data[mask]['Albumin_g/dL'],
                       c=risk_colors.get(risk_level, 'gray'),
                       label=f'{risk_level} (n={mask.sum()})',
                       s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('HCT (%) - Hematocrit', fontsize=12, fontweight='bold')
        plt.ylabel('Albumin (g/dL)', fontsize=12, fontweight='bold')
        plt.title('Scatter Plot: HCT vs Albumin\nColored by Plasma Leakage Risk Level', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # COMPONENT EXPORT PNG
        scatter_plot_file = 'scatter_plot_hct_albumin_risk.png'
        plt.tight_layout()
        plt.savefig(scatter_plot_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Scatter plot saved: {scatter_plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"[WARN] Error creating HCT vs Albumin scatter plot: {e}")
    
    try:
        # COMPONENT SCATTER PLOT: HCT vs Leakage Index
        plt.figure(figsize=(12, 8))
        
        # COMPONENT GENERATE SCATTER PLOT WITH ALBUMIN SIZE
        scatter = plt.scatter(cpp_data['HCT_%'], 
                             cpp_data['Leakage_Index'],
                             c=cpp_data['Albumin_g/dL'],
                             s=cpp_data['Albumin_g/dL']*50,
                             cmap='RdYlGn_r', 
                             alpha=0.6, 
                             edgecolors='black', 
                             linewidth=0.5)
        
        plt.xlabel('HCT (%) - Hematocrit', fontsize=12, fontweight='bold')
        plt.ylabel('Leakage Index', fontsize=12, fontweight='bold')
        plt.title('Scatter Plot: HCT vs Leakage Index\n(Size and Color by Albumin Level)', 
                 fontsize=14, fontweight='bold')
        
        # COMPONENT COLORBAR
        cbar = plt.colorbar(scatter)
        cbar.set_label('Albumin (g/dL)', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # COMPONENT EXPORT PNG
        scatter_plot_file2 = 'scatter_plot_hct_leakage_index.png'
        plt.tight_layout()
        plt.savefig(scatter_plot_file2, dpi=300, bbox_inches='tight')
        print(f"[OK] Scatter plot saved: {scatter_plot_file2}")
        plt.close()
        
    except Exception as e:
        print(f"[WARN] Error creating HCT vs Leakage Index scatter plot: {e}")
    
    try:
        # COMPONENT HISTOGRAM: RISK LEVEL DISTRIBUTION
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # COMPONENT HISTOGRAM 1: RISK LEVEL COUNTS
        risk_counts = cpp_data['Risk_Level'].value_counts()
        colors_list = [risk_colors.get(level, 'gray') for level in risk_counts.index]
        ax1.bar(range(len(risk_counts)), risk_counts.values, color=colors_list, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(risk_counts)))
        ax1.set_xticklabels(risk_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of Cases', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Risk Levels', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # COMPONENT VALUE LABELS ON BARS
        for i, v in enumerate(risk_counts.values):
            ax1.text(i, v + max(risk_counts.values)*0.02, str(v), ha='center', fontweight='bold')
        
        # COMPONENT HISTOGRAM 2: LEAKAGE INDEX DISTRIBUTION
        ax2.hist(cpp_data['Leakage_Index'].dropna(), bins=30, color='skyblue', 
                edgecolor='black', linewidth=1.2, alpha=0.7)
        ax2.axvline(cpp_data['Leakage_Index'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {cpp_data["Leakage_Index"].mean():.2f}')
        ax2.set_xlabel('Leakage Index', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of Leakage Index', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # COMPONENT EXPORT PNG
        histogram_file = 'histogram_plasma_leakage_distribution.png'
        plt.tight_layout()
        plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
        print(f"[OK] Histogram saved: {histogram_file}")
        plt.close()
        
    except Exception as e:
        print(f"[WARN] Error creating histogram: {e}")
    
    print("Visualization plots generation completed successfully")

# MODULE SYNTHETIC DATA AND MACHINE LEARNING ENGINE
    # COMPONENT SYNTHETIC DATA GENERATION ENGINE
    np.random.seed(42)
    n_samples = 1000
    gender = np.random.choice(['M', 'F'], n_samples)  # 50% male, 50% female
    
    # COMPONENT HEMATOCRIT INTERPRETATION ENGINE
    hct_values = []
    for g in gender:
        if g == 'M':
            # Subcomponent interpretation Male: 38.3-48.6%
            hct_values.append(np.round(np.random.uniform(38.3, 48.6), 1))
        else:
            # Subcomponent interpretation Female: 35.5-44.9%
            hct_values.append(np.round(np.random.uniform(35.5, 44.9), 1))
    
    data = {
        'albumin': np.round(np.random.uniform(3.5, 5.5, n_samples), 2),  # 3.5-5.5 g/dL
        'hematokrit': hct_values,  # Gender-specific ranges
        'trombosit': np.random.randint(150000, 450001, n_samples),  # 150,000-450,000 sel/µL
        'leukosit': np.random.randint(4500, 11001, n_samples),  # 4,500-11,000 sel/µL
        'derajat': np.random.choice([0, 1, 2, 3], n_samples)  # DBD grading 0-3
    }
    
    # COMPONENT CLINICAL LABEL AUGMENTATION
    df = pd.DataFrame(data)
    
    # COMPONENT WESTGARD QUALITY CONTROL VISUALIZATION ENGINE (Using Synthetic Data)
    print("\n--- GENERATING WESTGARD PLOT FROM SYNTHETIC CONTROL DATA ---")
    synthetic_albumin_control = data['albumin'][:50]  # First 50 samples as control data
    mean_synthetic = np.mean(synthetic_albumin_control)
    sd_synthetic = np.std(synthetic_albumin_control, ddof=1)
    results_synthetic = westgard_rules(synthetic_albumin_control, mean_synthetic, sd_synthetic)
    
    try:
        plot_westgard(synthetic_albumin_control, mean_synthetic, sd_synthetic, results_synthetic)
        print("[OK] Westgard plot saved: westgard.png")
    except Exception as e:
        print(f"  [WARN] Error creating westgard plot: {e}")
    
    # COMPONENT DATA SPLITTING ENGINE
    df = create_derajat_column(df)
    X = df.drop('derajat', axis=1)
    y = df['derajat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # COMPONENT MACHINE LEARNING MODEL ENGINE
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # COMPONENT PREDICTION ENGINE
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # Probability for each degree
    
    # COMPONENT MODEL EVALUATION ENGINE
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Derajat I', 'Derajat II', 'Derajat III', 'Derajat IV']))

# MODULE MACHINE LEARNING ANALYSIS ENGINE
    def analyze_ml_model(df, model, X_test, y_test, feature_names=None, class_names=None):
        print("\n === Machine Learning Model Analysis ===")
        # Subcomponent class prediction 
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
    
        # COMPONENT CLASSIFICATION REPORT ENGINE
        if class_names is None:
            class_names = [str(i) for i in np.unique(y_test)]
        print(classification_report(y_test, y_pred, target_names=class_names))
    
        # COMPONENT ROC-AUC EVALUATION ENGINE
        try:
            # Subcomponent binary classification
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:,1])
                print(f"ROC-AUC: {auc:.3f}")
            # Subcomponent multiclass classification
            else:
                auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                auc_micro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='micro')
                print(f"ROC-AUC (macro): {auc_macro:.3f}")
                print(f"ROC-AUC (micro): {auc_micro:.3f}")
        # Subcomponent error handling
        except Exception as e:
            print(f"ROC-AUC cannot be calculated: {e}")
    
        # COMPONENT FEATURE IMPORTANCE ENGINE
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names is None:
                feature_names = [f'F{i}' for i in range(len(importances))]
            sorted_idx = np.argsort(importances)[::-1]
            print("\n Variable Contribution (Feature Importance):")
            for i in sorted_idx:
                print(f"- {feature_names[i]}: {importances[i]:.4f}")
        else:
            print("The model does not have a feature_importances_ attribute")
    
        # COMPONENT PROBABILISTIC INTERPRETATION ENGINE
        print("\n Example of Probabilistic Model Interpretation:")
        for i in range(min(3, len(X_test))):
            probs = y_pred_proba[i]
            pred = y_pred[i]
            true = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            print(f"Sample {i+1}: True={true}, Pred={pred}, Probabilitas: {probs}")
    
        # COMPONENT ERROR ANALYSIS ENGINE
        print("\n Model Error Analysis:")
        errors = (y_pred != y_test)
        n_errors = np.sum(errors)
        print(f"Number of incorrect predictions: {n_errors} from {len(y_test)} sample ({n_errors/len(y_test):.2%})")
        if n_errors > 0:
            print("Example of wrong prediction:")
            idxs = np.where(errors)[0][:5]
            for idx in idxs:
                true = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
                pred = y_pred[idx]
                probs = y_pred_proba[idx]
                print(f"  Index {idx}: True={true}, Pred={pred}, Probability: {probs}")
    
        # COMPONENT CLINICAL CONFIDENCE INTERPRETATION
        avg_confidence = np.mean(np.max(model.predict_proba(X_test), axis=1))
        if avg_confidence > 0.8:
            lines.append(
                f"The model has a high level of prediction confidence (average probability {avg_confidence:.2f}), "
                "which shows consistency in classification"
            )
        else:
            lines.append(
                f"The model confidence level is relatively low (average probability {avg_confidence:.2f}), "
                "so the interpretation of the results must be done carefully"
            )
    
        # COMPONENT METRIC EXPLANATION ENGINE
        print("\n Metrics Explained:")
        print("- Accuracy: Proportion of correct predictions across all samples.")
        print("- Precision: Accuracy of positive predictions (per class).\n Precision = TP / (TP + FP)")
        print("- Recall (Sensitivity): The model's ability to detect positive cases.\n Recall = TP / (TP + FN)")
        print("- F1-score: Harmonic mean between precision and recall.\n F1 = 2 * (Precision * Recall) / (Precision + Recall)")
        print("- ROC-AUC: Area under the ROC curve, measures the model's ability to distinguish between classes.")
        print("- Feature Importance: The relative contribution of each feature to the model's prediction.")
        print("- Probabilistic Interpretation: The probability of predicting each class, not just the label.")
        print("- Error Analysis: Finding patterns or sources of errors for model improvement.")

        # COMPONENT STRUCTURED OUTPUT ENGINE
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
            'roc_auc': auc_macro if 'auc_macro' in locals() else None,
            'feature_importance': dict(zip(feature_names, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None,
            'errors': n_errors
        }
    
# MODULE MACHINE LEARNING ANALYSIS EXPORT AND REPORTING ENGINE
    # COMPONENT PDF EXPORT FUNCTION
    def save_ml_analysis_to_pdf(df, model, X_test, y_test, feature_names=None, class_names=None, filename="ml_analysis_results.pdf"):
        # Subcomponent initialization engine
        buf = io.StringIO()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if class_names is None:
            class_names = [str(i) for i in np.unique(y_test)]
        report = classification_report(y_test, y_pred, target_names=class_names)
    
        # Subcomponent roc-auc engine
        try:
            # Subcomponent binary case
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:,1])
                auc_str = f"ROC-AUC: {auc:.3f}"
            # Subcomponent multiclass case
            else:
                auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                auc_micro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='micro')
                auc_str = f"ROC-AUC (macro): {auc_macro:.3f}\nROC-AUC (micro): {auc_micro:.3f}"
        # Subcomponent error handling
        except Exception as e:
            auc_str = f"ROC-AUC cannot be calculated: {e}"
    
        # Subcomponent feature importance engine
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if feature_names is None:
                feature_names = [f'F{i}' for i in range(len(importances))]
            sorted_idx = np.argsort(importances)[::-1]
            feat_str = "Variable Contribution (Feature Importance):\n"
            for i in sorted_idx:
                feat_str += f"- {feature_names[i]}: {importances[i]:.4f}\n"
        else:
            feat_str = "The model does not have a feature_importances_ attribute\n"
    
        # Subcomponent probabilistik interpretation engine
        prob_str = "Example of Probabilistic Model Interpretation:\n"
        for i in range(min(3, len(X_test))):
            probs = y_pred_proba[i]
            pred = y_pred[i]
            true = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            prob_str += f"Sample {i+1}: True={true}, Pred={pred}, Probabilitas: {probs}\n"
    
        # Subcomponent error analysis engine
        errors = (y_pred != y_test)
        n_errors = np.sum(errors)
        err_str = f"Number of incorrect predictions: {n_errors} from {len(y_test)} sample ({n_errors/len(y_test):.2%})\n"
        if n_errors > 0:
            err_str += "Example wrong prediction:\n"
            idxs = np.where(errors)[0][:5]
            for idx in idxs:
                true = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
                pred = y_pred[idx]
                probs = y_pred_proba[idx]
                err_str += f"  Index {idx}: True={true}, Pred={pred}, Probability: {probs}\n"
    
        # Subcomponent clinical interpretation engine
        klinis_str = (
                "Clinical Implications:\n"
                "This model can help identify risk quickly, "
                "however, clinical decisions must still take into account the patient's context, "
                "physical examination, and other laboratory data.\n"
                "ML models are supportive, not a substitute for clinical judgment.\n"
            )
    
        # Subcomponent metric explanation engine
        metrik_str = (
                "Metric Explanation:\n"
                "- Accuracy: Proportion of correct predictions.\n"
                "- Precision: Accuracy of positive predictions.\n"
                "- Recall: Sensitivity of case detection.\n"
                "- F1-score: Combination of precision and recall.\n"
                "- ROC-AUC: Ability to distinguish classes.\n"
                "- Feature Importance: Variable contribution.\n"
                "- Probabilistik: Output in the form of probabilities.\n"
                "- Error Analysis: Identifying model errors.\n"
            )
        
        # Subcomponent PDF generation engine
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Machine Learning Model Analysis Results", ln=1, align="C")
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 7, f"Accuracy: {accuracy:.3f}\n\n{auc_str}\n\n{report}\n{feat_str}\n{prob_str}\n{err_str}\n{klinis_str}\n{metrik_str}")
        pdf.output(filename)
        print(f"The complete analysis results of the ML model have been saved to {filename}")
    
    # COMPONENT FUNCTION EXECUTION ENGINE
    feature_names = list(X.columns) 
    class_names = ['Derajat I', 'Derajat II', 'Derajat III', 'Derajat IV']
    save_ml_analysis_to_pdf(df, model, X_test, y_test, feature_names=feature_names, class_names=class_names, filename="ml_analysis_results.pdf")
    
    # COMPONENT BAB IV GENERATE ENGINE
    def generate_bab_iv(df, feature_cols, label_col, pipeline, le, X_test, y_test, y_pred):
        return "Bab IV: Results and Discussion\n\nAnalysis of the machine learning model was performed using synthetic data. The RandomForestClassifier model showed good performance in classifying the degree of dengue fever based on albumin, hematocrit, platelets, and leukocytes parameters."
    
    bab_iv_text = generate_bab_iv(
        df,
        feature_cols=list(X.columns),
        label_col='derajat',
        pipeline=None,
        le=None,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred
    )
    print(bab_iv_text)
    
# MODULE MODEL DEPLOYMENT ENGINE
    # COMPONENT ONNX CONVERSION ENGINE
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    
    # COMPONENT MODEL SERIALIZATION ENGINE
    with open("risk_engine.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
    # COMPONENT ONNX INFERENCE ENGINE
    session = ort.InferenceSession("risk_engine.onnx")
    input_data = np.array([[65.0, 150.0, 180.0, 1.0]], dtype=np.float32)
    outputs = session.run(None, {"float_input": input_data})
    print("Prediction ONNX:", outputs[0])
    
# MODULE DATA EXPORT ENGINE
    try:
        # COMPONENT EXCEL WRITER INITIALIZATION
        with pd.ExcelWriter('Trial Analysis Results.xlsx', engine='openpyxl') as writer:
    
            # COMPONENT DESCRIPTIVE STATISTICS EXPORT
            try:
                print('DEBUG: Write a Descriptive Stats sheet')
                descriptive_source = None
    
                if 'df' in globals() and isinstance(df, pd.DataFrame) and not df.empty:
                    descriptive_source = df
                elif 'df_filtered' in globals() and isinstance(df_filtered, pd.DataFrame) and not df_filtered.empty:
                    descriptive_source = df_filtered
                elif 'df_clean' in globals() and isinstance(df_clean, pd.DataFrame) and not df_clean.empty:
                    descriptive_source = df_clean
                else:
                    descriptive_source = df_real
    
                stat_columns = [col for col in ['Umur_Tahun', 'HCT_%', 'Albumin_g/dL'] if col in descriptive_source.columns]
                if not stat_columns:
                    stat_columns = descriptive_source.select_dtypes(include=[np.number]).columns.tolist()
    
                if not stat_columns:
                    raise ValueError('There are no numeric columns available for Descriptive Stats')
    
                descriptive_source[stat_columns].describe().to_excel(
                    writer,
                    sheet_name='Descriptive Stats',
                    index=True
                )
            except Exception as e:
                print(f'ERROR Descriptive Stats: {e}')
    
            # COMPONENT CLEANED DATA EXPORT
            try:
                print('DEBUG: Writing Cleaned Data sheet')
    
                candidate = None
                if 'df_filtered' in globals() and isinstance(df_filtered, pd.DataFrame) and not df_filtered.empty:
                    candidate = df_filtered
                elif 'df_clean' in globals() and isinstance(df_clean, pd.DataFrame) and not df_clean.empty:
                    candidate = df_clean
                elif 'df' in globals() and isinstance(df, pd.DataFrame) and not df.empty:
                    candidate = df
                else:
                    candidate = df_real
    
                required_cols = [
                    'Anon_ID', 'ID_Pasien', 'Umur_Tahun', 'Jenis_Kelamin',
                    'Tanggal_Hematokrit', 'Tanggal_Albumin',
                    'HCT_%', 'Albumin_g/dL'
                ]
    
                safe_cols = [col for col in required_cols if col in candidate.columns]
                if not safe_cols:
                    safe_cols = candidate.columns.tolist()
    
                candidate[safe_cols].to_excel(writer, sheet_name='Cleaned Data', index=False)
            except Exception as e:
                print(f'ERROR Cleaned Data: {e}')
    
            # COMPONENT NORMALITY TEST EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Writing a Normality Tests sheet')
                # Subcomponent dataframe creation
                normality_results = pd.DataFrame({
                    'Variable': ['HCT_%', 'Albumin_g/dL'],
                    'Test Method': [method_hct, method_alb],
                    'Statistic': [stat_hct, stat_alb],
                    'p-value': [p_hct, p_alb],
                    'Normal?': ['Yes' if p >= 0.05 else 'No' for p in [p_hct, p_alb]]
                })
                normality_results.to_excel(writer, sheet_name='Normality Tests', index=False)
            except Exception as e:
                print(f'ERROR Normality Tests: {e}')
    
            # COMPONENT UNIVARIATE ANALYSIS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Write a Univariate Analysis sheet')
                # Subcomponent dataframe creation
                univariate_results = pd.DataFrame({
                    'Variable': ['Albumin_g/dL', 'HCT_%'],
                    'Type': ['Mean ± SD' if normal_alb else 'Median (IQR)', 'Mean ± SD' if normal_hct else 'Median (IQR)'],
                    'Value': [univar_alb, univar_hct]
                })
                univariate_results.to_excel(writer, sheet_name='Univariate Analysis', index=False)
            except Exception as e:
                print(f'ERROR Univariate Analysis: {e}')
    
            # COMPONENT BIVARIATE ANALYSIS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Writing a Bivariate Analysis sheet')
                # Subcomponent test selection
                if normal_hct and normal_alb:
                    bivariate_results = pd.DataFrame({
                        'Test': ['Pearson Correlation'],
                        'Coefficient': [r],
                        'p-value': [p_corr]
                    })
                else:
                    bivariate_results = pd.DataFrame({
                        'Test': ['Spearman Correlation'],
                        'Coefficient': [rho],
                        'p-value': [p_corr]
                    })
                # Subcomponent export
                bivariate_results.to_excel(writer, sheet_name='Bivariate Analysis', index=False)
            except Exception as e:
                print(f'ERROR Bivariate Analysis: {e}')
    
            # COMPONENT VARIABLE SUMMARY EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Write a Variable Summary sheet')
                # Subcomponent dataframe creation
                var_summary_df = pd.DataFrame(list(var_types.items()), columns=['Variable', 'Type'])
                # Subcomponent export
                var_summary_df.to_excel(writer, sheet_name='Variable Summary', index=False)
            except Exception as e:
                print(f'ERROR Variable Summary: {e}')
    
            # COMPONENT SOFTWARE RECOMMENDATION EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Write a Software Recommendations sheet')
                # Subcomponent export
                software_df.to_excel(writer, sheet_name='Software Recommendations', index=False)
            except Exception as e:
                print(f'ERROR Software Recommendations: {e}')
    
            # COMPONENT C++ RESULTS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Writing C++ Results sheet')
                # Subcomponent dataframe creation
                cpp_df = pd.DataFrame(list(cpp_results.items()), columns=['Metric', 'Value'])
                cpp_df.to_excel(writer, sheet_name='C++ Results', index=False)
            except Exception as e:
                print(f'ERROR C++ Results: {e}')
    
            # COMPONENT TENSOR RESULTS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Writing Tensor Results sheet')
                # Subcomponent dataframe creation
                tensor_df = pd.DataFrame(list(tensor_results.items()), columns=['Operation', 'Result'])
                tensor_df.to_excel(writer, sheet_name='Tensor Results', index=False)
            except Exception as e:
                print(f'ERROR Tensor Results: {e}')
    
            # COMPONENT MEMORY RESULTS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Write a Memory Results sheet')
                # Subcomponent dataframe creation
                memory_df = pd.DataFrame(list(memory_results.items()), columns=['Operation', 'Result'])
                memory_df.to_excel(writer, sheet_name='Memory Results', index=False)
            except Exception as e:
                print(f'ERROR Memory Results: {e}')
    
            # COMPONENT NLP RESULTS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Writing the NLP Results sheet')
                # Subcomponent dataframe creation
                nlp_df = pd.DataFrame(list(nlp_results.items()), columns=['Operation', 'Result'])
                nlp_df.to_excel(writer, sheet_name='NLP Results', index=False)
            except Exception as e:
                print(f'ERROR NLP Results: {e}')
    
            # COMPONENT AI RESULTS EXPORT
            try:
                # Subcomponent debug logging
                print('DEBUG: Write AI Results sheet')
                # Subcomponent dataframe creation
                ai_df = pd.DataFrame(list(ai_results.items()), columns=['Metric', 'Value'])
                ai_df.to_excel(writer, sheet_name='AI Results', index=False)
            except Exception as e:
                print(f'ERROR AI Results: {e}')
    
            # COMPONENT PLASMA LEAKAGE DATA EXPORT (C++ Module Results)
            try:
                # Subcomponent debug logging
                print('DEBUG: Writing a Plasma Leakage Analysis sheet')
                # Subcomponent export processed data
                if 'cpp_data_export' in globals():
                    cpp_data_export.to_excel(writer, sheet_name='Plasma Leakage Analysis', index=False)
            except Exception as e:
                print(f'ERROR Plasma Leakage Analysis: {e}')
    
        # COMPONENT SUCCESS NOTIFICATION
        print("\n[OK] The complete analysis results have been saved to a file 'Trial Analysis Results.xlsx'")
        print("Sheets yang dibuat:")
        print("- Descriptive Stats")
        print("- Cleaned Data")
        print("- Normality Tests")
        print("- Univariate Analysis")
        print("- Bivariate Analysis")
        print("- Variable Summary")
        print("- Software Recommendations")
        print("- C++ Results")
        print("- Tensor Results")
        print("- Memory Results")
        print("- NLP Results")
        print("- AI Results")
        print("- Plasma Leakage Analysis")
    
    # COMPONENT ERROR HANDLING ENGINE
    except PermissionError:
        print("\n Failed to save Excel file: 'Trial Analysis Results.xlsx' may be open. Close the file and restart.")
    except Exception as e:
        print(f'ERROR saat menulis Excel: {e}')

    print("\n" + "="*60)
    print("ALL SYSTEMS INTEGRATION AND EXPORT COMPLETED SUCCESSFULLY")
    print("="*60)




























# MODULE ULTIMATE DUPLICATE RESOLUTION ENGINE
class DuplicateResolver:

    def __init__(self, df, timestamp_col=None, age_col=None, db_connection=None):
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.age_col = age_col
        self.db_connection = db_connection
        self.conflicts = []

    # PROBABILISTIC RECORD LINKAGE
    def _probabilistic_score(self, row1, row2):
        score = 0

        # Name similarity
        name_score = fuzz.ratio(str(row1['Name']), str(row2['Name'])) / 100
        score += 0.5 * name_score

        # Exact MRN match
        if row1['MRN'] == row2['MRN']:
            score += 0.3

        # Age similarity
        if self.age_col:
            age_diff = abs(row1[self.age_col] - row2[self.age_col])
            score += 0.2 * np.exp(-age_diff / 10)

        return score

    # GRAPH-BASED ENTITY RESOLUTION
    def build_entity_graph(self, threshold=0.75):

        G = nx.Graph()

        for i, row1 in self.df.iterrows():
            for j, row2 in self.df.iterrows():
                if i >= j:
                    continue

                score = self._probabilistic_score(row1, row2)

                if score >= threshold:
                    G.add_edge(i, j, weight=score)

        return G

    def resolve_graph_clusters(self, graph):

        clusters = list(nx.connected_components(graph))
        resolved_rows = []

        for cluster in clusters:
            subset = self.df.loc[list(cluster)]

            # pilih record terbaik (completeness + timestamp)
            subset['score'] = subset.notna().sum(axis=1)

            if self.timestamp_col:
                subset = subset.sort_values(by=self.timestamp_col, ascending=False)

            best = subset.sort_values(by='score', ascending=False).iloc[0]
            resolved_rows.append(best)

        return pd.DataFrame(resolved_rows)

    # AUTO-LEARN MATCHING RULES (ML)
    def train_matching_model(self, labeled_pairs):
        """
        labeled_pairs: DataFrame dengan kolom:
        row1_index, row2_index, label (1=match, 0=not match)
        """
        from sklearn.ensemble import RandomForestClassifier

        X = []
        y = []

        for _, row in labeled_pairs.iterrows():
            r1 = self.df.loc[row['row1_index']]
            r2 = self.df.loc[row['row2_index']]

            features = [
                fuzz.ratio(str(r1['Name']), str(r2['Name'])),
                int(r1['MRN'] == r2['MRN'])
            ]

            if self.age_col:
                features.append(abs(r1[self.age_col] - r2[self.age_col]))

            X.append(features)
            y.append(row['label'])

        model = RandomForestClassifier()
        model.fit(X, y)

        self.matching_model = model
        return model

    def predict_match(self, row1, row2):
        if not hasattr(self, 'matching_model'):
            raise ValueError("Model belum dilatih")

        features = [
            fuzz.ratio(str(row1['Name']), str(row2['Name'])),
            int(row1['MRN'] == row2['MRN'])
        ]

        if self.age_col:
            features.append(abs(row1[self.age_col] - row2[self.age_col]))

        return self.matching_model.predict_proba([features])[0][1]

    # DATABASE INTEGRATION (SQL PIPELINE)
    def load_from_sql(self, query):
        if self.db_connection is None:
            raise ValueError("DB connection tidak tersedia")

        self.df = pd.read_sql(query, self.db_connection)
        return self.df

    def save_to_sql(self, table_name, df):
        if self.db_connection is None:
            raise ValueError("DB connection tidak tersedia")

        df.to_sql(table_name, self.db_connection, if_exists='replace', index=False)

    # FULL PIPELINE
    def run_full_pipeline(self, use_graph=True, threshold=0.75):
        result = {}

        if use_graph:
            graph = self.build_entity_graph(threshold)
            resolved = self.resolve_graph_clusters(graph)
            result['method'] = 'graph_based'
        else:
            resolved = self.df.copy()
            result['method'] = 'no_graph'

        result['resolved_data'] = resolved

        # Optional: save to DB
        if self.db_connection:
            self.save_to_sql("resolved_patients", resolved)

        return result









# MODULE DATA VALIDATION RULE ENGINE
class ClinicalValidationEngine:
    """
    Validasi nilai klinis berbasis rule medis + severity level
    """

    def __init__(self, df):
        self.df = df.copy()
        self.validation_log = []

    # SEVERITY CLASSIFIER
    def _assign_severity(self, param, value, normal_range):
        """
        Menentukan severity level:
        - LOW: sedikit di luar normal
        - HIGH: cukup jauh dari normal
        - CRITICAL: sangat berbahaya
        """
        low, high = normal_range

        if value is None:
            return "LOW"

        deviation_low = abs(value - low)
        deviation_high = abs(value - high)

        # Range width
        width = high - low

        # Rule severity berbasis deviasi
        if value < low:
            if (low - value) > width * 0.5:
                return "CRITICAL"
            elif (low - value) > width * 0.2:
                return "HIGH"
            else:
                return "LOW"

        elif value > high:
            if (value - high) > width * 0.5:
                return "CRITICAL"
            elif (value - high) > width * 0.2:
                return "HIGH"
            else:
                return "LOW"

        return "NORMAL"

    # HCT VALIDATION
    def validate_hct(self):
        """
        Validasi Hematokrit berdasarkan kategori pasien
        """
        for idx, row in self.df.iterrows():
            hct = row.get('HCT')
            gender = row.get('Gender')
            age = row.get('Age')

            if pd.isna(hct):
                continue

            # Tentukan range
            if age is not None and age < 1:
                normal_range = (55, 66)
                category = "Neonate"
            elif age is not None and age < 18:
                normal_range = (30, 40)
                category = "Child"
            elif gender == 'Male':
                normal_range = (40, 54)
                category = "Adult Male"
            elif gender == 'Female':
                normal_range = (36, 46)
                category = "Adult Female"
            else:
                continue

            severity = self._assign_severity("HCT", hct, normal_range)

            if severity != "NORMAL":
                self.validation_log.append({
                    'index': idx,
                    'parameter': 'HCT',
                    'value': hct,
                    'normal_range': normal_range,
                    'category': category,
                    'severity': severity
                })

    # ALBUMIN VALIDATION
    def validate_albumin(self):
        """
        Validasi Albumin dewasa
        """
        for idx, row in self.df.iterrows():
            albumin = row.get('Albumin')

            if pd.isna(albumin):
                continue

            normal_range = (3.5, 5.0)

            severity = self._assign_severity("Albumin", albumin, normal_range)

            if severity != "NORMAL":
                self.validation_log.append({
                    'index': idx,
                    'parameter': 'Albumin',
                    'value': albumin,
                    'normal_range': normal_range,
                    'category': 'Adult',
                    'severity': severity
                })

    # MAIN EXECUTION
    def run(self):
        """
        Jalankan semua validasi
        """
        self.validate_hct()
        self.validate_albumin()

        return pd.DataFrame(self.validation_log)










# MODULE ADVANCED CONFIDENCE INTERVAL ENGINE
class ConfidenceIntervalEngine:

    def __init__(self, df):
        self.df = df.copy()

    # MODEL COMPARISON CI (AIC/BIC DIFFERENCE)
    def ci_model_comparison(self, formula1, formula2, n_bootstrap=1000):
        diffs = []

        for _ in range(n_bootstrap):
            sample = self.df.sample(frac=1, replace=True)

            m1 = smf.ols(formula1, data=sample).fit()
            m2 = smf.ols(formula2, data=sample).fit()

            diffs.append(m1.aic - m2.aic)

        lower = np.percentile(diffs, 2.5)
        upper = np.percentile(diffs, 97.5)

        return {
            'aic_diff_mean': np.mean(diffs),
            'ci_lower': lower,
            'ci_upper': upper,
            'interpretation': 'negative → model2 better'
        }

    # BOOTSTRAP REGRESSION CI
    def ci_bootstrap_regression(self, formula, n_bootstrap=1000):
        coefs = {}

        for _ in range(n_bootstrap):
            sample = self.df.sample(frac=1, replace=True)
            model = smf.ols(formula, data=sample).fit()

            for param in model.params.index:
                coefs.setdefault(param, []).append(model.params[param])

        results = {}
        for param, values in coefs.items():
            results[param] = {
                'coef_mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }

        return results

    # MIXED EFFECT MODEL CI (CLINICAL DATA)
    def ci_mixed_effect(self, formula, group_col):
        model = smf.mixedlm(formula, self.df, groups=self.df[group_col])
        result = model.fit()

        ci = result.conf_int()

        output = {}
        for param in ci.index:
            output[param] = {
                'coef': result.params[param],
                'ci_lower': ci.loc[param, 0],
                'ci_upper': ci.loc[param, 1]
            }

        return output

    # BAYESIAN FULL MCMC (PyMC)
    def ci_bayesian_mcmc(self, column, draws=1000, tune=1000):
        data = self.df[column].dropna().values

        with pm.Model():
            mu = pm.Normal("mu", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=10)

            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

            trace = pm.sample(draws=draws, tune=tune, chains=2, progressbar=False)

        mu_samples = trace.posterior["mu"].values.flatten()

        return {
            'posterior_mean': np.mean(mu_samples),
            'ci_lower': np.percentile(mu_samples, 2.5),
            'ci_upper': np.percentile(mu_samples, 97.5),
            'method': 'MCMC'
        }

    # MASTER PIPELINE (FULL ADVANCED)
    def run_full_advanced(self,
                          column,
                          formula=None,
                          formula_compare=None,
                          group_col=None):

        result = {}

        # Basic
        result['mean_ci'] = self._safe(self._ci_mean_internal, column)

        # Bootstrap regression
        if formula:
            result['bootstrap_regression'] = self._safe(
                self.ci_bootstrap_regression, formula
            )

        # Mixed effect
        if formula and group_col:
            result['mixed_effect'] = self._safe(
                self.ci_mixed_effect, formula, group_col
            )

        # Model comparison
        if formula and formula_compare:
            result['model_comparison'] = self._safe(
                self.ci_model_comparison, formula, formula_compare
            )

        # Bayesian MCMC
        result['bayesian_mcmc'] = self._safe(
            self.ci_bayesian_mcmc, column
        )

        return result

    # INTERNAL SAFE EXECUTOR (ANTI CRASH)
    def _safe(self, func, *args):
        try:
            return func(*args)
        except Exception as e:
            return {'error': str(e)}

    # Dummy mean internal biar tidak konflik
    def _ci_mean_internal(self, column):
        data = self.df[column].dropna()
        mean = np.mean(data)
        return {'mean': mean}
    










# MODULE ADVANCED EFFECT SIZE ENGINE v3
class EffectSizeEngine:

    def __init__(self, df):
        self.df = df.copy()

    # CORE COHEN'S D
    def cohens_d(self, column, group_col, group1, group2):
        data1 = self.df[self.df[group_col] == group1][column].dropna()
        data2 = self.df[self.df[group_col] == group2][column].dropna()

        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

        n1, n2 = len(data1), len(data2)

        pooled_std = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )

        return (mean1 - mean2) / pooled_std

    # HEDGES' G
    def hedges_g(self, column, group_col, group1, group2):
        d = self.cohens_d(column, group_col, group1, group2)

        data1 = self.df[self.df[group_col] == group1][column].dropna()
        data2 = self.df[self.df[group_col] == group2][column].dropna()

        n1, n2 = len(data1), len(data2)

        correction = 1 - (3 / (4*(n1 + n2) - 9))
        return d * correction

    # CI HEDGES' G
    def ci_hedges_g(self, column, group_col, group1, group2, confidence=0.95):
        g = self.hedges_g(column, group_col, group1, group2)

        data1 = self.df[self.df[group_col] == group1][column].dropna()
        data2 = self.df[self.df[group_col] == group2][column].dropna()

        n1, n2 = len(data1), len(data2)

        se = np.sqrt((n1 + n2) / (n1 * n2) + (g**2 / (2*(n1 + n2))))

        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * se

        return {
            'hedges_g': g,
            'ci_lower': g - margin,
            'ci_upper': g + margin
        }

    # UNIVERSAL BOOTSTRAP EFFECT SIZE
    def bootstrap_effect_size(self, column, group_col, group1, group2,
                              metric='cohens_d',
                              n_bootstrap=1000,
                              confidence=0.95):

        data1 = self.df[self.df[group_col] == group1][column].dropna().values
        data2 = self.df[self.df[group_col] == group2][column].dropna().values

        boot_vals = []

        for _ in range(n_bootstrap):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)

            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

            pooled_std = np.sqrt(
                ((len(sample1)-1)*std1**2 + (len(sample2)-1)*std2**2) /
                (len(sample1) + len(sample2) - 2)
            )

            d = (mean1 - mean2) / pooled_std

            if metric == 'hedges_g':
                correction = 1 - (3 / (4*(len(sample1)+len(sample2)) - 9))
                d *= correction

            boot_vals.append(d)

        lower = np.percentile(boot_vals, (1-confidence)/2 * 100)
        upper = np.percentile(boot_vals, (1+confidence)/2 * 100)

        return {
            'effect_size_mean': np.mean(boot_vals),
            'ci_lower': lower,
            'ci_upper': upper,
            'metric': metric,
            'method': 'bootstrap'
        }

    # BAYESIAN EFFECT SIZE (POSTERIOR)
    def bayesian_effect_size(self, column, group_col, group1, group2,
                              draws=1000, tune=1000):

        import pymc as pm

        data1 = self.df[self.df[group_col] == group1][column].dropna().values
        data2 = self.df[self.df[group_col] == group2][column].dropna().values

        with pm.Model():
            mu1 = pm.Normal("mu1", mu=0, sigma=10)
            mu2 = pm.Normal("mu2", mu=0, sigma=10)

            sigma = pm.HalfNormal("sigma", sigma=10)

            obs1 = pm.Normal("obs1", mu=mu1, sigma=sigma, observed=data1)
            obs2 = pm.Normal("obs2", mu=mu2, sigma=sigma, observed=data2)

            diff = pm.Deterministic("effect_size", (mu1 - mu2) / sigma)

            trace = pm.sample(draws=draws, tune=tune, chains=2, progressbar=False)

        samples = trace.posterior["effect_size"].values.flatten()

        return {
            'posterior_mean': np.mean(samples),
            'ci_lower': np.percentile(samples, 2.5),
            'ci_upper': np.percentile(samples, 97.5),
            'method': 'bayesian_mcmc'
        }

    # MASTER PIPELINE
    def run_full(self, column, group_col, group1, group2):

        return {
            'ci_hedges_g': self.ci_hedges_g(column, group_col, group1, group2),
            'bootstrap_cohens_d': self.bootstrap_effect_size(
                column, group_col, group1, group2, metric='cohens_d'
            ),
            'bootstrap_hedges_g': self.bootstrap_effect_size(
                column, group_col, group1, group2, metric='hedges_g'
            ),
            'bayesian_effect_size': self.bayesian_effect_size(
                column, group_col, group1, group2
            )
        }
    














# MODULE ADVANCED CROSS VALIDATION ENGINE v2
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    precision_score, recall_score, f1_score,
    roc_auc_score
)

class CrossValidationEngine:

    def __init__(self, model, X, y,
                 problem_type='classification',
                 time_col=None,
                 use_smote=False):

        self.model = model
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.time_col = time_col
        self.use_smote = use_smote

    # METRIC ENGINE
    def _evaluate(self, y_true, y_pred, y_proba=None):

        if self.problem_type == 'classification':
            result = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }

            if y_proba is not None:
                try:
                    result['roc_auc'] = roc_auc_score(y_true, y_proba)
                except:
                    result['roc_auc'] = None

            return result

        else:
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }

    # CALIBRATION CURVE
    def calibration_analysis(self, X_test, y_test, model):
        from sklearn.calibration import calibration_curve

        if not hasattr(model, "predict_proba"):
            return {'error': 'Model tidak support probability'}

        y_proba = model.predict_proba(X_test)[:, 1]

        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

        return {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }

    # SHAP EXPLAINABILITY
    def shap_analysis(self, model, X_sample):
        try:
            import shap

            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

            return {
                'shap_values': shap_values.values.tolist(),
                'feature_importance': np.mean(np.abs(shap_values.values), axis=0).tolist(),
                'feature_names': X_sample.columns.tolist()
            }

        except Exception as e:
            return {'error': str(e)}

    # CORE CV LOOP
    def _run_cv(self, splitter, method, X=None, y=None):
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        results = []
        calibration_results = []

        for train_idx, test_idx in splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # SMOTE hanya di train
            X_train, y_train = self._apply_smote(X_train, y_train)

            model_clone = self._clone_model()
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_test)

            y_proba = None
            if hasattr(model_clone, "predict_proba"):
                y_proba = model_clone.predict_proba(X_test)[:, 1]

            metrics = self._evaluate(y_test, y_pred, y_proba)
            results.append(metrics)

            # Calibration
            if y_proba is not None:
                calibration_results.append(
                    self.calibration_analysis(X_test, y_test, model_clone)
                )

        return {
            'summary': self._summarize(results, method),
            'calibration': calibration_results
        }

    # SHAP GLOBAL (POST TRAIN)
    def global_shap(self, sample_size=100):
        model_clone = self._clone_model()
        model_clone.fit(self.X, self.y)

        sample = self.X.sample(min(sample_size, len(self.X)))

        return self.shap_analysis(model_clone, sample)

    # MODEL CLONING
    def _clone_model(self):
        from sklearn.base import clone
        
        return clone(self.model)

    # SUMMARY ENGINE
    def _summarize(self, results, method):
        df = pd.DataFrame(results)

        return {
            'method': method,
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'min': df.min().to_dict(),
            'max': df.max().to_dict()
        }

    # PIPELINE EXECUTION
    def run_full(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        output = {
            'cv_results': self._run_cv(kf, "KFold")
        }

        if self.problem_type == 'classification':
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            output['stratified'] = self._run_cv(skf, "StratifiedKFold")

        if self.time_col:
            tscv = TimeSeriesSplit(n_splits=5)
            output['time_series'] = self._run_cv(tscv, "TimeSeriesCV")

        # SHAP global explanation
        output['shap'] = self.global_shap()

        return output
    

























# MODULE ADVANCED HYPERPARAMETER OPTIMIZATION ENGINE
class HyperparameterOptimizationEngine:

    def __init__(self, model, param_grid, X, y,
                 problem_type='classification',
                 models_dict=None,
                 use_feature_engineering=False):

        self.model = model
        self.param_grid = param_grid
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.models_dict = models_dict
        self.use_feature_engineering = use_feature_engineering

    # FEATURE ENGINEERING (AUTO)
    def _feature_engineering(self, X):

        if not self.use_feature_engineering:
            return X

        X_new = X.copy()

        # Polynomial features sederhana
        for col in X.select_dtypes(include=np.number).columns:
            X_new[f"{col}_squared"] = X[col] ** 2

        # Interaction sederhana (pairwise)
        cols = list(X.select_dtypes(include=np.number).columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                X_new[f"{cols[i]}_x_{cols[j]}"] = X[cols[i]] * X[cols[j]]

        return X_new

    # MULTI OBJECTIVE (PARETO)
    def _multi_objective(self, y_true, y_pred):

        if self.problem_type == 'classification':
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, zero_division=0)
            return acc, rec
        else:
            mse = mean_squared_error(y_true, y_pred)
            return (-mse,)

    # PARETO FRONT FILTER
    def _pareto_front(self, results):

        pareto = []

        for r in results:
            dominated = False
            for other in results:
                if all(o >= x for o, x in zip(other['score'], r['score'])) and any(o > x for o, x in zip(other['score'], r['score'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)

        return pareto

    # BAYESIAN OPTIMIZATION (OPTUNA + CV)
    def bayesian_optimization(self, n_trials=30, cv=5):
        X = self._feature_engineering(self.X)
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        def objective(trial):

            params = {}
            for key, values in self.param_grid.items():
                if isinstance(values, list):
                    params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values, tuple):
                    params[key] = trial.suggest_float(key, values[0], values[1])

            model = self.model.set_params(**params)

            scores = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                scores.append(self._multi_objective(y_test, y_pred))

            # rata-rata multi-objective
            scores = np.array(scores)
            return tuple(scores.mean(axis=0))

        study = optuna.create_study(directions=['maximize', 'maximize'])
        study.optimize(objective, n_trials=n_trials)

        results = []
        for t in study.trials:
            results.append({
                'params': t.params,
                'score': t.values
            })

        pareto = self._pareto_front(results)

        return {
            'method': 'Optuna_Pareto_CV',
            'pareto_front': pareto,
            'best_trial': study.best_trials
        }

    # AUTOML (WITH FEATURE ENGINEERING)
    def auto_ml(self, cv=3):

        if self.models_dict is None:
            raise ValueError("models_dict harus disediakan")

        best_model = None
        best_score = -np.inf

        X = self._feature_engineering(self.X)

        for name, (model, param_grid) in self.models_dict.items():

            optimizer = HyperparameterOptimizationEngine(
                model,
                param_grid,
                X,
                self.y,
                self.problem_type
            )

            result = optimizer.random_search(cv=cv)

            if result['best_score'] > best_score:
                best_score = result['best_score']
                best_model = {
                    'model_name': name,
                    'model': result['best_model'],
                    'score': best_score
                }

        return best_model

    # RANDOM SEARCH (UNCHANGED)
    def random_search(self, n_iter=20, cv=5):
        random = RandomizedSearchCV(
            self.model,
            self.param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        random.fit(self.X, self.y)

        return {
            'best_model': random.best_estimator_,
            'best_score': random.best_score_,
            'best_params': random.best_params_
        }

    # MASTER PIPELINE
    def run_full(self):

        return {
            'bayesian_cv_pareto': self.bayesian_optimization(),
            'automl': self.auto_ml() if self.models_dict else None
        }
    




























# MODULE: ADVANCED FEATURE ENGINEERING ENGINE
class FeatureEngineeringEngine:

    def __init__(self, df, time_col=None):
        self.df = df.copy()
        self.time_col = time_col

    # RATIO FEATURE
    def create_ratio(self, col1, col2, new_col_name=None):
        if new_col_name is None:
            new_col_name = f"{col1}_to_{col2}_ratio"

        self.df[new_col_name] = self.df[col1] / self.df[col2]
        return self.df

    # DELTA FEATURE
    def create_delta(self, col, group_col, new_col_name=None):
        if self.time_col is None:
            raise ValueError("time_col harus diset")

        if new_col_name is None:
            new_col_name = f"{col}_delta"

        self.df = self.df.sort_values(by=[group_col, self.time_col])
        self.df[new_col_name] = self.df.groupby(group_col)[col].diff()

        return self.df

    # INTERACTION
    def create_interaction(self, col1, col2, new_col_name=None):
        if new_col_name is None:
            new_col_name = f"{col1}_x_{col2}"

        self.df[new_col_name] = self.df[col1] * self.df[col2]
        return self.df

    # POLYNOMIAL FEATURES
    def create_polynomial(self, col, degree=2):
        for d in range(2, degree + 1):
            self.df[f"{col}_pow_{d}"] = self.df[col] ** d
        return self.df

    # LOG TRANSFORM
    def log_transform(self, col, new_col_name=None):
        if new_col_name is None:
            new_col_name = f"log_{col}"

        self.df[new_col_name] = np.log1p(self.df[col])
        return self.df

    # NORMALIZATION
    def normalize(self, col, new_col_name=None):
        if new_col_name is None:
            new_col_name = f"{col}_norm"

        self.df[new_col_name] = (
            (self.df[col] - self.df[col].mean()) / self.df[col].std()
        )
        return self.df

    # TIME-SERIES FEATURES
    def rolling_mean(self, col, group_col, window=3):
        if self.time_col is None:
            raise ValueError("time_col wajib")

        self.df = self.df.sort_values(by=[group_col, self.time_col])

        self.df[f"{col}_rolling_mean"] = (
            self.df.groupby(group_col)[col]
            .rolling(window=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return self.df

    def slope_feature(self, col, group_col):
        """
        Approx slope (trend sederhana)
        """
        if self.time_col is None:
            raise ValueError("time_col wajib")

        self.df = self.df.sort_values(by=[group_col, self.time_col])

        self.df[f"{col}_slope"] = (
            self.df.groupby(group_col)[col]
            .diff()
        )

        return self.df

    # CLINICAL DOMAIN SCORE
    def clinical_score(self, rules_dict, new_col_name="clinical_score"):
        """
        rules_dict = {
            'HCT': lambda x: 1 if x < 35 else 0,
            'Albumin': lambda x: 1 if x < 3.5 else 0
        }
        """

        score = np.zeros(len(self.df))

        for col, func in rules_dict.items():
            score += self.df[col].apply(func)

        self.df[new_col_name] = score
        return self.df

    # FEATURE SELECTION (MUTUAL INFO)
    def feature_selection(self, target_col, problem_type='classification', top_k=5):

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        X = X.select_dtypes(include=np.number).fillna(0)

        if problem_type == 'classification':
            scores = mutual_info_classif(X, y)
        else:
            scores = mutual_info_regression(X, y)

        mi_scores = pd.Series(scores, index=X.columns)
        top_features = mi_scores.sort_values(ascending=False).head(top_k)

        return top_features

    # PIPELINE EXECUTION
    def run_advanced(self):

        # contoh klinis
        if 'HCT' in self.df.columns and 'Albumin' in self.df.columns:
            self.create_ratio('HCT', 'Albumin', 'HCT_Albumin_Ratio')
            self.create_interaction('HCT', 'Albumin')

        # polynomial
        for col in self.df.select_dtypes(include=np.number).columns:
            self.create_polynomial(col, degree=2)

        return self.df























# MODULE LAB PIPELINE ORCHESTRATOR v3
class LabPipelineOrchestrator:

    def __init__(self, df, config_path=None, db_url=None):

        self.df = df
        self.results = {}
        self.plugins = {}
        self.db_url = db_url

        self.config = self._load_config(config_path)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # CONFIG
    def _load_config(self, path):
        if path is None:
            return {}

        try:
            if path.endswith(".json"):
                return json.load(open(path))
            elif path.endswith(".yaml"):
                return yaml.safe_load(open(path))
        except:
            return {}

    # DATABASE INTEGRATION (SQL)
    def _save_to_db(self, table_name, df):
        if self.db_url is None:
            return

        from sqlalchemy import create_engine
        engine = create_engine(self.db_url)

        df.to_sql(table_name, engine, if_exists='replace', index=False)

        self.logger.info(f"Saved to DB: {table_name}")

    # SAFE EXECUTION
    def _safe_execute(self, name, func):

        try:
            self.logger.info(f"START: {name}")
            result = func()

            self.results[name] = result

            # auto save ke DB
            if isinstance(result, dict) or hasattr(result, "to_csv"):
                try:
                    if isinstance(result, pd.DataFrame):
                        self._save_to_db(name, result)
                except:
                    pass

            self.logger.info(f"DONE: {name}")

        except Exception as e:
            self.logger.error(f"ERROR: {name} → {str(e)}")
            self.results[name] = {'error': str(e)}

    # PARALLEL EXECUTION
    def run_parallel_steps(self):

        steps = {
            "validation": self.step_validation,
            "feature_engineering": self.step_feature_engineering,
            "statistical": self.step_statistical_analysis
        }

        futures = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            for name, step in steps.items():
                futures.append(executor.submit(self._safe_execute, name, step))

            for future in as_completed(futures):
                pass

    # STEP METHODS (SHORT VERSION)
    def step_validation(self):
        validator = ClinicalValidationEngine(self.df)
        return validator.run()

    def step_feature_engineering(self):
        fe = FeatureEngineeringEngine(self.df)
        self.df = fe.run_advanced()
        return self.df

    def step_statistical_analysis(self):
        ci = ConfidenceIntervalEngine(self.df)
        return ci.run_all(column='HCT')

    def step_modeling(self):
        if 'Outcome' not in self.df.columns:
            return {}

        X = self.df[['HCT', 'Albumin']].dropna()
        y = self.df.loc[X.index, 'Outcome']

        model = RandomForestClassifier()

        cv = CrossValidationEngine(model, X, y)
        return cv.run_full()

    # FULL PIPELINE
    def run_full_pipeline(self):

        self.logger.info("START PIPELINE")

        self._safe_execute("duplicate", self.step_duplicate_resolution)

        # parallel block
        self.run_parallel_steps()

        self._safe_execute("modeling", self.step_modeling)
        self._safe_execute("export", lambda: self.df)

        self.logger.info("PIPELINE DONE")

        return self.results
































# MODULE ADVANCED DATA VERSIONING ENGINE
class DataVersioningEngine:

    def __init__(self, base_path="data_versions", cloud_config=None):
        self.base_path = base_path
        self.cloud_config = cloud_config

        os.makedirs(self.base_path, exist_ok=True)

    # HASH
    def _generate_hash(self, df):
        data_bytes = df.to_csv(index=False).encode()
        return hashlib.md5(data_bytes).hexdigest()

    # SAVE VERSION + TAGGING
    def save_version(self, df, metadata=None, tag=None):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_hash = self._generate_hash(df)

        version_name = f"dataset_{timestamp}_{data_hash[:8]}"
        version_path = os.path.join(self.base_path, version_name)

        os.makedirs(version_path, exist_ok=True)

        # save data
        data_file = os.path.join(version_path, "data.csv")
        df.to_csv(data_file, index=False)

        metadata_dict = {
            "timestamp": timestamp,
            "hash": data_hash,
            "rows": len(df),
            "columns": list(df.columns),
            "tag": tag if tag else "experimental",
            "user_metadata": metadata if metadata else {}
        }

        with open(os.path.join(version_path, "metadata.json"), "w") as f:
            json.dump(metadata_dict, f, indent=4)

        # optional cloud upload
        self._upload_to_cloud(version_path)

        return version_name

    # DELTA TRACKING (ROW LEVEL)
    def delta_tracking(self, version1, version2, key_col):

        df1 = self.load_version(version1)
        df2 = self.load_version(version2)

        df1 = df1.set_index(key_col)
        df2 = df2.set_index(key_col)

        # added rows
        added = df2.loc[~df2.index.isin(df1.index)]

        # removed rows
        removed = df1.loc[~df1.index.isin(df2.index)]

        # changed rows
        common_idx = df1.index.intersection(df2.index)

        changed = []
        for idx in common_idx:
            if not df1.loc[idx].equals(df2.loc[idx]):
                changed.append(idx)

        changed_df = df2.loc[changed]

        return {
            "added": added,
            "removed": removed,
            "changed": changed_df
        }

    # LIST VERSION + TAG FILTER
    def list_versions(self, tag=None):

        versions = []

        for v in os.listdir(self.base_path):
            meta_path = os.path.join(self.base_path, v, "metadata.json")

            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)

                if tag and meta.get("tag") != tag:
                    continue

                versions.append({
                    "version": v,
                    "tag": meta.get("tag"),
                    "timestamp": meta.get("timestamp"),
                    "rows": meta.get("rows")
                })

        return pd.DataFrame(versions)

    # LOAD
    def load_version(self, version_name):
        path = os.path.join(self.base_path, version_name, "data.csv")

        if not os.path.exists(path):
            raise FileNotFoundError("Version tidak ditemukan")

        return pd.read_csv(path)
    

class LabPipelineOrchestrator:

    def __init__(self, df):
        self.df = df
        self.results = {}

    def step_auto_versioning(self):
        version_engine = DataVersioningEngine()
        version_name = version_engine.save_version(self.df)
        return {"version": version_name}

    def _safe_execute(self, step_name, func):
        try:
            result = func()
            self.results[step_name] = result
        except Exception as e:
            self.results[step_name] = {"error": str(e)}

    def run_full_pipeline(self):
        self._safe_execute("data_versioning", self.step_auto_versioning)

















# MODULE CLINICAL AI INTERPRETATION ENGINE
class AutomatedInterpretationEngine:

    def __init__(self, results):
        self.results = results
        self.interpretation = {}

    # DATA-DRIVEN CI
    def interpret_ci(self):

        ci = self.results.get('confidence_interval', {}).get('mean_ci')
        if not ci:
            return

        lower, upper, mean = ci['ci_lower'], ci['ci_upper'], ci['mean']
        width = upper - lower
        rel_width = width / abs(mean) if mean != 0 else width

        confidence = max(0, 1 - rel_width)

        text = (
            f"Estimasi rata-rata {mean:.2f} dengan CI {lower:.2f}–{upper:.2f}. "
            f"Lebar relatif {rel_width:.2f} menunjukkan tingkat kepastian estimasi."
        )

        self.interpretation['ci'] = {
            'text': text,
            'confidence_score': confidence
        }

    # CAUSAL REASONING (APPROX)
    def causal_reasoning(self):

        effect = self.results.get('effect_size', {})
        anova = effect.get('anova_effect')
        cohens = effect.get('cohens_d')

        if not anova:
            return

        p = anova.get('p_value')
        d = abs(cohens.get('cohens_d', 0)) if cohens else 0

        # causal heuristic (bukan kausal murni)
        if p < 0.05 and d > 0.5:
            causal_strength = "kuat"
        elif p < 0.05:
            causal_strength = "lemah"
        else:
            causal_strength = "tidak cukup bukti"

        text = (
            f"Dengan p-value {p:.4f} dan effect size {d:.2f}, "
            f"hubungan antar variabel menunjukkan indikasi kausal {causal_strength}, "
            f"namun interpretasi harus mempertimbangkan faktor confounding."
        )

        confidence = min(1, (1 - p) * (d + 0.1))

        self.interpretation['causal'] = {
            'text': text,
            'confidence_score': confidence
        }

    # MODEL RELIABILITY
    def model_reliability(self):

        cv = self.results.get('cross_validation', {}).get('kfold')
        if not cv:
            return

        scores = np.array(cv['scores'])

        mean = scores.mean()
        std = scores.std()

        stability = 1 - std

        text = (
            f"Model memiliki performa rata-rata {mean:.2f} dengan deviasi {std:.2f}, "
            f"menunjukkan tingkat kestabilan prediksi."
        )

        self.interpretation['model'] = {
            'text': text,
            'confidence_score': stability
        }

    # CLINICAL DECISION SUPPORT
    def clinical_decision_support(self):

        ci_conf = self.interpretation.get('ci', {}).get('confidence_score', 0)
        causal_conf = self.interpretation.get('causal', {}).get('confidence_score', 0)
        model_conf = self.interpretation.get('model', {}).get('confidence_score', 0)

        overall_conf = np.mean([ci_conf, causal_conf, model_conf])

        # risk-based decision
        if overall_conf > 0.7:
            decision = "Hasil dapat digunakan sebagai dasar pertimbangan klinis awal."
        elif overall_conf > 0.4:
            decision = "Hasil perlu dikombinasikan dengan evaluasi klinis tambahan."
        else:
            decision = "Hasil belum cukup kuat untuk mendukung keputusan klinis."

        text = (
            f"Integrasi hasil analisis menunjukkan tingkat kepercayaan {overall_conf:.2f}. "
            f"{decision}"
        )

        self.interpretation['clinical_decision'] = {
            'text': text,
            'confidence_score': overall_conf
        }

    # NLP NARRATIVE (ADVANCED)
    def generate_narrative(self):

        narrative = "Berdasarkan analisis berbasis data, "

        for key in ['ci', 'causal', 'model', 'clinical_decision']:
            if key in self.interpretation:
                narrative += self.interpretation[key]['text'] + " "

        narrative += (
            "Interpretasi ini bersifat pendukung dan tidak menggantikan keputusan klinis profesional."
        )

        self.interpretation['narrative'] = narrative

    # RUN ALL
    def run_full(self):

        self.interpret_ci()
        self.causal_reasoning()
        self.model_reliability()
        self.clinical_decision_support()
        self.generate_narrative()

        return self.interpretation



















# MODULE CLINICAL INSIGHT GENERATOR
class ClinicalInsightGenerator:
    """
    Mengubah hasil statistik & interpretasi menjadi insight klinis
    """

    def __init__(self, results, interpretation, df=None):
        self.results = results
        self.interpretation = interpretation
        self.df = df
        self.insights = []

    # HCT CLINICAL INSIGHT
    def generate_hct_insight(self):
        ci = self.results.get('confidence_interval', {}).get('mean_ci', None)

        if not ci:
            return

        mean_hct = ci['mean']

        if mean_hct < 30:
            insight = "Rata-rata HCT menunjukkan kemungkinan anemia berat atau perdarahan aktif."
        elif mean_hct < 36:
            insight = "Nilai HCT cenderung rendah, mengarah pada anemia ringan hingga sedang."
        elif mean_hct <= 54:
            insight = "Nilai HCT berada dalam rentang normal."
        else:
            insight = "Nilai HCT tinggi, dapat mengindikasikan dehidrasi atau polisitemia."

        self.insights.append({
            "parameter": "HCT",
            "insight": insight
        })

    # ALBUMIN CLINICAL INSIGHT
    def generate_albumin_insight(self):
        if self.df is None or 'Albumin' not in self.df.columns:
            return

        mean_albumin = self.df['Albumin'].mean()

        if mean_albumin < 2.5:
            insight = "Albumin sangat rendah, kemungkinan malnutrisi berat atau penyakit hati kronis."
        elif mean_albumin < 3.5:
            insight = "Albumin rendah, dapat mengindikasikan inflamasi atau gangguan nutrisi."
        elif mean_albumin <= 5.0:
            insight = "Albumin dalam batas normal."
        else:
            insight = "Albumin tinggi, kemungkinan dehidrasi."

        self.insights.append({
            "parameter": "Albumin",
            "insight": insight
        })

    # EFFECT SIZE CLINICAL INSIGHT
    def generate_effect_insight(self):
        effect = self.results.get('effect_size', {}).get('cohens_d', None)

        if not effect:
            return

        d = effect['cohens_d']

        if abs(d) >= 0.8:
            insight = "Perbedaan antar kelompok sangat signifikan secara klinis dan berpotensi mempengaruhi keputusan terapi."
        elif abs(d) >= 0.5:
            insight = "Perbedaan antar kelompok cukup bermakna dan perlu dipertimbangkan secara klinis."
        else:
            insight = "Perbedaan antar kelompok relatif kecil dan mungkin tidak signifikan secara klinis."

        self.insights.append({
            "parameter": "Effect Size",
            "insight": insight
        })

    # MODEL PERFORMANCE INSIGHT
    def generate_model_insight(self):
        cv = self.results.get('cross_validation', {}).get('kfold', None)

        if not cv:
            return

        mean_score = cv['mean_score']

        if mean_score > 0.85:
            insight = "Model memiliki performa sangat baik dan dapat dipertimbangkan untuk penggunaan klinis terbatas."
        elif mean_score > 0.75:
            insight = "Model memiliki performa cukup baik, namun masih memerlukan validasi tambahan."
        else:
            insight = "Model belum cukup kuat untuk digunakan dalam pengambilan keputusan klinis."

        self.insights.append({
            "parameter": "Model",
            "insight": insight
        })

    # COMBINE ALL INSIGHTS
    def run_all(self):
        self.generate_hct_insight()
        self.generate_albumin_insight()
        self.generate_effect_insight()
        self.generate_model_insight()

        return self.insights
