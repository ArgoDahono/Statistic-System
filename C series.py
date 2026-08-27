import numpy as np
import json
import time

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import field

# Import C++ module
try:
    import ctypes as plc  # Pastikan nama modul sesuai dengan yang dihasilkan oleh pybind11
    # Cek apakah modul memiliki class yang dibutuhkan
    if hasattr(plc, 'PlasmaLeakageProcessor'):
        CPP_MODULE_AVAILABLE = True
        print("[Python] C++ module loaded successfully")
    elif hasattr(plc, 'PlasmaLeakageMonteCarlo'):
        # Handle kasus dimana nama class berbeda
        CPP_MODULE_AVAILABLE = True
        plc.PlasmaLeakageProcessor = plc.PlasmaLeakageMonteCarlo
        print("[Python] C++ module loaded successfully (PlasmaLeakageMonteCarlo found)")
    else:
        print("[Python] C++ module loaded but PlasmaLeakageProcessor not found")
        print("[Python] Available classes:", dir(plc))
except ImportError as e:
    print(f"[Python] Warning: C++ module not found: {e}")
    print("[Python] Using Python fallback simulation (pure Python)")
except Exception as e:
    print(f"[Python] Error loading C++ module: {e}")
    print("[Python] Using Python fallback simulation")

# DATA CLASSES
class SimulationConfig:
    """Konfigurasi simulasi"""
    monte_carlo_iterations: int = 1000000
    use_onnx: bool = False
    onnx_model_path: str = ""
    custom_params: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'monte_carlo_iterations': self.monte_carlo_iterations,
            'use_onnx': self.use_onnx,
            'onnx_model_path': self.onnx_model_path,
            'custom_params': self.custom_params
        }

class RiskAssessment:
    """Hasil risk assessment"""
    risk_score: float
    risk_level: str
    probability_critical: float
    expected_leakage: float
    confidence_interval: Tuple[float, float]
    risk_factors: Dict[str, float]
    processing_time_ms: float
    status: str
    
    def is_critical(self) -> bool:
        return self.risk_level == "CRITICAL"
    
    def is_high_risk(self) -> bool:
        return self.risk_level in ["HIGH", "CRITICAL"]
    
    def get_recommendation(self) -> str:
        recommendations = {
            "LOW": "Routine monitoring. No immediate intervention required.",
            "MODERATE": "Increase monitoring frequency. Consider prophylactic treatment.",
            "HIGH": "IMMEDIATE CLINICAL REVIEW REQUIRED. Consider aggressive intervention.",
            "CRITICAL": "EMERGENCY: Immediate intervention required. Activate emergency protocols."
        }
        return recommendations.get(self.risk_level, "Unknown risk level")

class SimulationResult:
    """Hasil simulasi lengkap"""
    # Statistics
    expected_leakage: float
    variance_leakage: float
    std_leakage: float
    min_leakage: float
    max_leakage: float
    
    # Percentiles
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    
    # Risk
    overall_risk_score: float
    probability_critical: float
    risk_level: str
    risk_factors: Dict[str, float]
    
    # Samples
    leakage_samples: List[float]
    risk_samples: List[float]
    
    # Distribution
    distribution: List[List[float]]
    
    # Metadata
    status: str
    message: str
    processing_time_ms: float
    total_iterations: int
    
    # Original input (for reference)
    input_data: Optional[np.ndarray] = None
    def from_dict(cls, data: Dict[str, Any], input_data: Optional[np.ndarray] = None) -> 'SimulationResult':
        """Buat SimulationResult dari dictionary"""
        return cls(
            expected_leakage=data.get('expected_leakage', 0),
            variance_leakage=data.get('variance_leakage', 0),
            std_leakage=data.get('std_leakage', 0),
            min_leakage=data.get('min_leakage', 0),
            max_leakage=data.get('max_leakage', 0),
            p5=data.get('p5', 0),
            p25=data.get('p25', 0),
            p50=data.get('p50', 0),
            p75=data.get('p75', 0),
            p95=data.get('p95', 0),
            overall_risk_score=data.get('overall_risk_score', 0),
            probability_critical=data.get('probability_critical', 0),
            risk_level=data.get('risk_level', 'UNKNOWN'),
            risk_factors=data.get('risk_factors', {}),
            leakage_samples=data.get('leakage_samples', []),
            risk_samples=data.get('risk_samples', []),
            distribution=data.get('distribution', []),
            status=data.get('status', 'UNKNOWN'),
            message=data.get('message', ''),
            processing_time_ms=data.get('processing_time_ms', 0),
            total_iterations=data.get('total_iterations', 0),
            input_data=input_data
        )
    
    def to_risk_assessment(self) -> RiskAssessment:
        """Konversi ke RiskAssessment"""
        return RiskAssessment(
            risk_score=self.overall_risk_score,
            risk_level=self.risk_level,
            probability_critical=self.probability_critical,
            expected_leakage=self.expected_leakage,
            confidence_interval=(self.p5, self.p95),
            risk_factors=self.risk_factors,
            processing_time_ms=self.processing_time_ms,
            status=self.status
        )

# MAIN CLIENT CLASS
class PlasmaLeakageClient:
    """
    Main client untuk komunikasi Python-C++
    
    Usage:
        client = PlasmaLeakageClient()
        
        # Kirim numpy array
        result = client.send_numpy(data)
        
        # Atau kirim dictionary
        result = client.send_dict({'values': [1, 2, 3, ...]})
        
        # Ambil risk assessment
        assessment = result.to_risk_assessment()
        print(assessment.get_recommendation())
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize client
        
        Args:
            config: Konfigurasi simulasi
        """
        self.config = config or SimulationConfig()
        self.processor = None
        self.use_cpp = False
        
        # Initialize C++ processor if available
        try:
            CPP_MODULE_AVAILABLE = True
        except ImportError:
            CPP_MODULE_AVAILABLE = False

        if CPP_MODULE_AVAILABLE:
            try:
                self.processor = plc.PlasmaLeakageProcessor()
                self.use_cpp = True
                print("[Python] Using C++ processor")
            except Exception as e:
                print(f"[Python] Failed to initialize C++ processor: {e}")
                self.use_cpp = False
        
        if not self.use_cpp:
            print("[Python] Using Python fallback processor")
    
    # DATA SENDING METHODS    
    def send_numpy(self, 
                   data: Union[np.ndarray, List[float]], 
                   iterations: Optional[int] = None) -> SimulationResult:
        """
        Kirim data numpy array ke C++
        
        Args:
            data: Array data (akan ditruncate ke 200 jika > 200)
            iterations: Jumlah iterasi Monte Carlo (default dari config)
            
        Returns:
            SimulationResult: Hasil simulasi
        """
        # Konversi ke numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        
        # Simpan input asli
        original_size = len(data)
        
        # Truncate jika > 200
        if len(data) > 200:
            data = data[:200]
            print(f"[Python] Warning: Data truncated from {original_size} to 200")
        
        # Validasi
        if len(data) == 0:
            return self._create_error_result("Empty data array")
        
        # Set iterations
        iters = iterations or self.config.monte_carlo_iterations
        
        # Kirim ke C++ atau Python fallback
        if self.use_cpp:
            try:
                result = self.processor.process_numpy(data, iters)
                return self._parse_result(result, data)
            except Exception as e:
                print(f"[Python] C++ processing failed: {e}")
                return self._python_fallback(list(data), iters)
        else:
            return self._python_fallback(list(data), iters)
    
    def send_dict(self, data_dict: Dict[str, Any]) -> SimulationResult:
        """
        Kirim data dalam format dictionary
        
        Args:
            data_dict: Dictionary dengan format:
                {
                    'values': [list of values],  # WAJIB
                    'mean': float,               # Opsional
                    'sd': float,                 # Opsional
                    'baseline': float,          # Opsional
                    'iterations': int,          # Opsional
                    'use_onnx': bool,            # Opsional
                    'onnx_model_path': str,     # Opsional
                    'custom_params': dict       # Opsional
                }
                
        Returns:
            SimulationResult: Hasil simulasi
        """
        # Validasi values
        if 'values' not in data_dict:
            return self._create_error_result("No 'values' key in dictionary")
        
        values = data_dict['values']
        if not isinstance(values, (list, np.ndarray)):
            return self._create_error_result("'values' must be list or array")
        
        # Convert to list
        values = list(values)
        if len(values) == 0:
            return self._create_error_result("Empty values list")
        
        # Truncate to 200
        original_size = len(values)
        if len(values) > 200:
            values = values[:200]
            print(f"[Python] Warning: Data truncated from {original_size} to 200")
        
        # Parse konfigurasi
        input_dict = {
            'values': values,
            'mean': data_dict.get('mean', 0.0),
            'sd': data_dict.get('sd', 0.0),
            'baseline': data_dict.get('baseline', 0.0),
            'monte_carlo_iterations': data_dict.get('iterations', 
                self.config.monte_carlo_iterations),
            'use_onnx': data_dict.get('use_onnx', self.config.use_onnx),
            'onnx_model_path': data_dict.get('onnx_model_path', 
                self.config.onnx_model_path),
            'custom_params': data_dict.get('custom_params', 
                self.config.custom_params)
        }
        
        # Kirim ke C++ atau Python fallback
        if self.use_cpp:
            try:
                result = self.processor.process(input_dict)
                return self._parse_result(result, np.array(values))
            except Exception as e:
                print(f"[Python] C++ processing failed: {e}")
                return self._python_fallback(values, 
                    input_dict['monte_carlo_iterations'])
        else:
            return self._python_fallback(values, 
                input_dict['monte_carlo_iterations'])
    
    def send_csv(self, 
                 csv_path: str, 
                 delimiter: str = ',',
                 iterations: Optional[int] = None) -> SimulationResult:
        """
        Kirim data dari file CSV
        
        Args:
            csv_path: Path ke file CSV
            delimiter: Delimiter CSV (default: ',')
            iterations: Jumlah iterasi
            
        Returns:
            SimulationResult: Hasil simulasi
        """
        try:
            # Load CSV
            data = np.loadtxt(csv_path, delimiter=delimiter)
            
            # Handle single column
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Use first column
            data = data[:, 0]
            
            print(f"[Python] Loaded {len(data)} points from {csv_path}")
            return self.send_numpy(data, iterations)
            
        except Exception as e:
            return self._create_error_result(f"Failed to load CSV: {e}")
    
    def send_json(self, 
                  json_path: str,
                  iterations: Optional[int] = None) -> SimulationResult:
        """
        Kirim data dari file JSON
        
        Args:
            json_path: Path ke file JSON
            iterations: Jumlah iterasi
            
        Returns:
            SimulationResult: hasil simulasi
        """
        try:
            with open(json_path, 'r') as f:
                data_dict = json.load(f)
            
            # Add iterations if provided
            if iterations:
                data_dict['iterations'] = iterations
            
            return self.send_dict(data_dict)
            
        except Exception as e:
            return self._create_error_result(f"Failed to load JSON: {e}")
    
    def send_dataframe(self, 
                       df,
                       column: str = None,
                       iterations: Optional[int] = None) -> SimulationResult:
        """
        Kirim data dari pandas DataFrame
        
        Args:
            df: pandas DataFrame
            column: Nama kolom yang akan digunakan (jika None, gunakan kolom pertama)
            iterations: Jumlah iterasi
            
        Returns:
            SimulationResult: Hasil simulasi
        """
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return self._create_error_result("Input must be pandas DataFrame")
            
            # Select column
            if column:
                if column not in df.columns:
                    return self._create_error_result(f"Column '{column}' not found")
                data = df[column].values
            else:
                data = df.iloc[:, 0].values
            
            return self.send_numpy(data, iterations)
            
        except ImportError:
            return self._create_error_result("pandas not installed")
        except Exception as e:
            return self._create_error_result(f"DataFrame error: {e}")
    
    # RESULT PARSING METHODS    
    def _parse_result(self, 
                     result, 
                     input_data: np.ndarray) -> SimulationResult:
        """
        Parse hasil dari C++ ke SimulationResult
        
        Args:
            result: Hasil dari C++ (dict atau objek)
            input_data: Data input asli
            
        Returns:
            SimulationResult: Hasil yang sudah diparse
        """
        # Convert to dict if needed
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif hasattr(result, 'cast'):
            # py::object to dict
            result_dict = dict(result)
        else:
            result_dict = result
        
        # Handle numpy arrays in result
        parsed = {}
        for key, value in result_dict.items():
            if isinstance(value, np.ndarray):
                parsed[key] = value.tolist()
            elif hasattr(value, 'item'):  # numpy scalar
                parsed[key] = value.item()
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                # Try to convert to list
                try:
                    parsed[key] = list(value)
                except:
                    parsed[key] = value
            else:
                parsed[key] = value
        
        return SimulationResult.from_dict(parsed, input_data)
    
    def _create_error_result(self, error_message: str) -> SimulationResult:
        """Buat hasil error"""
        return SimulationResult(
            expected_leakage=0,
            variance_leakage=0,
            std_leakage=0,
            min_leakage=0,
            max_leakage=0,
            p5=0, p25=0, p50=0, p75=0, p95=0,
            overall_risk_score=0,
            probability_critical=0,
            risk_level="ERROR",
            risk_factors={},
            leakage_samples=[],
            risk_samples=[],
            distribution=[],
            status="ERROR",
            message=error_message,
            processing_time_ms=0,
            total_iterations=0
        )
    
    # PYTHON FALLBACK 
    def _python_fallback(self, 
                        data: List[float], 
                        iterations: int) -> SimulationResult:
        """
        Python fallback simulation (jika C++ tidak tersedia)
        
        Args:
            data: List data
            iterations: Jumlah iterasi
            
        Returns:
            SimulationResult: Hasil simulasi
        """
        print(f"[Python] Running Python fallback simulation...")
        start_time = time.time()
        
        # Calculate statistics
        data_arr = np.array(data)
        mean = np.mean(data_arr)
        sd = np.std(data_arr)
        
        # Bootstrap simulation
        np.random.seed(42)
        leakage_results = []
        
        for i in range(iterations):
            # Bootstrap sample
            sample = np.random.choice(data_arr, size=len(data_arr), replace=True)
            
            # Add stochastic variation
            variation = np.random.normal(0, sd * 0.1)
            
            # Calculate leakage (simplified model)
            base_leakage = np.mean(sample) + variation
            leakage = max(0, min(100, base_leakage))
            
            leakage_results.append(leakage)
        
        # Calculate statistics
        leakage_arr = np.array(leakage_results)
        
        result_dict = {
            'expected_leakage': float(np.mean(leakage_arr)),
            'variance_leakage': float(np.var(leakage_arr)),
            'std_leakage': float(np.std(leakage_arr)),
            'min_leakage': float(np.min(leakage_arr)),
            'max_leakage': float(np.max(leakage_arr)),
            'p5': float(np.percentile(leakage_arr, 5)),
            'p25': float(np.percentile(leakage_arr, 25)),
            'p50': float(np.percentile(leakage_arr, 50)),
            'p75': float(np.percentile(leakage_arr, 75)),
            'p95': float(np.percentile(leakage_arr, 95)),
            'skewness': 0.0,
            'kurtosis': 0.0,
            'overall_risk_score': float(np.mean(leakage_arr) * 2),
            'probability_critical': float(np.mean(leakage_arr > 15)),
            'risk_level': 'LOW' if np.mean(leakage_arr) * 2 < 30 else 'MODERATE',
            'risk_factors': {
                'severity': float(np.mean(leakage_arr) * 2),
                'probability_critical': float(np.mean(leakage_arr > 15) * 100),
                'uncertainty': float((np.percentile(leakage_arr, 95) - np.percentile(leakage_arr, 5)) / 10),
                'variability': float(np.std(leakage_arr) / np.mean(leakage_arr) * 100),
                'skewness_risk': 0.0
            },
            'leakage_samples': [float(x) for x in leakage_arr[::len(leakage_arr)//10000][:10000]],
            'risk_samples': [float(min(100, x * 2)) for x in leakage_arr[::len(leakage_arr)//10000][:10000]],
            'distribution': [],
            'status': 'SUCCESS',
            'message': 'Python fallback simulation completed',
            'processing_time_ms': (time.time() - start_time) * 1000,
            'total_iterations': iterations
        }
        
        return SimulationResult.from_dict(result_dict, np.array(data))
    
    # RISK ANALYSIS METHODS    
    def analyze_risk(self, 
                    data: Union[np.ndarray, List[float], Dict]) -> RiskAssessment:
        """
        Langsung analisis risiko dari data
        
        Args:
            data: Data input (numpy array, list, atau dict)
            
        Returns:
            RiskAssessment: Hasil risk assessment
        """
        # Process data
        if isinstance(data, dict):
            result = self.send_dict(data)
        else:
            result = self.send_numpy(data)
        
        # Convert to risk assessment
        return result.to_risk_assessment()
    
    def get_risk_report(self, result: SimulationResult) -> str:
        """
        Generate risk report dari hasil simulasi
        
        Args:
            result: Hasil simulasi
            
        Returns:
            str: Risk report dalam format teks
        """
        assessment = result.to_risk_assessment()
        
        report = f"""

           PLASMA LEAKAGE RISK ANALYSIS REPORT
OVERALL ASSESSMENT
Risk Score: {assessment.risk_score:.2f}/100
Risk Level: {assessment.risk_level}
Probability of Critical Leakage: {assessment.probability_critical * 100:.2f}%
Expected Leakage: {assessment.expected_leakage:.2f}%
90% Confidence Interval: [{assessment.confidence_interval[0]:.2f}%, {assessment.confidence_interval[1]:.2f}%]

RISK FACTORS
"""
        for factor, value in assessment.risk_factors.items():
            report += f"{factor}: {value:.2f}\n"
        
        report += f"""
RECOMMENDATION
{assessment.get_recommendation()}

PROCESSING INFO
Processing Time: {result.processing_time_ms:.2f} ms
Total Iterations: {result.total_iterations}
Status: {result.status}

"""
        return report
    
    # PARAMETER MANAGEMENT    
    def set_parameter(self, name: str, value: float) -> bool:
        """
        Set parameter model
        
        Args:
            name: Nama parameter
            value: Nilai parameter
            
        Returns:
            bool: Berhasil atau tidak
        """
        if self.use_cpp:
            try:
                self.processor.set_parameter(name, value)
                return True
            except Exception as e:
                print(f"[Python] Failed to set parameter: {e}")
                return False
        else:
            self.config.custom_params[name] = value
            return True
    
    def get_parameter(self, name: str) -> Optional[float]:
        """
        Get parameter model
        
        Args:
            name: Nama parameter
            
        Returns:
            float: Nilai parameter atau None
        """
        if self.use_cpp:
            try:
                return self.processor.get_parameter(name)
            except:
                return None
        else:
            return self.config.custom_params.get(name)
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get semua parameter"""
        if self.use_cpp:
            try:
                return dict(self.processor.get_all_parameters())
            except:
                return self.config.custom_params
        else:
            return self.config.custom_params
    
    # BATCH PROCESSING    
    def process_batch(self, 
                     data_list: List[Union[np.ndarray, List[float]]],
                     iterations: Optional[int] = None) -> List[SimulationResult]:
        """
        Proses batch data
        
        Args:
            data_list: List dari data arrays
            iterations: Jumlah iterasi per simulasi
            
        Returns:
            List[SimulationResult]: Hasil simulasi untuk setiap data
        """
        results = []
        
        for i, data in enumerate(data_list):
            print(f"[Python] Processing batch {i+1}/{len(data_list)}...")
            result = self.send_numpy(data, iterations)
            results.append(result)
        
        return results
    
    def process_pandas(self, 
                      df,
                      value_column: str = None,
                      group_column: str = None,
                      iterations: Optional[int] = None) -> Union[List[SimulationResult], Dict]:
        """
        Proses data dari pandas DataFrame
        
        Args:
            df: pandas DataFrame
            value_column: Kolom nilai (jika None, gunakan semua numerik)
            group_column: Kolom grouping (jika ada)
            iterations: Jumlah iterasi
            
        Returns:
            List[SimulationResult] atau Dict dengan group keys
        """
        try:          
            if group_column:
                # Group processing
                results = {}
                for name, group in df.groupby(group_column):
                    if value_column:
                        data = group[value_column].values
                    else:
                        data = group.select_dtypes(include=[np.number]).values.flatten()
                    
                    print(f"[Python] Processing group '{name}'...")
                    results[name] = self.send_numpy(data, iterations)
                return results
            else:
                # Single processing
                if value_column:
                    data = df[value_column].values
                else:
                    data = df.select_dtypes(include=[np.number]).values.flatten()
                
                return [self.send_numpy(data, iterations)]
                
        except ImportError:
            print("[Python] pandas not installed")
            return []
        except Exception as e:
            print(f"[Python] Error processing DataFrame: {e}")
            return []
    
    # SAVE/LOAD METHODS    
    def save_result(self, 
                   result: SimulationResult, 
                   filepath: str,
                   format: str = 'json') -> bool:
        """
        Simpan hasil ke file
        
        Args:
            result: Hasil simulasi
            filepath: Path file output
            format: Format ('json', 'csv', 'pickle')
            
        Returns:
            bool: Berhasil atau tidak
        """
        try:
            if format == 'json':
                data = {
                    'statistics': {
                        'expected_leakage': result.expected_leakage,
                        'variance_leakage': result.variance_leakage,
                        'std_leakage': result.std_leakage,
                        'min_leakage': result.min_leakage,
                        'max_leakage': result.max_leakage,
                    },
                    'percentiles': {
                        'p5': result.p5,
                        'p25': result.p25,
                        'p50': result.p50,
                        'p75': result.p75,
                        'p95': result.p95,
                    },
                    'risk': {
                        'overall_risk_score': result.overall_risk_score,
                        'probability_critical': result.probability_critical,
                        'risk_level': result.risk_level,
                        'risk_factors': result.risk_factors,
                    },
                    'metadata': {
                        'status': result.status,
                        'message': result.message,
                        'processing_time_ms': result.processing_time_ms,
                        'total_iterations': result.total_iterations,
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            elif format == 'csv':
                # Save samples
                import pandas as pd
                df = pd.DataFrame({
                    'leakage': result.leakage_samples,
                    'risk': result.risk_samples
                })
                df.to_csv(filepath, index=False)
                
            elif format == 'pickle':
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(result, f)
            
            print(f"[Python] Result saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"[Python] Failed to save result: {e}")
            return False
    
    def load_result(self, filepath: str) -> Optional[SimulationResult]:
        """
        Load hasil dari file
        
        Args:
            filepath: Path file input
            
        Returns:
            SimulationResult atau None
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
            
            print(f"[Python] Result loaded from {filepath}")
            return result
            
        except Exception as e:
            print(f"[Python] Failed to load result: {e}")
            return None

# HELPER FUNCTIONS
def quick_analysis(data: Union[List[float], np.ndarray],
                  iterations: int = 1000000) -> RiskAssessment:
    """
    Quick risk analysis function
    
    Args:
        data: Data input
        iterations: Jumlah iterasi
        
    Returns:
        RiskAssessment: Hasil risk assessment
    """
    client = PlasmaLeakageClient()
    return client.analyze_risk(data)

def generate_sample_data(n: int = 100,
                        mean: float = 50.0,
                        sd: float = 10.0) -> np.ndarray:
    """
    Generate sample data untuk testing
    
    Args:
        n: Jumlah titik data
        mean: Mean
        sd: Standard deviation
        
    Returns:
        np.ndarray: Sample data
    """
    np.random.seed(42)
    return np.random.normal(mean, sd, n)

def compare_results(results: List[SimulationResult]) -> Dict:
    """
    Bandingkan beberapa hasil simulasi
    
    Args:
        results: List hasil simulasi
        
    Returns:
        Dict: Perbandingan
    """
    comparison = {
        'count': len(results),
        'risk_scores': [r.overall_risk_score for r in results],
        'risk_levels': [r.risk_level for r in results],
        'expected_leakages': [r.expected_leakage for r in results],
        'processing_times': [r.processing_time_ms for r in results],
        'mean_risk_score': np.mean([r.overall_risk_score for r in results]),
        'std_risk_score': np.std([r.overall_risk_score for r in results]),
    }
    
    return comparison

print("[Python] PlasmaLeakageClient module loaded successfully")
print("[Python] Available methods:")
print("- send_numpy(data, iterations=None)")
print("- send_dict(data_dict)")
print("- send_csv(csv_path, delimiter=',', iterations=None)")
print("- send_json(json_path, iterations=None)")
print("- send_dataframe(df, column=None, iterations=None)")
print("- analyze_risk(data)")
print("- get_risk_report(result)")
print("- set_parameter(name, value)")
print("- get_parameter(name)")
print("- get_all_parameters()")
print("- process_batch(data_list, iterations=None)")
print("- process_pandas(df, value_column=None, group_column=None, iterations=None)")
print("- save_result(result, filepath, format='json')")
print("- load_result(filepath)")
def main():
    # Contoh penggunaan
    client = PlasmaLeakageClient()
    
    # Generate sample data
    data = generate_sample_data(n=1000, mean=50, sd=10)
    
    # Analisis risiko
    assessment = client.analyze_risk(data)
    print(client.get_risk_report(assessment.to_risk_assessment()))
if __name__ == "__main__":
    main()

print("[Python] PlasmaLeakageClient module executed as main")