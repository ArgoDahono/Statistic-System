"""
GUARD SYSTEM INTEGRATION EXAMPLE
Demonstrates how to integrate the Guard System with Trial.py

This module shows:
1. Loading data from CSV
2. Running all guards
3. Separating valid and invalid records
4. Generating validation reports
"""

import pandas as pd
import sys
from pathlib import Path

# Add guards package to path
sys.path.insert(0, str(Path(__file__).parent))

from guards import GuardManager, create_guard_manager


def load_and_validate_data(csv_filepath: str, mode: str = 'STRICT'):
    """
    Load CSV data and validate using Guard System
    
    Args:
        csv_filepath: Path to CSV file
        mode: 'STRICT' or 'PERMISSIVE'
        
    Returns:
        Tuple of (valid_df, invalid_df, validation_result, manager)
    """
    
    # Load data
    print(f"\n[*] Loading data from: {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    print(f"[✓] Loaded {len(df)} records")
    
    # Create guard manager
    print(f"\n[*] Initializing Guard Manager (Mode: {mode})")
    manager = create_guard_manager()
    manager.set_validation_mode(mode)
    
    # Validate records
    print(f"\n[*] Validating records through all guards...")
    valid_df, invalid_df, validation_result = manager.validate_dataframe(df)
    
    # Print results
    manager.print_summary(validation_result)
    
    return valid_df, invalid_df, validation_result, manager


def save_validation_results(valid_df: pd.DataFrame, invalid_df: pd.DataFrame, 
                           manager: GuardManager, output_dir: str = '.'):
    """
    Save validation results to files
    
    Args:
        valid_df: DataFrame with valid records
        invalid_df: DataFrame with invalid records
        manager: GuardManager instance
        output_dir: Directory to save files
    """
    
    print(f"\n[*] Saving validation results to {output_dir}...")
    
    # Create output directory if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save valid records
    valid_file = Path(output_dir) / 'valid_records.csv'
    valid_df.to_csv(valid_file, index=False, encoding='utf-8')
    print(f"[✓] Valid records saved: {valid_file} ({len(valid_df)} records)")
    
    # Save invalid records
    if len(invalid_df) > 0:
        invalid_file = Path(output_dir) / 'invalid_records.csv'
        invalid_df.to_csv(invalid_file, index=False, encoding='utf-8')
        print(f"[✓] Invalid records saved: {invalid_file} ({len(invalid_df)} records)")
    else:
        print(f"[✓] No invalid records")
    
    # Save detailed report
    report_file = Path(output_dir) / 'validation_report.json'
    manager.save_report(str(report_file))
    print(f"[✓] Validation report saved: {report_file}")


def integrate_with_trial_processing(valid_df: pd.DataFrame):
    """
    Example function showing how to use validated data with Trial.py processing
    
    This would be called after guard validation to ensure only valid data
    is passed to subsequent processing pipelines.
    
    Args:
        valid_df: Validated DataFrame
        
    Returns:
        Processed data ready for Trial.py
    """
    
    print(f"\n[*] Processing {len(valid_df)} validated records for Trial system...")
    
    # Example: Group by patient and examine type
    grouped = valid_df.groupby('No_RM').size()
    print(f"[✓] Grouped into {len(grouped)} unique patients")
    
    # Example: Get exam types distribution
    exam_dist = valid_df['Jenis_Pemeriksaan'].value_counts()
    print(f"[✓] Exam types distribution:")
    for exam_type, count in exam_dist.items():
        print(f"   - {exam_type}: {count}")
    
    # Example: Get parameter distribution
    param_dist = valid_df['Parameter'].value_counts()
    print(f"[✓] Top 10 parameters:")
    for param, count in param_dist.head(10).items():
        print(f"   - {param}: {count}")
    
    return valid_df


def main():
    """Main integration function"""
    
    print("=" * 80)
    print("GUARD SYSTEM INTEGRATION WITH TRIAL.PY")
    print("=" * 80)
    
    # File paths
    csv_file = 'data_rekam_medis_dbd_300_long_format_standar.csv'
    output_dir = 'guard_validation_output'
    
    # Validate data
    try:
        valid_df, invalid_df, validation_result, manager = load_and_validate_data(
            csv_file, 
            mode='STRICT'
        )
    except Exception as e:
        print(f"[✗] Error during validation: {e}")
        return
    
    # Save results
    save_validation_results(valid_df, invalid_df, manager, output_dir)
    
    # Process validated data
    processed_df = integrate_with_trial_processing(valid_df)
    
    print(f"\n[✓] Integration complete!")
    print(f"[✓] Ready to pass to Trial.py processing pipeline")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review validated records in: guard_validation_output/valid_records.csv")
    print("2. Check any invalid records in: guard_validation_output/invalid_records.csv")
    print("3. Review detailed report in: guard_validation_output/validation_report.json")
    print("4. Use valid_df from this script in your Trial.py processing")
    print("=" * 80)
    
    return valid_df, invalid_df, validation_result, manager


if __name__ == '__main__':
    valid_df, invalid_df, validation_result, manager = main()
