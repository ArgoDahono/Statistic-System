#!/usr/bin/env python3
"""
QUICK REFERENCE GUIDE - Guard System
Panduan cepat untuk menggunakan Guard System
"""

from guards import (
    GuardManager,
    create_guard_manager,
    IdentityGuard,
    TemporalGuard,
    ClinicalGuard,
    DuplicateGuard
)
import pandas as pd

# ============================================================================
# BASIC USAGE - ValidateTuggal satu record
# ============================================================================

def example_1_single_record():
    """Example 1: Validate single record"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SINGLE RECORD VALIDATION")
    print("=" * 80)
    
    manager = create_guard_manager()
    
    record = {
        'No_RM': 'RM001',
        'Nama': 'Budi Hidayat',
        'Umur': 25,
        'Jenis_Kelamin': 'L',
        'Tanggal_Masuk': '2026-01-15',
        'Tanggal_Pemeriksaan': '2026-01-15',
        'Jenis_Pemeriksaan': 'Darah Rutin',
        'Parameter': 'WBC',
        'Hasil': 3.6,
        'Satuan': '',
        'Diagnosa': 'DBD'
    }
    
    result = manager.validate_record(record)
    
    print(f"\nNo_RM: {result['no_rm']}")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")


# ============================================================================
# BATCH USAGE - Validate multiple records
# ============================================================================

def example_2_batch_records():
    """Example 2: Validate multiple records"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: BATCH RECORD VALIDATION")
    print("=" * 80)
    
    manager = create_guard_manager()
    
    records = [
        {
            'No_RM': 'RM001', 'Nama': 'Budi Hidayat', 'Umur': 25, 'Jenis_Kelamin': 'L',
            'Tanggal_Masuk': '2026-01-15', 'Tanggal_Pemeriksaan': '2026-01-15',
            'Jenis_Pemeriksaan': 'Darah Rutin', 'Parameter': 'WBC',
            'Hasil': 3.6, 'Satuan': '', 'Diagnosa': 'DBD'
        },
        {
            'No_RM': 'RM215', 'Nama': 'Bayu Setiawan', 'Umur': 41, 'Jenis_Kelamin': 'L',
            'Tanggal_Masuk': '2026-01-01', 'Tanggal_Pemeriksaan': '2026-01-01',
            'Jenis_Pemeriksaan': 'Darah Rutin', 'Parameter': 'PLT',
            'Hasil': 116237.0, 'Satuan': '', 'Diagnosa': 'DBD'
        }
    ]
    
    result = manager.validate_records(records)
    
    print(f"\nTotal: {result.total_records}")
    print(f"Valid: {result.valid_records}")
    print(f"Invalid: {result.invalid_records}")
    print(f"Passed: {result.passed}")


# ============================================================================
# DATAFRAME USAGE - Validate CSV/DataFrame
# ============================================================================

def example_3_dataframe():
    """Example 3: Validate DataFrame from CSV"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: DATAFRAME VALIDATION")
    print("=" * 80)
    
    manager = create_guard_manager()
    manager.set_validation_mode('STRICT')
    
    # Load CSV
    df = pd.read_csv('data_rekam_medis_dbd_300_long_format_standar.csv')
    print(f"\nLoaded {len(df)} records from CSV")
    
    # Validate
    valid_df, invalid_df, result = manager.validate_dataframe(df)
    
    print(f"Valid: {len(valid_df)}")
    print(f"Invalid: {len(invalid_df)}")
    
    # Save results
    valid_df.to_csv('valid_records.csv', index=False)
    invalid_df.to_csv('invalid_records.csv', index=False)
    manager.save_report('validation_report.json')
    
    print("\nSaved to:")
    print("  - valid_records.csv")
    print("  - invalid_records.csv")
    print("  - validation_report.json")


# ============================================================================
# INDIVIDUAL GUARD USAGE
# ============================================================================

def example_4_identity_guard():
    """Example 4: Use Identity Guard directly"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: IDENTITY GUARD")
    print("=" * 80)
    
    guard = IdentityGuard()
    
    # Test valid
    result = guard.check_identity_consistency('RM001', 'Budi Hidayat', 25, 'L')
    print(f"\nValid identity: {result.is_valid}")
    
    # Test invalid (bad RM format)
    result = guard.check_identity_consistency('INVALID', 'Budi Hidayat', 25, 'L')
    print(f"Invalid RM: {result.is_valid}")
    print(f"Error: {result.errors[0]}")


def example_5_duplicate_guard():
    """Example 5: Check for duplicates"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: DUPLICATE GUARD")
    print("=" * 80)
    
    guard = DuplicateGuard()
    
    # Check known duplicate
    result = guard.detect_duplicates('RM215', 'Bayu Setiawan', 41)
    print(f"\nIs duplicate: {result.is_duplicate}")
    print(f"Type: {result.duplicate_type}")
    if result.warnings:
        print(f"Warning: {result.warnings[0]}")


def example_6_clinical_guard():
    """Example 6: Validate clinical data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: CLINICAL GUARD")
    print("=" * 80)
    
    guard = ClinicalGuard()
    
    # Valid
    result = guard.check_clinical_consistency(
        'RM001', 'Darah Rutin', 'WBC', 3.6, '', 'DBD'
    )
    print(f"\nWBC=3.6: {result.is_valid}")
    
    # Invalid (out of range)
    result = guard.check_clinical_consistency(
        'RM002', 'Darah Rutin', 'WBC', 50.0, '', 'DBD'
    )
    print(f"WBC=50: {result.is_valid}")
    print(f"Warning: {result.warnings[0] if result.warnings else 'None'}")


def example_7_temporal_guard():
    """Example 7: Validate dates"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: TEMPORAL GUARD")
    print("=" * 80)
    
    guard = TemporalGuard()
    
    # Valid sequence
    result = guard.check_temporal_consistency('RM001', '2026-01-15', '2026-01-15')
    print(f"\nValid sequence: {result.is_valid}")
    print(f"Days: {result.metadata['days_since_admission']}")
    
    # Invalid sequence (exam before admission)
    result = guard.check_temporal_consistency('RM002', '2026-02-20', '2026-02-10')
    print(f"Invalid sequence: {result.is_valid}")
    print(f"Error: {result.errors[0]}")


# ============================================================================
# MODES AND CONFIGURATION
# ============================================================================

def example_8_modes():
    """Example 8: STRICT vs PERMISSIVE modes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: VALIDATION MODES")
    print("=" * 80)
    
    # STRICT mode (default)
    print("\nSTRICT Mode:")
    manager = create_guard_manager()
    manager.set_validation_mode('STRICT')
    print(f"Mode: {manager.validation_mode}")
    print("- Fails if ANY error exists")
    print("- Strict data quality requirement")
    
    # PERMISSIVE mode
    print("\nPERMISSIVE Mode:")
    manager.set_validation_mode('PERMISSIVE')
    print(f"Mode: {manager.validation_mode}")
    print("- Only fails on critical errors")
    print("- Warnings are allowed")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def example_9_reports():
    """Example 9: Generate reports"""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: VALIDATION REPORTS")
    print("=" * 80)
    
    manager = create_guard_manager()
    
    # Create sample records
    records = [
        {
            'No_RM': 'RM001', 'Nama': 'Budi', 'Umur': 25, 'Jenis_Kelamin': 'L',
            'Tanggal_Masuk': '2026-01-15', 'Tanggal_Pemeriksaan': '2026-01-15',
            'Jenis_Pemeriksaan': 'Darah Rutin', 'Parameter': 'WBC',
            'Hasil': 3.6, 'Satuan': '', 'Diagnosa': 'DBD'
        }
    ]
    
    result = manager.validate_records(records)
    
    # Print summary
    print("\n")
    manager.print_summary(result)
    
    # Save detailed report
    manager.save_report('validation_report.json')
    print("\n[✓] Report saved to: validation_report.json")


# ============================================================================
# INTEGRATION WITH TRIAL.PY
# ============================================================================

def example_10_trial_integration():
    """Example 10: Integration with Trial.py"""
    print("\n" + "=" * 80)
    print("EXAMPLE 10: TRIAL.PY INTEGRATION")
    print("=" * 80)
    
    print("\nIn Trial.py, use:")
    print("""
    # Import guards
    from guards import create_guard_manager
    
    # Initialize
    guard_manager = create_guard_manager()
    
    # Validate data
    df = pd.read_csv('data.csv')
    valid_df, invalid_df, result = guard_manager.validate_dataframe(df)
    
    # Process only valid records
    print(f"Processing {len(valid_df)} valid records...")
    """)


# ============================================================================
# COMMON ERRORS AND FIXES
# ============================================================================

def example_11_error_handling():
    """Example 11: Common errors and fixes"""
    print("\n" + "=" * 80)
    print("EXAMPLE 11: ERROR HANDLING")
    print("=" * 80)
    
    guard = IdentityGuard()
    
    print("\nCommon Errors:")
    
    # Error 1: Bad RM format
    result = guard.check_identity_consistency('XX001', 'John', 25, 'L')
    print(f"1. Bad RM: {result.errors[0][:50]}...")
    
    # Error 2: Invalid age
    result = guard.check_identity_consistency('RM001', 'John', 150, 'L')
    print(f"2. Bad age: {result.errors[0][:50]}...")
    
    # Error 3: Invalid gender
    result = guard.check_identity_consistency('RM001', 'John', 25, 'X')
    print(f"3. Bad gender: {result.errors[0][:50]}...")
    
    print("\nFix: Ensure data matches expected format before validation")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def run_all_examples():
    """Run all examples"""
    print("\n")
    print("█" * 80)
    print("█ " + " " * 76 + " █")
    print("█ " + "GUARD SYSTEM QUICK REFERENCE - EXAMPLES".center(76) + " █")
    print("█ " + " " * 76 + " █")
    print("█" * 80)
    
    examples = [
        ("Single Record", example_1_single_record),
        ("Batch Records", example_2_batch_records),
        ("DataFrame", example_3_dataframe),
        ("Identity Guard", example_4_identity_guard),
        ("Duplicate Guard", example_5_duplicate_guard),
        ("Clinical Guard", example_6_clinical_guard),
        ("Temporal Guard", example_7_temporal_guard),
        ("Modes", example_8_modes),
        ("Reports", example_9_reports),
        ("Trial Integration", example_10_trial_integration),
        ("Error Handling", example_11_error_handling),
    ]
    
    print("\n\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i:2}. {name}")
    
    print("\n" + "=" * 80)
    print("NOTE: Some examples require the actual CSV file")
    print("      Comment them out if file not available")
    print("=" * 80)
    
    # Run examples that don't require CSV
    example_1_single_record()
    example_2_batch_records()
    example_4_identity_guard()
    example_5_duplicate_guard()
    example_6_clinical_guard()
    example_7_temporal_guard()
    example_8_modes()
    example_9_reports()
    example_10_trial_integration()
    example_11_error_handling()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    run_all_examples()
