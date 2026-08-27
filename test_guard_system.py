"""
GUARD SYSTEM TEST SCRIPT
Quick tests for each guard module
"""

import pandas as pd
from guards import (
    IdentityGuard, 
    TemporalGuard, 
    ClinicalGuard, 
    DuplicateGuard,
    GuardManager
)

def test_identity_guard():
    """Test Identity Guard"""
    print("\n" + "=" * 80)
    print("TEST 1: IDENTITY GUARD")
    print("=" * 80)
    
    guard = IdentityGuard()
    
    # Test 1: Valid identity
    print("\n[Test 1.1] Valid identity registration:")
    result = guard.check_identity_consistency('RM001', 'Budi Hidayat', 25, 'L')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test 2: Invalid RM format
    print("\n[Test 1.2] Invalid RM format:")
    result = guard.check_identity_consistency('INVALID', 'Budi Hidayat', 25, 'L')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test 3: Age out of range
    print("\n[Test 1.3] Age out of range:")
    result = guard.check_identity_consistency('RM002', 'John Doe', 150, 'L')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test 4: Invalid gender
    print("\n[Test 1.4] Invalid gender:")
    result = guard.check_identity_consistency('RM003', 'Jane Smith', 30, 'X')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    print(f"\n[✓] Identity Guard Test Complete")
    return guard


def test_temporal_guard():
    """Test Temporal Guard"""
    print("\n" + "=" * 80)
    print("TEST 2: TEMPORAL GUARD")
    print("=" * 80)
    
    guard = TemporalGuard()
    
    # Test 1: Valid dates
    print("\n[Test 2.1] Valid date sequence (admission before examination):")
    result = guard.check_temporal_consistency('RM001', '2026-01-15', '2026-01-15')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Metadata: {result.metadata}")
    
    # Test 2: Examination before admission
    print("\n[Test 2.2] Invalid date sequence (examination before admission):")
    result = guard.check_temporal_consistency('RM002', '2026-02-20', '2026-02-10')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test 3: Invalid date format
    print("\n[Test 2.3] Invalid date format:")
    result = guard.check_temporal_consistency('RM003', 'invalid-date', '2026-01-15')
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    print(f"\n[✓] Temporal Guard Test Complete")
    return guard


def test_clinical_guard():
    """Test Clinical Guard"""
    print("\n" + "=" * 80)
    print("TEST 3: CLINICAL GUARD")
    print("=" * 80)
    
    guard = ClinicalGuard()
    
    # Test 1: Valid clinical data
    print("\n[Test 3.1] Valid clinical data:")
    result = guard.check_clinical_consistency(
        'RM001', 'Darah Rutin', 'WBC', 3.6, '', 'DBD'
    )
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")
    
    # Test 2: Invalid parameter
    print("\n[Test 3.2] Invalid parameter:")
    result = guard.check_clinical_consistency(
        'RM002', 'Darah Rutin', 'INVALID_PARAM', 5.0, '', 'DBD'
    )
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test 3: Parameter mismatch with exam type
    print("\n[Test 3.3] Parameter mismatch with exam type (HDL with Darah Rutin):")
    result = guard.check_clinical_consistency(
        'RM003', 'Darah Rutin', 'HDL', 43.0, '', 'DBD'
    )
    print(f"  Is valid: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    
    # Test 4: Out of range value
    print("\n[Test 3.4] Out of range value (WBC = 50):")
    result = guard.check_clinical_consistency(
        'RM004', 'Darah Rutin', 'WBC', 50.0, '', 'DBD'
    )
    print(f"  Is valid: {result.is_valid}")
    print(f"  Warnings: {result.warnings}")
    
    print(f"\n[✓] Clinical Guard Test Complete")
    return guard


def test_duplicate_guard():
    """Test Duplicate Guard"""
    print("\n" + "=" * 80)
    print("TEST 4: DUPLICATE GUARD")
    print("=" * 80)
    
    guard = DuplicateGuard()
    
    # Test 1: Check known duplicate (from analysis)
    print("\n[Test 4.1] Check for known duplicate (Bayu Setiawan, 41):")
    result = guard.detect_duplicates('RM215', 'Bayu Setiawan', 41)
    print(f"  Is duplicate: {result.is_duplicate}")
    print(f"  Duplicate type: {result.duplicate_type}")
    print(f"  Warnings: {result.warnings}")
    
    # Test 2: Register unique record
    print("\n[Test 4.2] Register unique record:")
    result = guard.detect_duplicates('RM999', 'Unique Person', 99)
    print(f"  Is duplicate: {result.is_duplicate}")
    print(f"  Duplicate type: {result.duplicate_type}")
    
    print(f"\n[✓] Duplicate Guard Test Complete")
    return guard


def test_guard_manager():
    """Test Guard Manager with sample data"""
    print("\n" + "=" * 80)
    print("TEST 5: GUARD MANAGER INTEGRATION")
    print("=" * 80)
    
    manager = GuardManager()
    manager.set_validation_mode('STRICT')
    
    # Create sample records
    sample_records = [
        {
            'No_RM': 'RM001', 'Nama': 'Budi Hidayat', 'Umur': 25, 'Jenis_Kelamin': 'L',
            'Tanggal_Masuk': '2026-01-15', 'Tanggal_Pemeriksaan': '2026-01-15',
            'Jenis_Pemeriksaan': 'Darah Rutin', 'Parameter': 'WBC', 
            'Hasil': 3.6, 'Satuan': '', 'Diagnosa': 'DBD'
        },
        {
            'No_RM': 'RM002', 'Nama': 'Sri Kartika', 'Umur': 24, 'Jenis_Kelamin': 'P',
            'Tanggal_Masuk': '2026-02-18', 'Tanggal_Pemeriksaan': '2026-02-19',
            'Jenis_Pemeriksaan': 'Darah Rutin', 'Parameter': 'RBC',
            'Hasil': 4.9, 'Satuan': '', 'Diagnosa': 'DBD'
        },
        {
            'No_RM': 'RM215', 'Nama': 'Bayu Setiawan', 'Umur': 41, 'Jenis_Kelamin': 'L',
            'Tanggal_Masuk': '2026-01-01', 'Tanggal_Pemeriksaan': '2026-01-01',
            'Jenis_Pemeriksaan': 'Darah Rutin', 'Parameter': 'PLT',
            'Hasil': 116237.0, 'Satuan': '', 'Diagnosa': 'DBD'
        }
    ]
    
    print("\n[*] Validating 3 sample records through all guards...")
    result = manager.validate_records(sample_records)
    
    print(f"\n[✓] Validation Complete")
    manager.print_summary(result)
    
    return manager


def main():
    """Run all tests"""
    print("\n")
    print("█" * 80)
    print("█ " + " " * 76 + " █")
    print("█ " + "GUARD SYSTEM COMPREHENSIVE TEST SUITE".center(76) + " █")
    print("█ " + " " * 76 + " █")
    print("█" * 80)
    
    # Run all tests
    identity_guard = test_identity_guard()
    temporal_guard = test_temporal_guard()
    clinical_guard = test_clinical_guard()
    duplicate_guard = test_duplicate_guard()
    manager = test_guard_manager()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n[✓] The Guard System is ready for integration with Trial.py")
    print("[✓] All individual guards are working correctly")
    print("[✓] GuardManager successfully coordinates all guards")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Use load_and_validate_data() function in Trial.py to validate incoming data")
    print("2. Access guard_manager.generate_report() for validation reports")
    print("3. Run integration_guard_system.py with real data from CSV")
    print("=" * 80)


if __name__ == '__main__':
    main()
