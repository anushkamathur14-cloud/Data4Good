#!/usr/bin/env python3
"""
Verification script to check that test.json has correct format:
- ID column present
- type column present with valid predictions
- All 2000 rows
"""

import json
import os

def verify_output_file(file_path='data/test.json'):
    """Verify the output JSON file has correct structure"""
    
    print("="*70)
    print("VERIFYING OUTPUT FILE: data/test.json")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found at {file_path}")
        return False
    
    # Load the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON file - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Could not read file - {e}")
        return False
    
    print(f"\n✅ File loaded successfully")
    print(f"✅ Total records: {len(data)}")
    
    # Check record count
    if len(data) != 2000:
        print(f"⚠️  WARNING: Expected 2000 records, found {len(data)}")
    else:
        print(f"✅ Record count correct: 2000")
    
    # Check first record structure
    if len(data) == 0:
        print("❌ ERROR: File is empty")
        return False
    
    first_record = data[0]
    required_keys = ['ID', 'type']
    
    print(f"\n{'='*70}")
    print("CHECKING REQUIRED COLUMNS:")
    print(f"{'='*70}")
    
    all_present = True
    for key in required_keys:
        if key in first_record:
            print(f"✅ '{key}' column: PRESENT")
        else:
            print(f"❌ '{key}' column: MISSING")
            all_present = False
    
    if not all_present:
        print("\n❌ ERROR: Missing required columns!")
        return False
    
    # Check data quality
    print(f"\n{'='*70}")
    print("CHECKING DATA QUALITY:")
    print(f"{'='*70}")
    
    ids = []
    types = []
    missing_ids = 0
    missing_types = 0
    empty_types = 0
    
    for i, record in enumerate(data):
        # Check ID
        if 'ID' not in record or record['ID'] is None:
            missing_ids += 1
        else:
            ids.append(record['ID'])
        
        # Check type
        if 'type' not in record or record['type'] is None:
            missing_types += 1
        elif record['type'] == '':
            empty_types += 1
        else:
            types.append(record['type'])
    
    print(f"✅ Records with ID: {len(ids)}/{len(data)}")
    print(f"✅ Records with type: {len(types)}/{len(data)}")
    
    if missing_ids > 0:
        print(f"❌ Missing IDs: {missing_ids}")
    if missing_types > 0:
        print(f"❌ Missing types: {missing_types}")
    if empty_types > 0:
        print(f"❌ Empty type strings: {empty_types}")
    
    # Check type values
    print(f"\n{'='*70}")
    print("CHECKING TYPE VALUES:")
    print(f"{'='*70}")
    
    from collections import Counter
    type_counts = Counter(types)
    
    print(f"Type distribution:")
    for type_val, count in type_counts.most_common():
        print(f"  {type_val}: {count}")
    
    # Check for valid type values
    valid_types = {'factual', 'contradiction', 'irrelevant'}
    actual_types = set(types)
    
    invalid_types = actual_types - valid_types
    if invalid_types:
        print(f"\n⚠️  WARNING: Invalid type values found: {invalid_types}")
    else:
        print(f"\n✅ All type values are valid!")
    
    missing_valid_types = valid_types - actual_types
    if missing_valid_types:
        print(f"⚠️  WARNING: Expected type values not found: {missing_valid_types}")
    else:
        print(f"✅ All expected type values present!")
    
    # Check ID range and uniqueness
    print(f"\n{'='*70}")
    print("CHECKING ID COLUMN:")
    print(f"{'='*70}")
    
    if ids:
        print(f"✅ ID range: {min(ids)} to {max(ids)}")
        unique_ids = len(set(ids))
        print(f"✅ Unique IDs: {unique_ids}/{len(ids)}")
        if unique_ids != len(ids):
            print(f"⚠️  WARNING: Duplicate IDs found!")
    
    # Sample records
    print(f"\n{'='*70}")
    print("SAMPLE RECORDS:")
    print(f"{'='*70}")
    
    sample_size = min(5, len(data))
    for i in range(sample_size):
        record = data[i]
        print(f"\nRecord {i+1}:")
        print(f"  ID: {record.get('ID', 'MISSING')}")
        print(f"  type: {record.get('type', 'MISSING')}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY:")
    print(f"{'='*70}")
    
    all_good = (
        len(data) == 2000 and
        all_present and
        missing_ids == 0 and
        missing_types == 0 and
        empty_types == 0 and
        len(invalid_types) == 0
    )
    
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print("\nThe output file is correctly formatted with:")
        print("  ✅ ID column present for all records")
        print("  ✅ type column present with valid predictions")
        print("  ✅ All 2000 records included")
        print("  ✅ Valid type values (factual, contradiction, irrelevant)")
        return True
    else:
        print("⚠️  SOME ISSUES FOUND - Please review above")
        return False

if __name__ == "__main__":
    verify_output_file()

