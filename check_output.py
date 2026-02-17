#!/usr/bin/env python3
"""
Quick verification script to check if test.json has ID and type columns
"""

import json
import os

def check_output_file(file_path='data/test.json'):
    """Check if output file has ID and type columns with values"""
    
    print("="*70)
    print("VERIFYING OUTPUT FILE: data/test.json")
    print("="*70)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"\n✅ File loaded successfully")
    print(f"✅ Total records: {len(data)}")
    
    if len(data) == 0:
        print("❌ File is empty")
        return False
    
    # Check first record
    first = data[0]
    print(f"\n✅ First record keys: {list(first.keys())}")
    
    # Check for ID column
    has_id = 'ID' in first
    has_type = 'type' in first
    
    print(f"\n{'='*70}")
    print("COLUMN CHECK:")
    print(f"{'='*70}")
    print(f"✅ ID column present: {has_id}")
    print(f"✅ type column present: {has_type}")
    
    if not has_id:
        print("❌ MISSING: ID column")
        return False
    
    if not has_type:
        print("❌ MISSING: type column")
        return False
    
    # Check if type values are filled
    print(f"\n{'='*70}")
    print("PREDICTION CHECK:")
    print(f"{'='*70}")
    
    ids = [r.get('ID') for r in data]
    types = [r.get('type', '') for r in data]
    
    empty_types = sum(1 for t in types if not t or t == '')
    filled_types = len(types) - empty_types
    
    print(f"Records with ID: {len([i for i in ids if i is not None])}/{len(data)}")
    print(f"Records with type predictions: {filled_types}/{len(data)}")
    print(f"Empty type values: {empty_types}/{len(data)}")
    
    if empty_types == len(data):
        print("\n⚠️  WARNING: All type values are EMPTY!")
        print("   You need to run the prediction cells in the notebook first.")
        print("   Steps:")
        print("   1. Run Cell 42: Train final model")
        print("   2. Run Cell 43: Make predictions")
        print("   3. Run Cell 44: Save predictions")
        return False
    
    # Check type values
    unique_types = set([t for t in types if t and t != ''])
    valid_types = {'factual', 'contradiction', 'irrelevant'}
    
    print(f"\n{'='*70}")
    print("TYPE VALUES:")
    print(f"{'='*70}")
    print(f"Unique type values found: {unique_types}")
    print(f"Valid values expected: {valid_types}")
    
    invalid = unique_types - valid_types
    if invalid:
        print(f"⚠️  Invalid type values: {invalid}")
    else:
        print(f"✅ All type values are valid!")
    
    # Sample records
    print(f"\n{'='*70}")
    print("SAMPLE RECORDS:")
    print(f"{'='*70}")
    for i in range(min(5, len(data))):
        rec = data[i]
        print(f"Record {i+1}: ID={rec.get('ID')}, type='{rec.get('type')}'")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")
    
    if filled_types == len(data) and len(unique_types) > 0:
        print("✅ SUCCESS: File is ready!")
        print(f"   ✅ ID column: Present")
        print(f"   ✅ type column: Present with predictions")
        print(f"   ✅ All {len(data)} records have predictions")
        print(f"   ✅ Type values: {unique_types}")
        return True
    else:
        print("⚠️  File structure is correct but predictions are missing")
        print("   Run the prediction cells in the notebook to fill in type values")
        return False

if __name__ == "__main__":
    check_output_file()
