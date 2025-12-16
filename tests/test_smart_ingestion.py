import os
import pandas as pd
from stats_compass_core.state import DataFrameState
from stats_compass_core.data.smart_load import smart_load, SmartLoadInput

def test_smart_load():
    # Setup
    state = DataFrameState()
    
    # Create dummy CSV
    csv_path = os.path.abspath("test_smart.csv")
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
    
    # Create dummy JSON
    json_path = os.path.abspath("test_smart.json")
    pd.DataFrame({"x": ["foo", "bar"], "y": [10, 20]}).to_json(json_path)
    
    try:
        # Test CSV
        print(f"Testing CSV load from {csv_path}...")
        result_csv = smart_load(state, SmartLoadInput(path=csv_path))
        print(f"CSV Load Result: {result_csv.dataframe_name}, shape={result_csv.shape}")
        assert result_csv.dataframe_name == "test_smart"
        assert result_csv.shape[0] == 2
        
        # Test JSON
        print(f"Testing JSON load from {json_path}...")
        result_json = smart_load(state, SmartLoadInput(path=json_path))
        print(f"JSON Load Result: {result_json.dataframe_name}, shape={result_json.shape}")
        assert result_json.dataframe_name == "test_smart" # Name collision handling? 
        # Wait, smart_load doesn't handle name collision, it just overwrites in state.
        # But the name derived from filename is same.
        
        # Verify state
        print("State keys:", state.list_dataframes())
        
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(json_path):
            os.remove(json_path)

if __name__ == "__main__":
    test_smart_load()
