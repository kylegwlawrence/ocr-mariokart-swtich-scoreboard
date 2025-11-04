import json
import pandas as pd

def flatten_runtime_config(df):
    """
    Flatten runtime_config column in dataframe to multiple columns.
    
    Args:
        df: pandas DataFrame with 'runtime_config' column containing JSON strings
        
    Returns:
        pandas DataFrame with flattened columns and runtime_config removed
    """
    
    def flatten_dict(d, parent_key='', sep='_'):
        """Recursively flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and k == 'steps':
                continue
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def process_row(runtime_config_str):
        """Process single runtime_config string into flat dictionary"""
        row_data = {}
        
        # Parse JSON string
        data = json.loads(runtime_config_str)
        
        # Flatten preprocessing config (excluding steps)
        preprocessing_config = data.get('preprocessing_config', {})
        for key, value in preprocessing_config.items():
            if key != 'steps':
                row_data[key] = value
        
        # Flatten OCR config with parent prefixes
        ocr_engine_config = data.get('ocr_engine_config', {})
        flattened_ocr = flatten_dict(ocr_engine_config)
        row_data.update(flattened_ocr)
        
        # Process steps
        steps = preprocessing_config.get('steps', [])
        for item in steps:
            method = item["method"]
            params = item["params"]
            
            # Add boolean column for method
            row_data[f"has_{method}"] = True
            
            # Add parameter columns with method prefix
            for param_name, param_value in params.items():
                column_name = f"{method}_{param_name}"
                if isinstance(param_value, list):
                    row_data[column_name] = f"({','.join(map(str, param_value))})"
                else:
                    row_data[column_name] = param_value
        
        return row_data
    
    # Process each row
    flattened_rows = df['runtime_config'].apply(process_row)
    flattened_df = pd.DataFrame(flattened_rows.tolist())
    
    # Combine with original dataframe (excluding runtime_config)
    df_without_config = df.drop('runtime_config', axis=1)
    result_df = pd.concat([df_without_config.reset_index(drop=True), 
                           flattened_df.reset_index(drop=True)], axis=1)
    
    return result_df


def process_csv(input_csv_path, output_csv_path):
    """
    Read CSV, flatten runtime_config, and save result.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file
    """
    df = pd.read_csv(input_csv_path)
    result_df = flatten_runtime_config(df)
    result_df.to_csv(output_csv_path, index=False)
    print(f"CSV created successfully at: {output_csv_path}")
    print(f"Shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")
    return result_df