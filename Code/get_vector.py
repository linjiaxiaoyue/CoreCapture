import os
import json
import pandas as pd
import numpy as np
from gensim.models import FastText


SOURCE_ROOT_DIR = "../Data/preprocessed/test"  # ../Data/preprocessed/test
OUTPUT_ROOT_DIR = "../Data/processed"
FASTTEXT_MODEL_PATH = "../Code/fasttext_model/fasttext.model"
VECTOR_DIMENSION = 50
MODE = 2  # 1--trian，2--test

def get_sentence_vector_gensim(sentence, model, vector_size):
    if pd.isna(sentence) or sentence.strip() == "":
        return np.zeros(vector_size)
    
    slice_names = [s.strip() for s in sentence.split('||') if s.strip()]
    all_word_vectors = []
    
    for slice_name in slice_names:
        words = slice_name.split()
        for word in words:
            if word in model.wv:
                all_word_vectors.append(model.wv[word])
    
    if not all_word_vectors:
        return np.zeros(vector_size)
    
    return np.mean(all_word_vectors, axis=0)

def load_single_json_data(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)

        required_columns = ['trace_sequence', 'duration', 'pre_interval', 'label', 'window_name']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns in {json_file_path}: {missing_cols}, skipping this file")
            return None
        
        if len(df) == 0:
            print(f"{json_file_path} is empty, skipping this file")
            return None
            
        return df
        
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {json_file_path}, skipping this file")
        return None
    except Exception as e:
        print(f"Failed to load {json_file_path}: {e}, skipping this file")
        return None

def extract_features_and_labels(df, model):
    call_chains = df['trace_sequence'].fillna("").astype(str).tolist()
    text_vectors = np.array([
        get_sentence_vector_gensim(chain, model, VECTOR_DIMENSION) 
        for chain in call_chains
    ])

    duration = pd.to_numeric(df['duration'], errors='coerce').fillna(0).values.reshape(-1, 1)
    pre_interval = pd.to_numeric(df['pre_interval'], errors='coerce').fillna(0).values.reshape(-1, 1)
    
    normalized_duration = np.log1p(duration)
    normalized_interval = np.log1p(pre_interval)

    X = np.hstack((text_vectors, normalized_duration, normalized_interval))

    y = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int).values

    window_names = df['window_name'].fillna("unknown").astype(str).values

    return X, y, window_names

def process_all_json_files(model):
    all_X = []
    all_y = []
    all_window_names = []
    
    for root, dirs, files in os.walk(SOURCE_ROOT_DIR):
        for file_name in files:
            if file_name.lower().endswith(".json"):
                source_json_path = os.path.join(root, file_name)
                print(f"\n=====================================")
                print(f"Processing file: {source_json_path}")
                
                df = load_single_json_data(source_json_path)
                if df is None:
                    continue
                
                print(f"Successfully loaded {len(df)} call chain records")
                
                X, y, window_names = extract_features_and_labels(df, model)
                if X is None or y is None or window_names is None:
                    print(f"Feature extraction failed for {source_json_path}, skipping")
                    continue
                
                all_X.append(X)
                all_y.append(y)
                all_window_names.append(window_names)
                
                curr_total = sum(arr.shape[0] for arr in all_X)
                print(f"File processed | Current total samples: {curr_total}")

    # Merge all data
    print("\n=====================================")
    print("Merging all data...")
    total_X = np.vstack(all_X)
    total_y = np.hstack(all_y)
    total_window_names = np.hstack(all_window_names)

    # Strict validation: sample counts must match
    if not (total_X.shape[0] == total_y.shape[0] == total_window_names.shape[0]):
        raise ValueError(
            f"Sample count mismatch!\n"
            f"X samples: {total_X.shape[0]}\n"
            f"y samples: {total_y.shape[0]}\n"
            f"window_names samples: {total_window_names.shape[0]}"
        )

    # Create output directory
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    
    # Save files
    if MODE == 1:
        np.save(os.path.join(OUTPUT_ROOT_DIR, "X_train.npy"), total_X)
        np.save(os.path.join(OUTPUT_ROOT_DIR, "y_train.npy"), total_y)
        print(f"\n🎉 All data processing completed!")
        print(f"X_train.npy shape: {total_X.shape}")
        print(f"y_train.npy shape: {total_y.shape}")
        print(f"All files saved to: {OUTPUT_ROOT_DIR}")
    if MODE == 2:
        np.save(os.path.join(OUTPUT_ROOT_DIR, "X_test.npy"), total_X)
        np.save(os.path.join(OUTPUT_ROOT_DIR, "y_test.npy"), total_y)
        np.save(os.path.join(OUTPUT_ROOT_DIR, "window_names.npy"), total_window_names)
        print(f"\n All data processing completed!")
        print(f"X_test.npy shape: {total_X.shape}")
        print(f"y_test.npy shape: {total_y.shape}")
        print(f"window_names.npy shape: {total_window_names.shape}")
        print(f"All files saved to: {OUTPUT_ROOT_DIR}")
    

def main():
    """
    Main function
    """
    if not os.path.exists(SOURCE_ROOT_DIR):
        print(f"[ERROR] Source root directory does not exist!")
        return

    print(f"\nLoading Gensim FastText model...")
    model = FastText.load(FASTTEXT_MODEL_PATH)
    print("Model loaded successfully (full model)")
    
    process_all_json_files(model)

if __name__ == "__main__":
    main()