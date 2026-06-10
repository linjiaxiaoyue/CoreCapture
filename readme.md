# CoreCapture

## Project Structure
```
CoreCapture
├─ Code
│  ├─ fasttext_model      # Directory for fastText pre-trained models
│  ├─ model_checkpoints   # Directory for storing model
│  ├─ data_processed.py   
│  ├─ get_vector.py       
│  ├─ test_anomaly_window.py  
│  ├─ test.py             
│  └─ train.py         
└─ Data
   ├─ dataset            # Directory for raw data files
   ├─ preprocessed       # Directory for intermediate preprocessed data
   ├─ processed          # Directory for final processed data ready for training
   └─ template           # Directory for Drain template
```

## How To Use
### For Raw Systrace Data Preprocess, run:
```shell
python data_processed.py
```

### For Model Training, run:
```shell
python get_vector.py
python train.py
```

### For Model Test, run:
```shell
python test.py
```

### For Systraces Event-Level AD, run:
```shell
python test_anomaly_window.py
```