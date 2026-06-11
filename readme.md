# CoreCapture

## Project Structure

```text
CoreCapture
├─ Code
│  ├─ fasttext_model      # Directory for fastText pre-trained models
│  ├─ model_checkpoints   # Directory for storing model
│  ├─ preprocess.py   
│  ├─ get_vector.py       
│  ├─ test_anomaly_event.py  
│  ├─ test.py             
│  └─ train.py         
└─ Data
   ├─ dataset            # Directory for raw data files
   ├─ preprocessed       # Directory for intermediate preprocessed data
   │   ├─ test
   │   └─ train
   ├─ processed          # Directory for intermediate processed data ready for train and test
   └─ template           # Directory for Drain template
```

## Data Preparation

To run CoreCapture, using systrace data as an example, the following data preparation steps are required:

- **1: Raw systrace files**: The original systrace logs to be parsed, stored in `Data/dataset/`.
- **2: Preprocessed JSON data**: Run `preprocess.py` to parse raw systrace files. The results are saved in `Data/preprocessed/`, where each JSON file represents one event and contains several parsed traces.
- **3: Train/test split**: Manually split the preprocessed JSON files into training and test sets, and place them in `Data/preprocessed/train` or `Data/preprocessed/test` for subsequent training and testing.

## Train and Test

### For Train, run:

```shell
python get_vector.py
python train.py
```

### For Test, run:

```shell
python get_vector.py
python test.py
```

### For Event-Level AD, run:

```shell
python test_anomaly_event.py
```
