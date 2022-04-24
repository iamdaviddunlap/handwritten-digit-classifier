# Handwritten Digit Classifier

# Data

All data can be found in the `data` directory. There is one folder for each language.
Each folder has the train/test split ready. All unprocessed data for Odia is in `data/odia/scan`.

The custom dataset was processed using `python parse_scan.py`.

# Requirements

Run `pip install -r requirements.txt`.

# Run

Run `python sklearn_ml.py` or `python cnn.py`. 

In each of these files, there is a variable for TRAIN that can be set to true or false. The model can also be changed.

Evaluation prints out confusion matrices.