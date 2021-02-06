# Subtask 2

#### Objective:

Detection of antecedent and consequence – extract boundaries of antecedent and consequent from the input text.

Correspondence to: ifajcik@fit.vutbr.cz Full list of results reported in the paper is
available [here](https://docs.google.com/spreadsheets/d/1msV2PqCM4OgiYQHvayuEcby5IpSGDM9OH2nHg8_QwlU/edit?usp=sharing).

### Data

Data are downloaded automatically when running training or validation. See method `check_for_download`
in `subtask_2/datasets/dataset_task2.py` for details.

### Replication of results

0. __REQUIREMENTS__  
   Install requirements into python3.6
   ```
   cd subtask_2
   python -m pip install -r requirements.txt
   ```
1. __IMPORT PATHS__  
   Set your `PYTHONPATH` to `subtask_2` folder. Run every script from this folder.
   ```
   export PYTHONPATH=$(pwd)
   ```
2. __TRAINING__  
   You can train `bert-base`, `bert-large`, `ALBERT-xxlarge-v2` and `RoBERTa` by running their respective run_scripts:
   ```
   python  subtask_2/src_scripts/transformers_task2/run_{model_name}_task2.py
   ```

   You can adjust all the parameters inside run_script by changing the `config` dictionary.

3. __INFERENCE__    
     If you haven't trained your model, you can use ours `RoBERTa-large` 
     checkpoint available [here](http://www.stud.fit.vutbr.cz/~ifajcik/semeval2020/task5/semeval2020task5b_roberta_large_EM_74.65_F1_88.63_L_0.60_statedict.pt) (if link won't work try copy it into new window/plug it into `wget`).
   
   __validation__  
   Simply set `eval_only` key in runfile configuration to `True` and set `model_path` key  to path to your checkpoint (relative from your PYTHONPATH). For example, assuming the provided checkpoint is in directory `subtask2`, and you modify your run_file as following (e.g. run_roberta_large_task2.py):
   ```
   config={
   ...
   "model_path": "semeval2020task5b_roberta_large_EM_74.65_F1_88.63_L_0.60_statedict.pt",
   "eval_only": True
   ...
   }
   ```
   You should obtain following outputs, based on 10% of data split from original training data.
   ```
   total EM: 74.64788732394366
   total F1: 88.62808947513123
   antecedent EM: 79.43661971830986
   antecedent F1: 92.22556136965888
   consequent EM: 69.85915492957747
   consequent F1: 85.03061758060359
   no-consequent ACCURACY: 91.54929577464789
   ```
   
   __predicting the custom_inputs__
   You wil need to create inputs in test data format - two-column csv with header - you can create 2 example input file called `input_data.csv` with following contents:
   ```
   sentenceID,sentence
   203551,Sandwich in Kent; until 2011 the flagship of Pfizer's European research should have been closed down years before.
   203552,"If the deals were properly accounted for, Bank of America's Tier 1 capital ratio -- a key metric monitored by bank regulators -- would have declined 0.01 percent on Sept 30, 2008, when the largest such error existed."
   ```
   Next you will need to use `src_scripts/transformers_task2/run_prediction.py` script with correct configuration setting. Note the script is supplied with directory path, which can contain multiple models, therefore you can also use it for ensemble prediction. For our example file `input_data.csv`, the configuration dictionary (located at the start of this file) can be set as following.
   ```python
   config = {
    "objective_variant": "independent", 
    "decoding_variant": "independent",
    "tokenizer_type": "roberta-large", 
    "model_type": "roberta-large",
    "max_antecedent_length": 116,  # optimized via 1000 hyperopt trials
    "max_consequent_length": 56,  # optimized via 1000 hyperopt trials
    "tensorboard_logging": False,
    "model_dir": "saved",  # directory containing all checkpoints
    "test_file": "input_data.csv", # path to input data
    "dropout_rate": 0.,
    "cache_dir": "./.BERTcache",
    "working_directory": ".predictions" # directory where the immediate predictions and final prediction will be saved
   }
   ```
Now, all you need to do is run `src_scripts/transformers_task2/run_prediction.py` and you will obtain two prediction files in the prediction directory (`.predictions` in our example). 
* `result_<timestamp>.csv`  
  Contains predictions in competition format, e.g.:
  ```
  sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid
   203551,18,112,-1,-1
   203552,0,39,42,214
  ```
* `result_<timestamp>.csv_readable.csv`  
  Contains predictions in human readable format, ordered as `sentenceID,sentence, antecedent, consequent, antecedent_startid,antecedent_endid,consequent_startid,consequent_endid` e.g.:
  ```
  203551,Sandwich in Kent; until 2011 the flagship of Pfizer's European research should have been closed down years before.,until 2011 the flagship of Pfizer's European research should have been closed down years before,,18,112,-1,-1
   203552,"If the deals were properly accounted for, Bank of America's Tier 1 capital ratio -- a key metric monitored by bank regulators -- would have declined 0.01 percent on Sept 30, 2008, when the largest such error existed.",If the deals were properly accounted for,"Bank of America's Tier 1 capital ratio -- a key metric monitored by bank regulators -- would have declined 0.01 percent on Sept 30, 2008, when the largest such error existed",0,39,42,214
   ```

4. __HELP I AM STUCK/LOST__  
If anything won't work well, trying reaching out for help via email (`ifajciK@fit.vutbr.cz`) or create an issue :wink: .
