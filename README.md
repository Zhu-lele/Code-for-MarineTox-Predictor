**MarineTox-Predictor**


**Project Structure**

config.py - Model configuration and hyperparameters

dataset_utils.py - Dataset processing and graph construction

graph_utils.py - Molecular graph building from SMILES

load_dataset.py - Data loading and preprocessing

metrics_calculator.py - Evaluation metrics and loss functions

train_eval_manager.py - Training loops and model evaluation

vis_utils.py - Visualization of molecular attention weights


**Quick Start**

1. Prepare Data
Format your CSV file with columns: smiles, group (train/val/test), and toxicity labels.

2. Build Graph Dataset
python
python load_dataset.py
This converts SMILES to graph format and saves as .bin file.

3. Configure Model
Edit config.py to set your tasks and parameters:

python
args['select_task_list'] = ['task1', 'task2', ...]

args['num_epochs'] = 500

args['batch_size'] = 128

4. Train Model
Run the training script (main training file not included in provided code).


**Supported Tasks**

38 aquatic toxicity endpoints including acute/chronic tests for fish, crustaceans, and algae.
