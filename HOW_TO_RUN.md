# how to run - complete guide

**5*a by ahmed abobakr**

## quick start

### 1. train models
```bash
cd /home/Ahmed-abobakr/Downloads/ML_Ahmed
source .venv/bin/activate
python3 train_model_fast.py
```
time: 2-3 minutes

### 2. run web application
```bash
.venv/bin/streamlit run app.py --server.port 8501
```
access: http://localhost:8501

### 3. run jupyter notebook
```bash
.venv/bin/jupyter notebook 02_Clustering_House_Prices.ipynb
```
access: http://localhost:8888

## important notes

- always use `.venv/bin/` prefix or activate venv first
- system python does not have required packages
- models must be trained before running app
- port 8501 for streamlit, 8888 for jupyter

## troubleshooting

problem: module not found
solution: use .venv/bin/python or activate venv

problem: model artifacts not found  
solution: run train_model_fast.py first

problem: port in use
solution: change port number or kill process

**project ready for github**
**5*a by ahmed abobakr**
