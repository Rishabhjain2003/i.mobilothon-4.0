# README for Predictive Maintenance System Repository.

## Repository Overview  
This repository contains the workflow and code for building a machine learning application to predict engine health and assess machine confidence levels. It includes data preprocessing, model training, and a user-friendly application powered by Streamlit.  

---

## Repository Structure  

### 1. `engine_data.csv`  
- **Description**: Contains the raw engine data, including features and target variables (if applicable).  
- **Usage**: Used as input for data preprocessing in `Data_Preprocessing.ipynb`.  

### 2. `Data_Preprocessing.ipynb`  
- **Description**:  
  This Jupyter Notebook contains the preprocessing pipeline:  
  - **Feature Engineering**: Transformation and creation of new features to improve model performance.  
  - **Outlier Detection and Removal**: Techniques to identify and eliminate outliers from the dataset to ensure data integrity.  
- **Output**: Produces `cleaned_engine_data.csv`.  

### 3. `cleaned_engine_data.csv`  
- **Description**: Cleaned dataset generated after preprocessing. This dataset is used for model training.  

### 4. `Model_Training.ipynb`  
- **Description**:  
  - Defines the architecture of the neural network model.  
  - Includes model evaluation and performance reports.  
  - Provides functionality to predict engine health and assess machine confidence levels.  

### 5. `Application.py`  
- **Description**:  
  - Implements a graphical user interface (GUI) using Streamlit.  
  - Allows users to input data and get predictions on engine health along with confidence levels.  

---

## Setup Instructions  

### Prerequisites  
- Python 3.8 or later  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `streamlit`, and others as required in the notebooks and scripts.  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/repository-name.git  
   cd repository-name  
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  

### Running the Application  
1. Preprocess the raw data:  
   Open and execute `Data_Preprocessing.ipynb` to generate the cleaned dataset (`cleaned_engine_data.csv`).  

2. Train the model:  
   Run `Model_Training.ipynb` to train and evaluate the neural network model.  

3. Launch the application:  
   ```bash
   streamlit run Application.py  
   ```  

---

## Usage  

1. Input engine data into the Streamlit application.  
2. Receive predictions on engine health.  
3. View confidence levels for the predictions.  

---
