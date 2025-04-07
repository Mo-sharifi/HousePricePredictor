# 🏠 House Price Predictor

This project is a machine learning-based system designed to predict house prices based on various features such as area, number of rooms, amenities, and location.

---

## 📦 Requirements

- Python 3.x  
- pip  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- streamlit  
- requests  

To install all dependencies, use the `requirements.txt` file.

---

## ⚙️ Installation

1. **Create a virtual environment**:

```bash
python3 -m venv venv
```

2. **Activate the virtual environment**:

- On Linux/macOS:
```bash
source venv/bin/activate
```

3. **Install the dependencies**:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Load the dataset  
Import your dataset in CSV format.

### 2. Data Preprocessing  
The data is automatically cleaned, including:
- Removing missing values  
- Removing outliers  
- Converting data into proper formats

### 3. Modeling  
This project uses the following models:
- **Decision Tree**
- **Random Forest**

### 4. Visualizing the Results  
The predictions are visualized using **Streamlit** for an interactive UI.

---

## ▶️ Running the Project

To run the project in the terminal:

```bash
python main.py
```

To launch the Streamlit app:

```bash
streamlit run app.py
```

---

## 🔧 Core Functions

- `read_dataset(file_path)`  
  Loads a CSV file and returns a pandas DataFrame.

- `preprocess_data(data)`  
  Cleans and transforms the data by removing irrelevant columns and handling missing values.

- `remove_outlier(data)`  
  Detects and removes outliers from the dataset.

- `show_dataframe(df)`  
  Displays the DataFrame in a formatted table using `tabulate`.

---

## 🤖 Machine Learning Models

- **Decision Tree**  
  Predicts house prices using input features like area, number of rooms, and location.

- **Random Forest**  
  An ensemble model built from multiple decision trees to improve accuracy and robustness.

---

## 📊 Output

Model predictions are visualized using charts and tables, allowing users to explore how different features influence the predicted house prices.

---

## 👨‍💻 Developed by

- **Mohammadreza Sharifi**  
  [GitHub Profile](https://github.com/Mo-sharifi)

---
