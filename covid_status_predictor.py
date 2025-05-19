import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Load models
models = {
    "Logistic Regression": "logistic_model.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

# Load label encoders
input_label_encoders = joblib.load("input_label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Create main window
root = tk.Tk()
root.title("COVID Status Predictor")
root.geometry("350x350")
root.resizable(False, False)
root.configure(bg="lightblue")

# Create input labels and entries
entries = {}
fields = ["age", "sex", "date_announced", "home_quarantined", "date_of_onset_of_symptoms", "pregnant"]

for i, field in enumerate(fields):
    tk.Label(root, text=field.replace("_", " ").capitalize(), anchor="w", width=20, bg="lightblue").grid(row=i, column=0, padx=10, pady=5, sticky='e')

    entry_frame = tk.Frame(root, bg="lightblue")
    entry_frame.grid(row=i, column=1, padx=10, pady=5, sticky='w')

    if field == "sex":
        entry = ttk.Combobox(entry_frame, values=["Male", "Female"], width=22)
    elif field == "home_quarantined":
        entry = ttk.Combobox(entry_frame, values=["Yes", "No"], width=22)
    elif field == "pregnant":
        entry = ttk.Combobox(entry_frame, values=["Yes", "No"], width=22)
    elif field in ["date_announced", "date_of_onset_of_symptoms"]:
        entry = DateEntry(entry_frame, width=20, background='darkblue', foreground='white', borderwidth=2)
    else:
        entry = tk.Entry(entry_frame, width=24)

    entry.pack()
    entries[field] = entry

# Model selection
tk.Label(root, text="Choose Model", anchor="w", width=20, bg="lightblue").grid(row=len(fields), column=0, padx=10, pady=10, sticky='e')
model_var = tk.StringVar()
model_selector = ttk.Combobox(root, textvariable=model_var, values=list(models.keys()), width=24)
model_selector.grid(row=len(fields), column=1, padx=10, pady=10, sticky='w')
model_selector.current(0)

# Prediction result label
result_label = tk.Label(root, text="Status: ", font=("Arial", 12, "bold"), bg="lightblue")
result_label.grid(row=len(fields)+2, column=0, columnspan=2, pady=10)

# Input validation function
def validate_inputs():
    try:
        for field in fields:
            if not entries[field].get():
                raise ValueError(f"Please fill in the {field.replace('_', ' ').capitalize()} field.")

        age = float(entries["age"].get())
        if age <= 0:
            raise ValueError("Age must be a positive number.")

        datetime.strptime(entries["date_announced"].get(), "%m/%d/%y")
        datetime.strptime(entries["date_of_onset_of_symptoms"].get(), "%m/%d/%y")

        return True

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
        return False
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))
        return False

# Prediction function
def predict_status():
    try:
        if not validate_inputs():
            return

        input_data = [entries[field].get() for field in fields]
        input_data[fields.index("age")] = float(input_data[fields.index("age")])

        input_data[fields.index("sex")] = input_label_encoders["sex"].transform([input_data[fields.index("sex")]])[0]
        input_data[fields.index("home_quarantined")] = input_label_encoders["home_quarantined"].transform([input_data[fields.index("home_quarantined")]])[0]
        input_data[fields.index("pregnant")] = input_label_encoders["pregnant"].transform([input_data[fields.index("pregnant")]])[0]

        reference_date = datetime(2020, 1, 1)
        date_announced = datetime.strptime(entries["date_announced"].get(), "%m/%d/%y")
        date_of_onset = datetime.strptime(entries["date_of_onset_of_symptoms"].get(), "%m/%d/%y")

        input_data[fields.index("date_announced")] = (date_announced - reference_date).days
        input_data[fields.index("date_of_onset_of_symptoms")] = (date_of_onset - reference_date).days

        df_input = pd.DataFrame([input_data], columns=fields)

        selected_model_file = models[model_var.get()]
        model = joblib.load(selected_model_file)

        prediction = model.predict(df_input)[0]
        decoded_prediction = target_encoder.inverse_transform([prediction])[0]

        result_label.config(text=f"Status: {decoded_prediction}")

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# Predict button
tk.Button(root, text="Predict", command=predict_status, width=20).grid(row=len(fields)+1, column=0, columnspan=2, pady=10)

root.mainloop()
