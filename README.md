### *Project Title:* `Thyroid Mlflow App`

### *Technologies:* `Machine Learning Technology`

### *Domain:* `Healthcare`

### *Deploy:* `link`

## Thyroid Diagnosis Challenges
Thyroid disorders affect many people worldwide, the challenge for healthcare professionals in accurately diagnosing. The Thyroid App using Streamlit, to streamline the diagnostic process and help healthcare professionals to make informed decisions.

# Run Locally

Clone the project

```bash
  git clone
```

Go to the project directory

```bash
  cd Thyroid-Mlflow-app
```

Install dependencies

```bash
  pip install -r requirements.txt
```

On local

```bash
  python src/pipeline.py
  ```

Start the server `streamlit`

```bash
  streamlit run app.py
```

## Attributes Information
* Age: Age of the patient (numeric)
* Sex: Sex of the patient (categorical: 'M' for male, 'F' for female)
* On Thyroxine: Whether the patient is on thyroxine medication (binary: 'Yes' or 'No')
* Query on Thyroxine: Whether the patient is querying about thyroxine medication (binary: 'Yes' or 'No')
* On Antithyroid Meds: Whether the patient is on antithyroid medication (binary: 'Yes' or 'No')
Sick: Whether the patient is sick (binary: 'Yes' or 'No')
* Pregnant: Whether the patient is pregnant (binary: 'Yes' or 'No')
* Thyroid Surgery: Whether the patient has undergone thyroid surgery (binary: 'Yes' or 'No')
* I131 Treatment: Whether the patient is undergoing I131 treatment (binary: 'Yes' or 'No')
* Query Hypothyroid: Whether the patient believes they have hypothyroidism (binary: 'Yes' or 'No')
* Query Hyperthyroid: Whether the patient believes they have hyperthyroidism (binary: 'Yes' or 'No')
* Lithium: Whether the patient is taking lithium medication (binary: 'Yes' or 'No')
* Goitre: Whether the patient has goitre (binary: 'Yes' or 'No')
* Tumor: Whether the patient has a tumor (binary: 'Yes' or 'No')
* Hypopituitary: Whether the patient has hypopituitarism (binary: 'Yes' or 'No')
* Psych: Whether the patient has psychiatric issues (binary: 'Yes' or 'No')
* TSH: Thyroid-stimulating hormone level in blood (numeric)
* T3: Triiodothyronine level in blood (numeric)
* TT4: Total thyroxine level in blood (numeric)
* T4U: Thyroxine utilization rate in blood (numeric)
* FTI: Free thyroxine index in blood (numeric)
* Referral_Source - (str)
* classes - hyperthyroidism diagnosis (str) 

*Approach:* The classical machine learning tasks:
- EDA, 
- Data Preprocessing,
- Feature Engineering, 
- Model Training and Testing.

inputs: Age, Sex, On_thyroxine, Query_on_thyroxine, On_antithyroid_medication, Sick, Pregnant, Thyroid_surgery, I131_treatment, Query_hypothyroid, Query_hyperthyroid, Lithium, Goitre, Tumor, Hypopituitary, Psych, TSH, T3, TT4, T4U, FTI, Referral_Source.

outputs: classes

