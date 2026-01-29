import pandas as pd
import numpy as np
import joblib
import os

# Scikit-Learn Ecosystem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline as SkPipeline

# Advanced Libraries
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class MaintenanceModelTrainer:
    def __init__(self, data_path, model_dir):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None

        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        """Loads data and performs initial cleanup."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # FIX: The dataset has 14 columns. We map all of them.
        df.columns = [
            "UDI", "Product_ID", "Type", "Air_Temp", "Process_Temp",
            "Rot_Speed", "Torque", "Tool_Wear", "Target",
            "TWF", "HDF", "PWF", "OSF", "RNF"
        ]
        return df

    def feature_engineering(self, df):
        """
        PhD Level: Physics-Informed Feature Engineering.
        """
        print("Engineering physics-based features...")

        # 1. Temperature Difference: (Process - Air)
        df['Temp_Diff'] = df['Process_Temp'] - df['Air_Temp']

        # 2. Power Factor: Power is proportional to Torque * Speed
        df['Power_Factor'] = df['Rot_Speed'] * df['Torque']

        # 3. Strain 1: Tool Wear * Torque
        df['Tool_Strain'] = df['Tool_Wear'] * df['Torque']

        # DROP IRRELEVANT & LEAKAGE COLUMNS
        # We drop 'TWF', 'HDF' etc because they are the answers (Target Leakage).
        # We only want to predict based on sensor readings (Temp, Speed, Torque).
        drop_cols = ['UDI', 'Product_ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        df = df.drop(columns=drop_cols)

        return df

    def build_pipeline(self):
        """
        Constructs a Professional Stacking Ensemble Pipeline.
        """
        # Define Features (Updated list after engineering)
        numeric_features = ['Air_Temp', 'Process_Temp', 'Rot_Speed', 'Torque',
                            'Tool_Wear', 'Temp_Diff', 'Power_Factor', 'Tool_Strain']
        categorical_features = ['Type']

        # Preprocessing Step
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Base Learners (Level 0)
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ]

        # Stacking Classifier (Level 1)
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=StratifiedKFold(n_splits=5)
        )

        # Full Pipeline: Preprocessor -> SMOTE -> Stacking
        model_pipeline = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', stacking_clf)
        ])

        self.model = model_pipeline
        return self.model

    def run(self):
        # 1. Load & Engineer
        df = self.load_data()
        df = self.feature_engineering(df)

        # 2. Split X and y
        X = df.drop('Target', axis=1)
        y = df['Target']

        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 4. Build & Train
        print("Training Stacking Ensemble with SMOTE...")
        self.build_pipeline()
        self.model.fit(X_train, y_train)

        # 5. Evaluation
        print("\n--- Model Evaluation ---")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        # 6. Save Artifacts
        print(f"\nSaving artifacts to {self.model_dir}...")
        joblib.dump(self.model, os.path.join(self.model_dir, 'best_model.pkl'))
        # Save feature names for the UI to know input order
        joblib.dump(list(X.columns), os.path.join(self.model_dir, 'feature_names.pkl'))

        print("Training Complete. Ready for Deployment.")


if __name__ == "__main__":
    # Adjust path relative to src/
    DATA_PATH = os.path.join('..', 'data', 'ai4i2020.csv')
    MODEL_DIR = os.path.join('..', 'models')

    trainer = MaintenanceModelTrainer(DATA_PATH, MODEL_DIR)
    trainer.run()