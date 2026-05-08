import pandas as pd
import numpy as np

# ── Chargez le fichier principal ──────────────────────────────
CSV_PATH = 'dataset/Battery Data/train_dataset_4.csv'
df = pd.read_csv(CSV_PATH)

# 1. Colonnes disponibles
print("=" * 60)
print("COLONNES :")
print(df.columns.tolist())

# 2. Batteries disponibles
print("\n" + "=" * 60)
print("BATTERIES DISPONIBLES :")
print(df['cell'].unique())

# 3. Cycles par batterie
print("\n" + "=" * 60)
print("NOMBRE DE CYCLES PAR BATTERIE :")
print(df.groupby('cell')['cycle'].max().sort_values())

# 4. Valeurs manquantes
print("\n" + "=" * 60)
print("VALEURS MANQUANTES PAR COLONNE :")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 5. SOH_C range
print("\n" + "=" * 60)
print("RANGE DE SOH_C :")
print(f"  Min : {df['SOH_C'].min():.2f}%")
print(f"  Max : {df['SOH_C'].max():.2f}%")
print(f"  Moyenne : {df['SOH_C'].mean():.2f}%")

# 6. Aperçu
print("\n" + "=" * 60)
print("APERÇU (5 premières lignes) :")
print(df.head())