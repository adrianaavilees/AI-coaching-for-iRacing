import pandas as pd
import numpy as np

# Check train metadata for n_samples stats
train = pd.read_csv('data/processed/train_metadata.csv')
test = pd.read_csv('data/processed/test_metadata.csv')
all_meta = pd.concat([train, test])

print('=== Puntos originales por vuelta (n_samples) ===')
print(all_meta['n_samples'].describe())
print()
print(f'Min: {all_meta["n_samples"].min()}')
print(f'Max: {all_meta["n_samples"].max()}')
print(f'Median: {all_meta["n_samples"].median()}')
print(f'Mean: {all_meta["n_samples"].mean():.0f}')
print()

# Distribution buckets
bins = [0, 2000, 3000, 4000, 5000, 6000, 7000, 10000]
labels = ['<2k', '2-3k', '3-4k', '4-5k', '5-6k', '6-7k', '7k+']
all_meta['bucket'] = pd.cut(all_meta['n_samples'], bins=bins, labels=labels)
print('Distribución:')
print(all_meta['bucket'].value_counts().sort_index())
print()
print(f'Ratio compresión mínimo: {all_meta["n_samples"].min()} -> 1000 = {all_meta["n_samples"].min()/1000:.1f}x')
print(f'Ratio compresión máximo: {all_meta["n_samples"].max()} -> 1000 = {all_meta["n_samples"].max()/1000:.1f}x')
print(f'Ratio compresión mediana: {all_meta["n_samples"].median()} -> 1000 = {all_meta["n_samples"].median()/1000:.1f}x')