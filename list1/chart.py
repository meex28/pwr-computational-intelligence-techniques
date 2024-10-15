import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj dane z pliku CSV
df = pd.read_csv('results.csv')

# Posortuj dane według prawdopodobieństwa krzyżowania i mutacji
df = df.sort_values(['cross_prob', 'mut_prob'])

# Zdefiniuj własną paletę kolorów
custom_palette = sns.color_palette("husl", n_colors=len(df['mut_prob'].unique()))

# Stwórz wykres
plt.figure(figsize=(15, 10))
sns.barplot(x='cross_prob', y='colors', hue='mut_prob', data=df, palette=custom_palette)

# Dostosuj wykres
plt.title('Liczba kolorów w zależności od prawdopodobieństwa krzyżowania i mutacji', fontsize=16)
plt.xlabel('Prawdopodobieństwo krzyżowania', fontsize=12)
plt.ylabel('Liczba kolorów', fontsize=12)
plt.legend(title='Prawd. mutacji', title_fontsize='12', fontsize='10')

# Dodaj etykiety wartości na słupkach
for i in plt.gca().containers:
    plt.gca().bar_label(i, fontsize=8, padding=2)

# Obróć etykiety na osi x dla lepszej czytelności
plt.xticks(rotation=45)

# Dostosuj układ
plt.tight_layout()

# Zapisz wykres
plt.savefig('genetic_algorithm_results_chart_reversed.png', dpi=300)

# Wyświetl wykres
plt.show()