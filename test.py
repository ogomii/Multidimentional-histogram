import my_module
import numpy as np

# Données d'exemple
data = np.array([[1.0, 2.0, 2.5], [3.0, 3.5, 4.0], [4.5, 5.0, 5.5]])  # Exemple de tableau 2D
bins = [np.linspace(0, 6, 4), np.linspace(0, 6, 4), np.linspace(0, 6, 4)]  # Exemple de bords de bin pour chaque dimension

# Appeler la fonction C++ depuis Python
result = my_module.calculate_histogram(data, bins)

# Afficher le résultat
print("Histogram:", result[0])
print("Bin Edges:", result[1])
