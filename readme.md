# MNIST Classification avec MLflow

Ce projet implémente un réseau de neurones dense pour la classification de chiffres manuscrits du dataset MNIST, avec suivi des expérimentations via MLflow.

## 📋 Description

Le modèle utilise un réseau de neurones entièrement connecté (Dense Neural Network) pour classifier des images de chiffres de 0 à 9. MLflow est intégré pour tracker les hyperparamètres, métriques et modèles.

## 🏗️ Architecture du Modèle

- **Couche d'entrée** : 784 neurones (images 28×28 pixels aplaties)
- **Couche cachée** : 512 neurones avec activation ReLU
- **Dropout** : 20% pour la régularisation
- **Couche de sortie** : 10 neurones avec activation Softmax (classification multi-classe)

## 📦 Prérequis

```bash
pip install tensorflow numpy mlflow
```

## 🚀 Utilisation

### Entraînement du modèle

```bash
python mnist_mlflow.py
```

### Visualisation des résultats avec MLflow

```bash
mlflow ui
```

Puis ouvrez votre navigateur à l'adresse : `http://localhost:5000`

## ⚙️ Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Époques | 5 |
| Batch size | 128 |
| Validation split | 0.1 (10%) |
| Neurones cachés | 512 |
| Dropout rate | 0.2 (20%) |
| Optimiseur | Adam |
| Loss | Sparse Categorical Crossentropy |

## 📊 Métriques Suivies

Le script enregistre automatiquement dans MLflow :

- **Hyperparamètres** : nombre d'époques, batch size, architecture, etc.
- **Métriques d'entraînement** : loss et accuracy par époque
- **Métriques de validation** : val_loss et val_accuracy par époque
- **Métriques de test** : test_loss et test_accuracy finales
- **Modèle** : sauvegarde au format MLflow et H5

## 📈 Résultats Attendus

Le modèle devrait atteindre une précision d'environ **97-98%** sur le jeu de test après 5 époques.

## 📁 Structure du Projet

```
.
├── mnist_mlflow.py          # Script principal d'entraînement
├── mnist_model.h5           # Modèle sauvegardé (généré)
├── mlruns/                  # Dossier MLflow (généré)
└── README.md                # Ce fichier
```

## 🔍 Fonctionnalités MLflow

### Expérimentation
- Nom de l'expérience : `MNIST_Classification`
- Enregistrement automatique de tous les hyperparamètres
- Tracking des métriques à chaque époque

### Modèle Registry
- Le modèle est enregistré sous le nom : `MNIST_DNN_Model`
- Signature du modèle inférée automatiquement
- Format compatible pour le déploiement

### Tags
- `model_type`: Dense Neural Network
- `dataset`: MNIST
- `framework`: TensorFlow/Keras

## 🎯 Utilisation du Modèle Sauvegardé

### Chargement avec MLflow

```python
import mlflow.keras

# Charger le modèle
model = mlflow.keras.load_model("runs:/<RUN_ID>/model")

# Faire des prédictions
predictions = model.predict(x_test)
```

### Chargement avec Keras

```python
from tensorflow import keras

model = keras.models.load_model('mnist_model.h5')
predictions = model.predict(x_test)
```

## 🛠️ Personnalisation

Vous pouvez modifier les hyperparamètres au début du script :

```python
epochs = 5
batch_size = 128
hidden_units = 512
dropout_rate = 0.2
```

## 📝 Notes

- Les données MNIST sont automatiquement téléchargées par Keras
- Le modèle est normalisé (pixels entre 0 et 1)
- La validation split de 10% est appliquée automatiquement
- Tous les résultats sont trackés dans le dossier `mlruns/`

## 🤝 Contribution

Pour améliorer le modèle, vous pouvez :
- Ajuster les hyperparamètres
- Ajouter des couches cachées supplémentaires
- Tester différents optimiseurs
- Implémenter l'early stopping

## 📜 Licence

Ce projet est à des fins éducatives dans le cadre d'un TP de Deep Learning.