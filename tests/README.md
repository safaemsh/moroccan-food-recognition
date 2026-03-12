\#  Tests Unitaires



Ce dossier contient les tests automatisés pour vérifier le bon fonctionnement du projet Morocco Food Recognition.



\##  Fichiers



\- \*\*test\_preprocessing.py\*\* : Tests du preprocessing des images

\- \*\*test\_model.py\*\* : Tests de la configuration et performance du modèle



\##  Lancer les tests



\### Tous les tests

```bash

python -m unittest discover tests/

```



\### Un fichier spécifique

```bash

cd tests

python test\_preprocessing.py

python test\_model.py

```



\### Avec plus de détails

```bash

python test\_preprocessing.py -v

python test\_model.py -v

```



\##  Ce qui est testé



\### Preprocessing (test\_preprocessing.py)

\-  Images redimensionnées à 224×224

\-  Pixels normalisés entre \[0, 1]

\-  Format RGB (3 canaux)

\-  Dimensions du batch correctes

\-  Conversion grayscale → RGB



\### Modèle (test\_model.py)

\-  Fichier class\_indices.json existe

\-  25 classes présentes

\-  Noms de classes valides

\-  Fichier training\_history.json existe

\-  Structure de l'historique correcte

\-  Cohérence des données (même nombre d'epochs)

\-  Accuracy finale > 95%

\-  Pas d'overfitting sévère (gap < 5%)



\##  Résultats attendus



Tous les tests doivent passer (OK) :

```

test\_batch\_dimensions (\_\_main\_\_.TestPreprocessing) ...  OK

test\_image\_format\_conversion (\_\_main\_\_.TestPreprocessing) ...  OK

test\_image\_resize\_dimensions (\_\_main\_\_.TestPreprocessing) ...  OK

test\_pixel\_normalization (\_\_main\_\_.TestPreprocessing) ...  OK

test\_rgb\_channels (\_\_main\_\_.TestPreprocessing) ...  OK

\----------------------------------------------------------------------

Ran 5 tests in 0.012s



OK

```



\##  Ajouter un nouveau test



1\. Créer une classe héritant de `unittest.TestCase`

2\. Nommer les méthodes `test\_xxx`

3\. Utiliser les assertions :

&#x20;  - `self.assertEqual(a, b)` : vérifie que a == b

&#x20;  - `self.assertTrue(condition)` : vérifie que condition est True

&#x20;  - `self.assertGreater(a, b)` : vérifie que a > b

&#x20;  - `self.assertIn(item, liste)` : vérifie que item est dans liste



4\. Lancer avec `python -m unittest`



\##  Exemple de nouveau test

```python

def test\_nouvelle\_fonctionnalite(self):

&#x20;   """Description du test"""

&#x20;   # Arrange (préparer)

&#x20;   donnees = \[1, 2, 3]

&#x20;   

&#x20;   # Act (exécuter)

&#x20;   resultat = sum(donnees)

&#x20;   

&#x20;   # Assert (vérifier)

&#x20;   self.assertEqual(resultat, 6)

&#x20;   print(" Test nouvelle fonctionnalité : OK")

```



\---



\*\*Auteur :\*\* Safa Mashate  

\*\*Projet :\*\* Morocco Food Recognition - ENSA Agadir 2026

