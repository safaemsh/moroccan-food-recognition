"""
Tests unitaires pour le modèle et ses configurations

"""

import unittest
import json
import os
import sys

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestModel(unittest.TestCase):
    """Tests pour vérifier la configuration du modèle"""
    
    @classmethod
    def setUpClass(cls):
        """Définir les chemins une seule fois"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.class_indices_path = os.path.join(base_dir, 'models', 'training', 'models', 'saved_models', 'class_indices.json')
        cls.history_path = os.path.join(base_dir, 'models', 'training', 'models', 'saved_models', 'training_history.json')
    
    def test_class_indices_file_exists(self):
        """Vérifie que le fichier class_indices.json existe"""
        self.assertTrue(os.path.exists(self.class_indices_path), "Le fichier class_indices.json n'existe pas")
        print(" PASSED : Test existence class_indices.json : OK")
    
    def test_correct_number_of_classes(self):
        """Vérifie qu'il y a exactement 25 classes"""
        with open(self.class_indices_path, 'r') as f:
            classes = json.load(f)
        
        self.assertEqual(len(classes), 25, f"Attendu 25 classes, trouvé {len(classes)}")
        print(f"PASSED : Test nombre de classes : {len(classes)} classes OK")
    
    def test_class_names_valid(self):
        """Vérifie que les noms de classes sont des chaînes non vides"""
        with open(self.class_indices_path, 'r') as f:
            classes = json.load(f)
        
        # Le format peut être {"0": "Baghrir"} ou {"Baghrir": 0}
        for key, value in classes.items():
            # Vérifier que soit la clé soit la valeur est une string non vide
            if isinstance(value, str):
                self.assertTrue(len(value) > 0, f"Classe {key} a un nom vide")
            elif isinstance(key, str):
                self.assertTrue(len(key) > 0, "Nom de classe vide")
        
        print(f"PASSED : Test validité noms de classes : OK")
    
    def test_training_history_exists(self):
        """Vérifie que le fichier training_history.json existe"""
        self.assertTrue(os.path.exists(self.history_path), "Le fichier training_history.json n'existe pas")
        print("PASSED : Test existence training_history.json : OK")
    
    def test_training_history_structure(self):
        """Vérifie que l'historique d'entraînement contient les bonnes clés"""
        with open(self.history_path, 'r') as f:
            history = json.load(f)
        
        required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        for key in required_keys:
            self.assertIn(key, history, f"Clé '{key}' manquante dans l'historique")
        
        print("PASSED : Test structure historique : OK")
    
    def test_training_history_consistency(self):
        """Vérifie que toutes les listes de l'historique ont la même longueur"""
        with open(self.history_path, 'r') as f:
            history = json.load(f)
        
        lengths = [len(history[key]) for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']]
        self.assertTrue(all(l == lengths[0] for l in lengths))
        
        print(f"PASSED : Test cohérence historique : OK - {lengths[0]} epochs")
    
    def test_final_accuracy_reasonable(self):
        """Vérifie que l'accuracy finale est raisonnable (> 70%)"""
        with open(self.history_path, 'r') as f:
            history = json.load(f)
        
        final_val_accuracy = history['val_accuracy'][-1]
        self.assertGreater(final_val_accuracy, 0.70, 
                          f"Val accuracy ({final_val_accuracy:.4f}) < 70%")
        
        print(f"PASSED : Test accuracy : OK - {final_val_accuracy:.4f} (> 70%)")
    
    def test_no_severe_overfitting(self):
        """Vérifie qu'il n'y a pas d'overfitting sévère (gap < 15%)"""
        with open(self.history_path, 'r') as f:
            history = json.load(f)
        
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        gap = abs(final_train_acc - final_val_acc)
        
        self.assertLess(gap, 0.15, f"Gap ({gap:.4f}) > 15%")
        
        print(f"PASSED : Test overfitting : OK - Gap = {gap:.4f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" TESTS DU MODÈLE - MOROCCAN FOOD RECOGNITION")
    print("="*60 + "\n")
    
    unittest.main(verbosity=2)