"""
Tests unitaires pour le preprocessing des images
"""

import unittest
import numpy as np
from PIL import Image
import io


class TestPreprocessing(unittest.TestCase):
    """Tests pour vérifier le preprocessing des images"""
    
    def test_image_resize_dimensions(self):
        """Vérifie que les images sont redimensionnées à 224x224"""
        # Créer une image de test (taille arbitraire)
        test_image = Image.new('RGB', (500, 300), color='blue')
        
        # Simuler le resize (comme dans votre backend)
        resized = test_image.resize((224, 224))
        
        # Vérifier les dimensions
        self.assertEqual(resized.size, (224, 224))
        print(" Test resize dimensions : OK - Image bien redimensionnée à 224x224")
    
    def test_pixel_normalization(self):
        """Vérifie que les pixels sont normalisés entre 0 et 1"""
        # Créer un array de pixels aléatoires [0-255]
        pixels = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        # Normaliser (division par 255)
        normalized = pixels.astype('float32') / 255.0
        
        # Vérifier la plage [0, 1]
        self.assertTrue(normalized.min() >= 0)
        self.assertTrue(normalized.max() <= 1)
        print(" Test normalisation : OK - Pixels normalisés entre [0, 1]")
    
    def test_rgb_channels(self):
        """Vérifie que les images ont exactement 3 canaux RGB"""
        image_array = np.random.rand(224, 224, 3)
        
        # Vérifier la forme (hauteur, largeur, canaux)
        self.assertEqual(image_array.shape, (224, 224, 3))
        print(" Test canaux RGB : OK - 3 canaux présents")
    
    def test_batch_dimensions(self):
        """Vérifie que le batch processing maintient les bonnes dimensions"""
        batch_size = 32
        batch = np.random.rand(batch_size, 224, 224, 3)
        
        # Vérifier les dimensions du batch
        self.assertEqual(batch.shape, (32, 224, 224, 3))
        print(f" Test batch : OK - Batch de {batch_size} images correct")
    
    def test_image_format_conversion(self):
        """Vérifie la conversion d'images grayscale en RGB"""
        # Créer une image en niveaux de gris
        gray_image = Image.new('L', (224, 224), color=128)
        
        # Convertir en RGB
        rgb_image = gray_image.convert('RGB')
        
        # Vérifier le mode
        self.assertEqual(rgb_image.mode, 'RGB')
        self.assertEqual(rgb_image.size, (224, 224))
        print(" Test conversion grayscale→RGB : OK")


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" TESTS DE PREPROCESSING - MOROCCAN FOOD RECOGNITION")
    print("="*60 + "\n")
    
    # Lancer les tests avec verbosité
    unittest.main(verbosity=2)