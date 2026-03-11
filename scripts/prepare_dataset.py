"""
Script pour préparer les données du dataset Kaggle
Crée les splits train/validation/test
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Configuration
DATA_DIR =  "../dataset"  # Dossier contenant le dataset Kaggle
OUTPUT_DIR = "dataset/splits"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_directory_structure():
    """
    Créer la structure des dossiers
    """
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
    print("✅ Structure des dossiers créée")

def get_images_by_class(data_dir):
    """
    Organiser les images par classe
    """
    images_by_class = defaultdict(list)
    
    # Parcourir le dossier de données
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        
        if os.path.isdir(class_path):
            # Lister toutes les images de cette classe
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    images_by_class[class_name].append(img_path)
    
    return images_by_class

def split_and_copy_images(images_by_class):
    """
    Diviser les images en train/val/test et les copier
    """
    stats = {
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    for class_name, images in images_by_class.items():
        # Mélanger les images
        random.shuffle(images)
        
        # Calculer les tailles des splits
        total = len(images)
        train_size = int(total * TRAIN_RATIO)
        val_size = int(total * VAL_RATIO)
        
        # Diviser
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Créer les dossiers de classe
        for split in ['train', 'val', 'test']:
            class_dir = os.path.join(OUTPUT_DIR, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Copier les images
        for img_path in train_images:
            dest = os.path.join(OUTPUT_DIR, 'train', class_name, os.path.basename(img_path))
            shutil.copy2(img_path, dest)
            stats['train'] += 1
        
        for img_path in val_images:
            dest = os.path.join(OUTPUT_DIR, 'val', class_name, os.path.basename(img_path))
            shutil.copy2(img_path, dest)
            stats['val'] += 1
        
        for img_path in test_images:
            dest = os.path.join(OUTPUT_DIR, 'test', class_name, os.path.basename(img_path))
            shutil.copy2(img_path, dest)
            stats['test'] += 1
        
        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    return stats

def main():
    """
    Fonction principale
    """
    print("🚀 Début de la préparation des données")
    print(f"Dataset source: {DATA_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Ratios - Train: {TRAIN_RATIO}, Val: {VAL_RATIO}, Test: {TEST_RATIO}")
    print("-" * 60)
    
    # Vérifier que le dossier source existe
    if not os.path.exists(DATA_DIR):
        print(f"❌ Erreur: Le dossier {DATA_DIR} n'existe pas")
        print("   Téléchargez d'abord le dataset depuis Kaggle:")
        print("   https://www.kaggle.com/datasets/othmanehilal/morocco-food-dataset")
        return
    
    # Créer la structure
    create_directory_structure()
    
    # Obtenir les images par classe
    print("\n📂 Analyse du dataset...")
    images_by_class = get_images_by_class(DATA_DIR)
    
    if not images_by_class:
        print("❌ Aucune image trouvée dans le dataset")
        return
    
    print(f"✅ {len(images_by_class)} classes trouvées")
    print(f"   Total images: {sum(len(imgs) for imgs in images_by_class.values())}")
    
    # Diviser et copier
    print("\n📋 Division des données:")
    random.seed(42)  # Pour la reproductibilité
    stats = split_and_copy_images(images_by_class)
    
    # Afficher les statistiques finales
    print("\n" + "=" * 60)
    print("✅ Préparation terminée!")
    print(f"   Train: {stats['train']} images")
    print(f"   Validation: {stats['val']} images")
    print(f"   Test: {stats['test']} images")
    print(f"   Total: {sum(stats.values())} images")
    print("=" * 60)
    print("\nVous pouvez maintenant lancer l'entraînement avec:")
    print("  python models/training/train_model.py")

if __name__ == "__main__":
    main()