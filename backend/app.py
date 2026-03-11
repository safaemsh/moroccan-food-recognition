"""
Backend API pour la reconnaissance de plats marocains
FastAPI + TensorFlow
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

# Initialiser FastAPI
app = FastAPI(title="Moroccan Food Recognition API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemins
MODEL_PATH = "../models/training/models/saved_models/moroccan_food_model.h5"
CLASS_INDICES_PATH = "../models/training/models/saved_models/class_indices.json"

CALORIES_DATA_PATH = "models/calories_data.json"

# Charger le modèle au démarrage
model = None
class_names = {}
calories_data = {}

# Base de données des calories (calories pour 100g et portion moyenne)
CALORIES_DATABASE = {
    "couscous": {"calories_per_100g": 112, "portion_moyenne_g": 300, "description": "Plat traditionnel berbère"},
    "tajine": {"calories_per_100g": 150, "portion_moyenne_g": 350, "description": "Ragoût cuit lentement"},
    "harira": {"calories_per_100g": 85, "portion_moyenne_g": 250, "description": "Soupe traditionnelle"},
    "pastilla": {"calories_per_100g": 280, "portion_moyenne_g": 200, "description": "Pâté feuilleté sucré-salé"},
    "briouates": {"calories_per_100g": 320, "portion_moyenne_g": 100, "description": "Petits triangles frits"},
    "rfissa": {"calories_per_100g": 180, "portion_moyenne_g": 300, "description": "Galettes déchiquetées"},
    "zaalouk": {"calories_per_100g": 65, "portion_moyenne_g": 150, "description": "Salade d'aubergines"},
    "mechoui": {"calories_per_100g": 250, "portion_moyenne_g": 200, "description": "Agneau rôti"},
    "kefta": {"calories_per_100g": 220, "portion_moyenne_g": 150, "description": "Boulettes de viande"},
    "msemen": {"calories_per_100g": 290, "portion_moyenne_g": 80, "description": "Crêpe feuilletée"}
}

@app.on_event("startup")
async def load_model_and_data():
    """
    Charger le modèle et les données au démarrage
    """
    global model, class_names, calories_data
    
    try:
        # Charger le modèle
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Modèle chargé avec succès")
        else:
            print("⚠️ Modèle non trouvé, utilisez d'abord train_model.py")
        
        # Charger les indices de classes
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
                class_indices = json.load(f)
                # Inverser le dictionnaire pour obtenir index -> nom
                class_names = {v: k for k, v in class_indices.items()}
                print(f"✅ {len(class_names)} classes chargées")
        
        # Charger ou créer les données de calories
        if os.path.exists(CALORIES_DATA_PATH):
            with open(CALORIES_DATA_PATH, 'r', encoding='utf-8') as f:
                calories_data = json.load(f)
        else:
            # Utiliser la base de données par défaut
            calories_data = CALORIES_DATABASE
            os.makedirs(os.path.dirname(CALORIES_DATA_PATH), exist_ok=True)
            with open(CALORIES_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(calories_data, f, ensure_ascii=False, indent=2)
            print("✅ Base de données des calories créée")
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prétraiter l'image pour la prédiction
    """
    # Redimensionner
    image = image.resize((224, 224))
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertir en array et normaliser
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def estimate_calories(dish_name: str):
    """
    Estimer les calories pour un plat
    """
    dish_name_lower = dish_name.lower()
    
    # Chercher dans la base de données
    for key in calories_data:
        if key.lower() in dish_name_lower or dish_name_lower in key.lower():
            data = calories_data[key]
            cal_per_100g = data.get('calories_per_100g', 150)
            portion = data.get('portion_moyenne_g', 200)
            total_calories = (cal_per_100g * portion) / 100
            
            return {
                "calories_per_100g": cal_per_100g,
                "portion_moyenne_g": portion,
                "calories_totales": round(total_calories, 2),
                "description": data.get('description', '')
            }
    
    # Valeur par défaut si non trouvé
    return {
        "calories_per_100g": 150,
        "portion_moyenne_g": 200,
        "calories_totales": 300,
        "description": "Estimation basée sur une moyenne"
    }

@app.get("/")
async def root():
    """
    Page d'accueil de l'API
    """
    return {
        "message": "API de Reconnaissance de Plats Marocains",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Prédire un plat à partir d'une image",
            "/classes": "GET - Liste des classes disponibles",
            "/health": "GET - Vérifier l'état de l'API"
        }
    }

@app.get("/health")
async def health_check():
    """
    Vérifier l'état de l'API
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": len(class_names) > 0
    }

@app.get("/classes")
async def get_classes():
    """
    Obtenir la liste des classes
    """
    return {
        "classes": list(class_names.values()),
        "total": len(class_names)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédire le plat à partir d'une image
    """
    # Vérifier que le modèle est chargé
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        # Faire la prédiction
        predictions = model.predict(processed_image, verbose=0)
        
        # Obtenir les top 3 prédictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_3_indices:
            dish_name = class_names.get(idx, f"Classe_{idx}")
            confidence = float(predictions[0][idx])
            
            # Estimer les calories
            calorie_info = estimate_calories(dish_name)
            
            results.append({
                "plat": dish_name,
                "confiance": round(confidence * 100, 2),
                "calories": calorie_info
            })
        
        # Résultat principal (meilleure prédiction)
        best_prediction = results[0]
        
        return {
            "success": True,
            "prediction": best_prediction["plat"],
            "confiance": best_prediction["confiance"],
            "calories": best_prediction["calories"],
            "top_3_predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.post("/update-calories")
async def update_calories(dish: str, calories_per_100g: int, portion_moyenne_g: int, description: str = ""):
    """
    Mettre à jour les informations caloriques d'un plat
    """
    global calories_data
    
    calories_data[dish] = {
        "calories_per_100g": calories_per_100g,
        "portion_moyenne_g": portion_moyenne_g,
        "description": description
    }
    
    # Sauvegarder
    with open(CALORIES_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(calories_data, f, ensure_ascii=False, indent=2)
    
    return {"success": True, "message": f"Calories mises à jour pour {dish}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)