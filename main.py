from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import mahotas
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, File, UploadFile

# Crear la aplicación FastAPI
app = FastAPI()

# Configuraciones globales
width, height = 350, 450
fixed_size = (width, height)
bins = 8

# Aquí colocamos todas las funciones anteriores (omitiendo el bloque `if __name__ == "__main__"`)

def check_file_permissions(file_path):
    """Check if file has read permissions."""
    try:
        return os.access(file_path, os.R_OK)
    except Exception as e:
        print(f"[ERROR] Error checking file permissions: {str(e)}")
        return False

def load_image(file_path):
    """Load image using OpenCV or PIL as fallback."""
    if not check_file_permissions(file_path):
        print(f"[ERROR] No read permissions for: {file_path}")
        return None, None
    
    # Try OpenCV first
    image = cv2.imread(file_path)
    if image is not None:
        return image, "OpenCV"

    # Try PIL as fallback
    try:
        pil_image = Image.open(file_path)
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, "PIL"
    except Exception as e:
        print(f"[ERROR] Failed to load image with PIL: {str(e)}")
        return None, None

def fd_hu_moments(image):
    """Extract Hu Moments features."""
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature
    except Exception as e:
        print(f"[ERROR] Error extracting Hu Moments: {str(e)}")
        return None

def fd_haralick(image):
    """Extract Haralick features."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick
    except Exception as e:
        print(f"[ERROR] Error extracting Haralick features: {str(e)}")
        return None

def fd_histogram(image, mask=None):
    """Extract color histogram features."""
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], 
                           [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        print(f"[ERROR] Error extracting histogram features: {str(e)}")
        return None

def estimate_hole_depth(image, contours):
    """Estimate depth of holes in potato."""
    try:
        depth_estimates = []
        for cnt in contours:
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
            hole_pixels = cv2.bitwise_and(image, image, mask=mask)
            mask_pixels = mask == 255
            if np.any(mask_pixels):
                mean_intensity = np.mean(hole_pixels[mask_pixels])
                if mean_intensity > 0:
                    estimated_depth = 255 / mean_intensity
                    depth_estimates.append(estimated_depth)
        return depth_estimates
    except Exception as e:
        print(f"[ERROR] Error estimating hole depth: {str(e)}")
        return []

def analyze_potato_holes_with_depth(image):
    """Analyze holes in potato and their depth."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        depth_estimates = estimate_hole_depth(gray, contours)
        return np.mean(depth_estimates) if depth_estimates else 0
    except Exception as e:
        print(f"[ERROR] Error analyzing potato holes: {str(e)}")
        return 0

def measure_potato_size(image):
    """Measure potato dimensions and shape characteristics."""
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("[WARNING] No contours found.")
            return None
        
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        
        if M['m00'] == 0:
            print("[WARNING] Contour area is 0.")
            return None
        
        area = M['m00']
        
        mu20 = M['mu20'] / M['m00']
        mu02 = M['mu02'] / M['m00']
        mu11 = M['mu11'] / M['m00']
        
        temp = np.sqrt(4 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02))
        
        major_axis = 2 * np.sqrt(2) * np.sqrt(mu20 + mu02 + temp)
        minor_axis = 2 * np.sqrt(2) * np.sqrt(mu20 + mu02 - temp)
        aspect_ratio = round(major_axis / minor_axis if minor_axis > 0 else 0, 2)
        
        return {
            'area': int(area),
            'major_axis': int(major_axis),
            'minor_axis': int(minor_axis),
            'aspect_ratio': aspect_ratio
        }
    except Exception as e:
        print(f"[ERROR] Error measuring potato size: {str(e)}")
        return None

def get_measurements(image):
    """Get basic measurements from potato image."""
    if image is None:
        print("[ERROR] Image is empty or failed to load.")
        return None
        
    measurements = measure_potato_size(image)
    if measurements is None:
        print("[ERROR] Failed to get potato measurements.")
        return None
    
    return [
        measurements['area'],
        measurements['major_axis'],
        measurements['minor_axis'],
        measurements['aspect_ratio']
    ]

def extract_features(image_path):
    try:
        # Cargar imagen
        image, method = load_image(image_path)
        if image is None:
            print(f"[ERROR] No se pudo cargar la imagen: {image_path}")
            return None

        print(f"[INFO] Imagen cargada exitosamente con {method}. Forma: {image.shape}")
        image = cv2.resize(image, fixed_size)

        # Extraer características en el mismo orden que el código original
        fv_histogram = fd_histogram(image)
        fv_haralick = fd_haralick(image)
        fv_hu_moments = fd_hu_moments(image)
        fv_eje = get_measurements(image)  # Cambiado de obtener_solo_medidas a get_measurements
        avg_depth = analyze_potato_holes_with_depth(image)

        # Concatenar todas las características
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments, fv_eje, [avg_depth]])

        # Normalizar características
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = scaler.fit_transform(global_feature.reshape(1, -1))

        return rescaled_features.flatten()

    except Exception as e:
        print(f"[ERROR] Error al procesar la imagen: {str(e)}")
        return None

def predict_with_onnx(features, model_path):
    """Predict using ONNX model."""
    try:
        # Cargar el modelo
        print("[INFO] Cargando modelo ONNX...")
        sess = ort.InferenceSession(model_path)

        # Obtener información del modelo
        print("[INFO] Información del modelo:")
        print("Inputs:", sess.get_inputs())
        print("Outputs:", sess.get_outputs())

        # Preparar entrada
        input_name = sess.get_inputs()[0].name
        print(f"[INFO] Nombre de entrada: {input_name}")
        print(f"[INFO] Features shape antes: {features.shape}")
        
        # Convertir y reshape datos
        input_data = features.astype(np.float32)
        if len(input_data.shape) == 1:
            input_data = np.expand_dims(input_data, axis=0)
        print(f"[INFO] Input shape después: {input_data.shape}")

        # Realizar predicción
        print("[INFO] Ejecutando predicción...")
        try:
            output = sess.run(None, {input_name: input_data})
            print(f"[INFO] Salida del modelo: {output}")
            
            predicted_index = int(np.argmax(output[0]))
            confidence = float(output[0].flatten()[predicted_index])
            
            print(f"[INFO] Clase predicha: {predicted_index}")
            print(f"[INFO] Confianza: {confidence}")
            
            return predicted_index, confidence
        except Exception as inner_e:
            print(f"[ERROR] Error en la predicción: {str(inner_e)}")
            return None, 0.0

    except Exception as e:
        print(f"[ERROR] Error general en predict_with_onnx: {str(e)}")
        return None, 0.0

def classify_potato_image(image_path, model_path):
    """Main function to classify potato image."""
    try:
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return None
            
        features = extract_features(image_path)
        if features is None:
            return None
        
        return predict_with_onnx(features, model_path)
        
    except Exception as e:
        print(f"[ERROR] Classification error: {str(e)}")
        return None
        
    except Exception as e:
        print(f"[ERROR] Classification error: {str(e)}")
        return None


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os

app = FastAPI()

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    try:
        # Ruta para el archivo temporal
        temp_file_path = f"temp_{file.filename}"
        # Ruta del modelo
        model_path = "models/random_forest_model.onnx"
        
        print(f"[INFO] Procesando archivo: {file.filename}")
        
        try:
            # Guardar archivo
            with open(temp_file_path, "wb") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
            print("[INFO] Archivo guardado temporalmente")
            
            # Verificar modelo
            if not os.path.exists(model_path):
                print(f"[ERROR] Modelo no encontrado: {model_path}")
                return JSONResponse(
                    content={"error": f"Model not found at {model_path}"},
                    status_code=404
                )
            
            # Extraer características (ajusta esta función a tu caso)
            features = extract_features(temp_file_path)
            if features is None:
                print("[ERROR] Fallo en extracción de características")
                return JSONResponse(
                    content={"error": "Feature extraction failed"},
                    status_code=400
                )
            
            # Predicción (ajusta esta función a tu caso)
            predicted_class, confidence = predict_with_onnx(features, model_path)
            
            if predicted_class is None:
                print("[ERROR] La predicción retornó None")
                return JSONResponse(
                    content={"error": "Prediction failed"},
                    status_code=400
                )
            
            print(f"[INFO] Predicción exitosa: clase={predicted_class}, confianza={confidence}")
            return {
                "status": "success",
                "predicted_class": int(predicted_class),
                "confidence": float(confidence)
            }
                
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print("[INFO] Archivo temporal eliminado")
                
    except Exception as e:
        print(f"[ERROR] Error inesperado: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


# Para pruebas locales
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
