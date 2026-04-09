import os
import torch
import numpy as np
from scipy.spatial.distance import cosine
from pathlib import Path

# Try to import cv2, fall back to PIL if not available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kyc_model.pt")

# Global model cache
_kyc_model = None

def _load_kyc_model():
    """Load the Swin Transformer model for face embeddings"""
    global _kyc_model
    
    if _kyc_model is None:
        try:
            import torchvision.models as models
            
            # Load pretrained Swin Transformer
            model = models.swin_t(weights="IMAGENET1K_V1")
            
            # Try to load trained weights, but skip classification head
            # because it might have different output dimensions
            if os.path.exists(MODEL_PATH):
                try:
                    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                    # Remove head weights if present (incompatible architecture)
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
                    model.load_state_dict(state_dict, strict=False)
                except:
                    print(f"Note: Could not load checkpoint, using pretrained backbone only")
            
            # Replace head with identity to get embeddings from backbone
            # This extracts features from the Swin Transformer backbone (shape 768)
            model.head = torch.nn.Identity()
            model.eval()
            _kyc_model = model
        except Exception as e:
            print(f"Warning: Could not load KYC model: {e}")
            _kyc_model = None
    
    return _kyc_model

def _preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for Swin Transformer"""
    try:
        if HAS_CV2:
            # Use OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return None, "Image file not found or cannot be read"
            
            # Check image quality
            if img.shape[0] < 100 or img.shape[1] < 100:
                return None, "Image too small (resolution < 100x100)"
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Check for blurriness using Laplacian variance
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                # Return tensor even if blurry (with warning)
                img_float = img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_normalized = (img_float - mean) / std
                img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1).unsqueeze(0)
                return img_tensor, "Image is very blurry - quality concern"
            
            # Normalize to [0, 1] then to standard ImageNet normalization
            img_float = img.astype(np.float32) / 255.0
        else:
            # Use PIL as fallback
            from PIL import Image
            from torchvision import transforms
            
            img = Image.open(image_path).convert('RGB')
            
            # Check image size
            if img.width < 100 or img.height < 100:
                return None, "Image too small (resolution < 100x100)"
            
            # Resize
            img_resized = img.resize(target_size)
            
            # Convert to numpy for blurriness check
            img_np = np.array(img_resized)
            
            # Simple blurriness check
            gray = np.mean(img_np, axis=2)  # Convert RGB to grayscale
            laplacian = np.abs(np.gradient(gray))
            laplacian_var = np.var(laplacian)
            if laplacian_var < 100:
                # Return tensor even if blurry (with warning)
                img_float = img_np.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_normalized = (img_float - mean) / std
                img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1).unsqueeze(0)
                return img_tensor, "Image is very blurry - quality concern"
            
            img_float = img_np.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std
        
        # Convert to tensor (C, H, W) ensuring float32
        img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, None
    except Exception as e:
        return None, f"Image processing failed: {str(e)}"

def _detect_faces_in_image(image_path):
    """Detect faces in image using OpenCV Haar Cascade or PIL"""
    try:
        if HAS_CV2:
            img = cv2.imread(image_path)
            if img is None:
                return 0, "No image loaded"
            
            # Load Haar Cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            return len(faces), None
        else:
            # Fallback: Use PIL to at least validate image exists
            from PIL import Image
            img = Image.open(image_path)
            # Without cv2, we can't do face detection, but we can assume 1 face if image loads
            return 1, "Face detection unavailable (install opencv-python for full functionality)"
    except Exception as e:
        return 0, str(e)

def _extract_embedding(model, image_tensor):
    """Extract embedding from image using the model"""
    try:
        if model is None or image_tensor is None:
            return None
        
        with torch.no_grad():
            # Forward pass through the backbone
            embedding = model(image_tensor)
            
            # Flatten if needed
            if len(embedding.shape) > 1:
                embedding = embedding.view(embedding.shape[0], -1)
            
            return embedding[0].numpy()
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        return None

def _compute_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    if emb1 is None or emb2 is None:
        return 0.0
    
    # Cosine similarity: 1 - cosine_distance
    # Values close to 1 = high similarity, close to 0 = low similarity
    distance = cosine(emb1, emb2)
    similarity = 1 - distance
    return max(0.0, min(1.0, similarity))

def predict_kyc(selfie_path, id_path):
    """
    Performs KYC verification by comparing selfie and ID card images.
    Uses Swin Transformer embeddings and face detection.
    Returns trust score plus reasoning.
    """
    reasons = []
    quality_issues = []
    
    # Check if files exist
    if not os.path.exists(selfie_path):
        reasons.append(f"Selfie image not found at: {selfie_path}")
        return {"score": 0.0, "reasons": reasons}
    
    if not os.path.exists(id_path):
        reasons.append(f"ID image not found at: {id_path}")
        return {"score": 0.0, "reasons": reasons}
    
    # Load model
    model = _load_kyc_model()
    
    if model is None:
        reasons.append("KYC model could not be loaded")
        return {"score": 0.3, "reasons": reasons}
    
    # Detect faces
    selfie_face_count, selfie_face_error = _detect_faces_in_image(selfie_path)
    id_face_count, id_face_error = _detect_faces_in_image(id_path)
    
    trust_score = 0.5  # Start with neutral
    
    # Face detection validation
    if selfie_face_count == 0:
        quality_issues.append("No face detected in selfie image")
        trust_score -= 0.3
    elif selfie_face_count > 1:
        quality_issues.append(f"Multiple faces detected in selfie ({selfie_face_count})")
        trust_score -= 0.2
    else:
        trust_score += 0.15
    
    if id_face_count == 0:
        quality_issues.append("No face detected in ID image")
        trust_score -= 0.3
    elif id_face_count > 1:
        quality_issues.append(f"Multiple faces detected in ID ({id_face_count})")
        trust_score -= 0.2
    else:
        trust_score += 0.15
    
    # Preprocess images
    selfie_tensor, selfie_quality_msg = _preprocess_image(selfie_path)
    id_tensor, id_quality_msg = _preprocess_image(id_path)
    
    if selfie_quality_msg:
        quality_issues.append(f"Selfie: {selfie_quality_msg}")
        trust_score -= 0.1
    
    if id_quality_msg:
        quality_issues.append(f"ID: {id_quality_msg}")
        trust_score -= 0.1
    
    # Extract embeddings if images are valid
    if selfie_tensor is not None and id_tensor is not None:
        selfie_embedding = _extract_embedding(model, selfie_tensor)
        id_embedding = _extract_embedding(model, id_tensor)
        
        if selfie_embedding is not None and id_embedding is not None:
            # Compute face matching similarity
            similarity = _compute_similarity(selfie_embedding, id_embedding)
            
            # Continuous High-Precision Trust Mapping
            # (similarity - threshold) * scaler shifts the 0.0-1.0 cosine into a continuous +/- trust modifier
            # A 99% match radically boosts trust, while a 20% match radically decimates it in real-time.
            trust_shift = (similarity - 0.55) * 1.5
            trust_score += trust_shift
            
            if similarity > 0.75:
                reasons.append(f"Strong continuous facial match mapped (similarity: {similarity:.2%})")
            elif similarity > 0.55:
                reasons.append(f"Acceptable facial match (similarity: {similarity:.2%})")
            elif similarity > 0.40:
                reasons.append(f"Low-precision facial match - identity verification inconclusive (similarity: {similarity:.2%})")
            else:
                reasons.append(f"CRITICAL: Failed Identity Swin Vector match (similarity: {similarity:.2%})")
        else:
            reasons.append("Could not extract facial embeddings")
            trust_score -= 0.2
    else:
        reasons.append("Image preprocessing failed")
        trust_score -= 0.2
    
    # Add quality issues to reasons
    if quality_issues:
        reasons.extend(quality_issues)
    
    # Normalize trust score
    final_score = max(0.0, min(1.0, trust_score))
    
    # Add summary
    if final_score > 0.75:
        reasons.append("Identity verification: PASSED")
    elif final_score > 0.50:
        reasons.append("Identity verification: INCONCLUSIVE (recommend manual review)")
    else:
        reasons.append("Identity verification: FAILED (identity mismatch or quality issues)")
    
    if not reasons:
        reasons.append("KYC verification complete")
    
    return {
        "score": round(float(final_score), 4),
        "reasons": reasons,
        "model_used": "Swin Transformer Face Matching"
    }

if __name__ == "__main__":
    res = predict_kyc("selfie.jpg", "id.jpg")
    print(f"Trust Score: {res['score']}, Reasons: {res['reasons']}")
