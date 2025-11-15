import gdown
import os

def build_model():
    id_MobileNetV3 = "146aT2Ze5KwgOj-ooIWDn7Uw8YN4yTxPE"
    id_VisionTriX = "1CYxcJ3HQBO_mhiXFhd-pmdvszQrN55ij"

    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    path_MobileNetV3 = os.path.join(weights_dir, "MobileNetV3.pth")
    path_VisionTriX = os.path.join(weights_dir, "Hybrid-CDCN-ResViT.pth")

    model1 = f"https://drive.google.com/uc?id={id_MobileNetV3}"
    model2 = f"https://drive.google.com/uc?id={id_VisionTriX}"

    print("Downloading model MobileNetV3...")
    gdown.download(model1, path_MobileNetV3, quiet=False)
    print("Saved to:", path_MobileNetV3)

    print("Downloading model Hybrid-CDCN-ResViT...")
    gdown.download(model2, path_VisionTriX, quiet=False)
    print("Saved to:", path_VisionTriX)

if __name__ == "__main__":
    build_model()
