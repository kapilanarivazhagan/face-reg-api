import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Force CPU (Render free tier has no GPU)
device = "cpu"

_mtcnn = None
_resnet = None

def get_models():
    global _mtcnn, _resnet

    if _mtcnn is None:
        print("Loading MTCNN model...")
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            select_largest=True,
            keep_all=False,
            device=device
        )

    if _resnet is None:
        print("Loading FaceNet model...")
        _resnet = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(device)

    return _mtcnn, _resnet