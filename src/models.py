import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    select_largest=True,
    keep_all=False,
    device=device
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)
