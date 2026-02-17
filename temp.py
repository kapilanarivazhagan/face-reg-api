from src.verification import verify_face

with open(r"C:\Users\SPURGE\Downloads\ph-1735367417935-1.webp", "rb") as f1, open(r"C:\Users\SPURGE\Downloads\17644793443994.jpg", "rb") as f2:
    result = verify_face(f1, f2)

print("\nRESULT:\n", result)
