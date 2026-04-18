import os

TRAINING_ROOT = "./training_data"
categories = ["Bee", "Cricket", "Grasshopper", "Mosquito"]

for cat in categories:
    path = os.path.join(TRAINING_ROOT, cat)
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith(".wav")]
        print(f"{cat}: {len(files)} files")
    else:
        print(f"{cat}: Directory not found")
