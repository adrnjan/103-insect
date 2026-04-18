import os
import shutil

# --- CONFIGURE THIS ---
SOURCE_DIR = "."  # Root folder containing all the insect subfolders
OUTPUT_DIR = "./categorized"  # Where the 5 category folders will be created
# ----------------------

# Mapping from folder keywords → category
CATEGORY_MAP = {
    "Cricket": [
        "cricket", "conehead"
    ],
    "Grasshopper": [
        "grasshopper", "katydid"
    ],
    "Bee": [
        "bee", "bumblebee", "honeybee"
    ],
    "Mosquito": [
        "mosquito"
    ],
    "Background_Noise": [
        "fly", "housefly", "cicada", "drone"
    ],
}

def get_category(folder_name):
    name_lower = folder_name.lower()
    for category, keywords in CATEGORY_MAP.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return None  # Unmapped folders are skipped

def main():
    # Create output category folders
    for category in CATEGORY_MAP:
        os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)

    moved = 0
    skipped = 0
    unmatched = []

    for folder_name in os.listdir(SOURCE_DIR):
        folder_path = os.path.join(SOURCE_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        category = get_category(folder_name)

        if category is None:
            unmatched.append(folder_name)
            continue

        dest_folder = os.path.join(OUTPUT_DIR, category)

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".wav"):
                src = os.path.join(folder_path, file_name)
                # Prefix filename with source folder to avoid collisions
                new_name = f"{folder_name}__{file_name}"
                dst = os.path.join(dest_folder, new_name)

                if os.path.exists(dst):
                    print(f"  [SKIP - exists] {new_name}")
                    skipped += 1
                    continue

                shutil.move(src, dst)
                print(f"  [MOVED] {folder_name}/{file_name} → {category}/{new_name}")
                moved += 1

    print(f"\nDone! Moved: {moved} | Skipped (already exists): {skipped}")

    if unmatched:
        print(f"\nUnmatched folders (not moved):")
        for f in unmatched:
            print(f"   - {f}")

if __name__ == "__main__":
    main()