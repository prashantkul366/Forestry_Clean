import os

# DON'T create a new root folder
BASE_DIR = "."   # current directory

structure = [
    "configs",
    "data",
    "models",
    "losses",
    "engine",
    "utils",
    "scripts"
]

files = {
    "configs": ["config.py"],
    "data": ["dataset.py", "transforms.py"],
    "models": ["model.py"],
    "losses": ["losses.py"],
    "engine": ["train.py", "validate.py", "metrics.py"],
    "utils": ["plotting.py", "threshold.py", "visualization.py"],
    "scripts": ["train.py"],
}

# Create folders + __init__.py
for folder in structure:
    folder_path = os.path.join(BASE_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)

    # make it a package
    init_file = os.path.join(folder_path, "__init__.py")
    if not os.path.exists(init_file):
        open(init_file, "w").close()

    # create files
    for f in files.get(folder, []):
        file_path = os.path.join(folder_path, f)
        if not os.path.exists(file_path):
            open(file_path, "w").close()

# root-level files
for root_file in ["README.md", "requirements.txt"]:
    if not os.path.exists(root_file):
        open(root_file, "w").close()

print("✅ Project structure created in current directory!")