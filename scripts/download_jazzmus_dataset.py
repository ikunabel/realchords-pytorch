import ast
from pathlib import Path
from datasets import load_dataset


DATASET_NAME = "PRAIG/JAZZMUS"
OUTPUT_DIR = Path("data/jazzmus/")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving to: {OUTPUT_DIR.absolute()}")

# Load dataset
ds = load_dataset(DATASET_NAME)

print(f"Total items: {len(ds['train'])}")

# Load and save all items
for idx, item in enumerate(ds["train"]):
    print(f"Processing item {idx + 1}/{len(ds['train'])}")
    
    # Parse annotation
    annotation = ast.literal_eval(item["annotation"])
    encoding = annotation["encodings"]
    
    mxml_encoding = encoding["musicxml"]
    kern_encoding = encoding["**kern"]
    
    # Save MusicXML
    with open(OUTPUT_DIR / f"{idx:04d}.musicxml", "w") as f:
        f.write(mxml_encoding)
    
    # Save Kern
    with open(OUTPUT_DIR / f"{idx:04d}.krn", "w") as f:
        f.write(kern_encoding)

print(f"\nSaved {len(ds['train'])} items to {OUTPUT_DIR}")

