import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("speedups4.csv", header=None)

colnames = ["size", "block size", "Global ns", "Global s", "Pad ns", "Pad s", "LBP ns", "LBP s"]
df.columns = colnames

sizes = sorted(df["size"].unique())
block_sizes = sorted(df["block size"].unique())

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Plot LBP ns vs Image Size, with each block size as a separate curve
plt.figure(figsize=(10, 6))
for block_size in block_sizes:
    subset = df[df["block size"] == block_size]
    subset = subset.sort_values("size")
    plt.plot(subset["size"], subset["Global s"], marker='o', label=f"Block size {block_size}")

plt.xlabel("Image Size", fontsize=20)
plt.ylabel("Global Speedup", fontsize=20)
plt.xticks(fontsize=18)  
plt.yticks(fontsize=18)
plt.xscale("log", base=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "glb_s_vs_image_size.pdf"))
