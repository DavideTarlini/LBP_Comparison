import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("speedups4.csv", header=None)

colnames = ["size", "block size", "Global ns", "Global s", "Pad ns", "Pad s", "LBP ns", "LBP s"]
df = df.set_axis(colnames, axis="columns")

sizes = df["size"].unique()
block_sizes = df["block size"].unique()

plt.figure(figsize=(10, 6))
for size in sizes:
    subset = df[df["size"] == size]
    plt.plot(subset["block size"], subset["Global ns"], marker='o', label=f"Image size {size}")
plt.xlabel("Block Size")
plt.ylabel("Global Speedup")
plt.xscale("log", base=2)
plt.legend(title="Image Size")
plt.grid(True)
plt.tight_layout()

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "glb_ns.pdf"))

plt.figure(figsize=(10, 6))
for size in sizes:
    subset = df[df["size"] == size]
    plt.plot(subset["block size"], subset["Pad ns"], marker='o', label=f"Image size {size}")
plt.xlabel("Block Size")
plt.ylabel("Kernel Speedup")
plt.xscale("log", base=2)
plt.legend(title="Image Size")
plt.grid(True)
plt.tight_layout()

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "lbp_ns.pdf"))