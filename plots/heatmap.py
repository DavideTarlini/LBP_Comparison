import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("speedups4.csv", header=None)

colnames = ["size", "block size", "Global ns", "Global s", "Pad ns", "Pad s", "LBP ns", "LBP s"]
df = df.set_axis(colnames, axis="columns")

sizes = df["size"].unique()
block_sizes = df["block size"].unique()


global_heatmap_ns = df.pivot(index="size", columns="block size", values="Global ns")
global_heatmap_s = df.pivot(index="size", columns="block size", values="Global s")
pad_heatmap_ns = df.pivot(index="size", columns="block size", values="Pad ns")
pad_heatmap_s = df.pivot(index="size", columns="block size", values="Pad s")
lbp_heatmap_ns = df.pivot(index="size", columns="block size", values="LBP ns")
lbp_heatmap_s = df.pivot(index="size", columns="block size", values="LBP s")


vmin = min(global_heatmap_ns.min().min(), global_heatmap_s.min().min())
vmax = max(global_heatmap_ns.max().max(), global_heatmap_s.max().max())


plt.figure(figsize=(14, 6))
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
sns.heatmap(global_heatmap_ns, annot=True, fmt=".2f", cmap="YlOrRd", vmin=vmin, vmax=vmax, annot_kws={"size": 16})
plt.xlabel("Block Size", fontsize=16)
plt.ylabel("Image Size", fontsize=16)
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "heatmap_glns.pdf"))

plt.figure(figsize=(14, 6))
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
sns.heatmap(global_heatmap_s, annot=True, fmt=".2f", cmap="YlOrRd", vmin=vmin, vmax=vmax, annot_kws={"size": 16})
plt.xlabel("Block Size", fontsize=16)
plt.ylabel("Image Size", fontsize=16)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "heatmap_gls.pdf"))


vmin = min(pad_heatmap_ns.min().min(), pad_heatmap_s.min().min())
vmax = max(pad_heatmap_ns.max().max(), pad_heatmap_s.max().max())

plt.figure(figsize=(14, 6))
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
sns.heatmap(pad_heatmap_ns, annot=True, fmt=".2f", cmap="YlOrRd", vmin=vmin, vmax=vmax, annot_kws={"size": 16})
plt.xlabel("Block Size", fontsize=16)
plt.ylabel("Image Size", fontsize=16)
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "heatmap_padns.pdf"))

plt.figure(figsize=(14, 6))
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
sns.heatmap(pad_heatmap_s, annot=True, fmt=".2f", cmap="YlOrRd", vmin=vmin, vmax=vmax, annot_kws={"size": 16})
plt.xlabel("Block Size", fontsize=16)
plt.ylabel("Image Size", fontsize=16)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "heatmap_pads.pdf"))



vmin = min(lbp_heatmap_ns.min().min(), lbp_heatmap_s.min().min())
vmax = max(lbp_heatmap_ns.max().max(), lbp_heatmap_s.max().max())

plt.figure(figsize=(14, 6))
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
sns.heatmap(lbp_heatmap_ns, annot=True, fmt=".2f", cmap="YlOrRd", vmin=vmin, vmax=vmax, annot_kws={"size": 16})
plt.xlabel("Block Size", fontsize=16)
plt.ylabel("Image Size", fontsize=16)
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "heatmap_lbpns.pdf"))

plt.figure(figsize=(14, 6))
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)
sns.heatmap(lbp_heatmap_s, annot=True, fmt=".2f", cmap="YlOrRd", vmin=vmin, vmax=vmax, annot_kws={"size": 16})
plt.xlabel("Block Size", fontsize=16)
plt.ylabel("Image Size", fontsize=16)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "heatmap_lbps.pdf"))