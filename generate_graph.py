import matplotlib.pyplot as plt

# 1. The Data (From your Table 6.2)
k_values = [1, 2, 4, 8, 16, 32, 64]
dense_mismatch = [0.56, 0.44, 0.40, 0.36, 0.24, 0.16, 0.16]
mmr_mismatch = [0.56, 0.44, 0.36, 0.32, 0.28, 0.16, 0.04]
hyde_mismatch = [0.56, 0.56, 0.56, 0.4933, 0.44, 0.36, 0.1733]

# 2. Setup the academic plot style
plt.figure(figsize=(9, 5.5)) # Standard academic ratio
plt.style.use('default') # Clean, unopinionated background

# 3. Plot each strategy with distinct colors, line styles, and markers for B&W readability
plt.plot(k_values, dense_mismatch, 
         color='#1f77b4', linestyle='-', linewidth=2.5, marker='o', markersize=8, label='Dense Similarity')

plt.plot(k_values, mmr_mismatch, 
         color='#ff7f0e', linestyle='--', linewidth=2.5, marker='s', markersize=8, label='MMR Search')

plt.plot(k_values, hyde_mismatch, 
         color='#2ca02c', linestyle='-.', linewidth=2.5, marker='^', markersize=8, label='HyDE Search')

# 4. Format the Axes
plt.xscale('log', base=2) # Represents the exponential doubling of k
plt.xticks(k_values, labels=[str(k) for k in k_values], fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0.0, 0.65) # Pad the top slightly for visual breathing room

# 5. Add Labels and Legends (No embedded title for academic formatting!)
plt.xlabel('Retrieval Depth (k-chunks)', fontsize=13, fontweight='bold', labelpad=10)
plt.ylabel('Document-Level Mismatch Rate (DRM)', fontsize=13, fontweight='bold', labelpad=10)
plt.legend(fontsize=12, loc='lower left')

# 6. Add clean gridlines to make it easy to read specific thresholds
plt.grid(True, which='major', axis='both', linestyle=':', linewidth=1.2, alpha=0.7, color='gray')

plt.tight_layout()

# 7. Export as high-quality vector graphics
output_pdf = "drm_curve.pdf"
output_svg = "drm_curve.svg"

plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_svg, format='svg', bbox_inches='tight')

print(f"Academic graphs successfully generated:\n - {output_pdf}\n - {output_svg}")
