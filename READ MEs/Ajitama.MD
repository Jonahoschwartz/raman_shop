![alt text](../pngs/Ajitama.png "Title")

## Tools to Turn You into a PyMOL Pro

This package is different from the other ones in Raman Shop. The purpose of this package is to be loaded into your pymolrc to give you tons of tools to improve your PyMOL experience! To use, go to `File > Edit pymolrc` then add `run path/to/Ajitama.py`. Reload and get PyMOLing!

---

## Color Utilities

### `hex_to_rgb(hex_code)`
Converts a hex color (e.g. `"#4FB9AF"`) to a normalized RGB triplet (0–1 scale) for PyMOL coloring.

**Example:**
```python
hex_to_rgb("#4FB9AF")  # Returns [0.31, 0.725, 0.686]
```

---

### `color_by_hex(color_name, hex_code, selection="all")`
Defines a new PyMOL color from a hex code and applies it to the specified selection.

**Example:**
```python
color_by_hex("my_teal", "#4FB9AF", "chain A")
```

---

### `get_colors_rf_diffusion()`
Registers a palette of pastel RFdiffusion-style custom colors for use in PyMOL. Includes colors like `good_teal`, `good_pink`, `good_blue`, etc.

**Example:**
```python
get_colors_rf_diffusion()
color good_teal, chain A
```

---

### `coloraf(selection="all")`
Colors a structure by B-factor using AlphaFold-style confidence thresholds:
- **Blue** for >90 (very high confidence)
- **Cyan** for 70–90 (high confidence)
- **Yellow** for 50–70 (low confidence)
- **Orange** for ≤50 (very low confidence)

**Example:**
```python
coloraf myprotein
```

---

### `color_by_csv(csv_path, selection="all", chain=None, colors="blue_white_red")`
Colors a protein using per-residue scores from a CSV file (one value per line). Loads values into the B-factor field and applies a color spectrum.

**Example:**
```python
color_by_csv "scores.csv", myprotein, chain="A", colors="rainbow"
```

---

## Rendering & Visualization

### `get_lighting_rf_diffusion()`
Sets lighting and rendering parameters for high-quality images in the RFdiffusion-style (e.g., disables specular highlights, sets ambient light, enables cartoon ring mode).

**Example:**
```python
get_lighting_rf_diffusion()
ray
```

---

## Animation

### `spin_movie(spins=1, seconds=5, fps=30)`
Creates a smooth Y-axis rotation movie in PyMOL. You can export it using `mpng` or `movie.produce`.

**Parameters:**
- `spins`: Number of full 360° rotations
- `seconds`: Duration of the movie
- `fps`: Frames per second

**Example:**
```python
spin_movie 2, 10, 30  # 2 spins over 10 seconds at 30 fps
mpng frames/frame_.png  # Export frames
```

---

## Mutation Analysis

### `color_by_mutation(obj1, obj2, waters=0, labels=0)`
Aligns two protein structures, superimposes them, and highlights mutations. Mutated residues are colored according to their similarity in the BLOSUM90 substitution matrix (blue = similar, red = dissimilar).

**Parameters:**
- `obj1`, `obj2`: Protein objects or selections to compare
- `waters`: If 1, shows waters colored by structure
- `labels`: If 1, labels mutated sidechains

**Color scheme:**
- **Cyan**: Mutated sidechains in obj1
- **Blue→Red**: Mutated sidechains in obj2 (by BLOSUM90 similarity)
- **Wheat**: Conserved/non-mutated regions
- **Gray**: Unaligned regions

**Example:**
```python
color_by_mutation protein1, protein2, waters=1, labels=1
```

---

## DNA Analysis

### `color_by_chains()` (registered as `cbce`)
Colors each chain uniquely across all objects. Quick shortcut for `util.color_chains`.

**Example:**
```python
cbce
```

---

### `dna_selections(display='all')` (registered as `DNAselections`)
Creates useful selections for protein-DNA interface analysis.

**Parameters:**
- `display`: Display mode - `'all'`, `'labels'`, or `'none'`

**Selections created:**
- `DNA`: All DNA residues
- `DNAbases`: DNA bases only (no backbone)
- `DNAbb`: DNA backbone only
- `sc_base`: Protein sidechains within 7Å of DNA bases
- `dna_h2o`: Waters within 3.6Å of DNA bases
- `pbb`: Protein backbone (C, N, CA atoms)

**Display (when `display='all'`):**
- DNA bases and nearby sidechains: sticks
- DNA backbone: ribbon
- Protein: cartoon
- Waters near DNA: spheres (marine color, transparent)

**Example:**
```python
DNAselections all
DNAselections_nodisplay     # Create selections without display
DNAselections_labelsonly    # Create selections with labels only
```

---

## Registered PyMOL Commands

All major functions are registered as commands and can be used directly in PyMOL:

### Color & Rendering
- `color_by_hex`
- `get_colors_rf_diffusion`
- `get_lighting_rf_diffusion`
- `coloraf`
- `color_by_csv`

### Animation
- `spin_movie`

### Analysis
- `color_by_mutation`
- `cbce` (color by chains)
- `DNAselections`
- `DNAselections_nodisplay`
- `DNAselections_labelsonly`
