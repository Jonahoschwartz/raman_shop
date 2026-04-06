![alt text](../pngs/Ajitama.png "Title")

## Tools to Turn You into a PyMOL Pro

This package is different from the other ones in Raman Shop. The purpose of this package is to be loaded into your pymolrc to give you tons of tools to improve your PyMOL experience! To use, go to `File > Edit pymolrc` then add `run path/to/pymol_utilities.py`. Reload and get PyMOLing!

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

## Color Palettes

All palette functions register named colors into PyMOL. Once called, use the color names directly with `color`, `cmd.color()`, etc.

### `get_colors_rf_diffusion()`
Pastel RFdiffusion-style colors for protein design visualizations. Includes `good_teal`, `good_pink`, `good_blue`, `good_gray`, etc.

**Example:**
```python
get_colors_rf_diffusion()
color good_teal, chain A
```

---

### `get_colors_okabe_ito()`
8-color palette by Okabe & Ito (2002), designed to be distinguishable under deuteranopia, protanopia, and tritanopia. A gold standard for accessible scientific figures.

Includes: `oi_black`, `oi_orange`, `oi_sky_blue`, `oi_green`, `oi_yellow`, `oi_blue`, `oi_vermillion`, `oi_pink`

**Example:**
```python
get_colors_okabe_ito()
color oi_vermillion, mutantA   # = #D55E00
color oi_blue, wildtype        # = #0072B2
```

> 💡 This palette matches the default Okabe-Ito assignments commonly used in Python (matplotlib, seaborn). If your plots use `#D55E00`, `#CC79A7`, `#009E73`, etc., call this function to keep your PyMOL figures consistent.

---

### `get_colors_ibm()`
5-color palette from the IBM Design Language, optimized for colorblind accessibility.

Includes: `ibm_blue`, `ibm_purple`, `ibm_magenta`, `ibm_orange`, `ibm_gold`

**Example:**
```python
get_colors_ibm()
color ibm_magenta, chain B
```

---

### `get_colors_wong()`
7-color palette published by Bang Wong (2011) in *Nature Methods*. Closely related to Okabe-Ito (minus black); frequently cited in journal figure guidelines.

Includes: `wong_orange`, `wong_sky_blue`, `wong_green`, `wong_yellow`, `wong_blue`, `wong_vermil`, `wong_pink`

**Example:**
```python
get_colors_wong()
color wong_green, chain A
```

---

### `get_colors_paul_tol()`
6-color "Bright" scheme by Paul Tol, designed for high contrast in print, on screen, and under colorblindness simulation.

Includes: `tol_blue`, `tol_cyan`, `tol_green`, `tol_yellow`, `tol_red`, `tol_purple`

**Example:**
```python
get_colors_paul_tol()
color tol_red, binding_site
```

---

### `get_colors_tableau()`
The Tableau 10 categorical palette, widely used in data visualization (D3.js, Tableau).

Includes: `tab_blue`, `tab_orange`, `tab_red`, `tab_teal`, `tab_green`, `tab_yellow`, `tab_purple`, `tab_pink`, `tab_brown`, `tab_gray`

**Example:**
```python
get_colors_tableau()
color tab_teal, chain C
```

---

### `get_colors_mpl()`
Matplotlib's default `tab10` color cycle. Useful for keeping PyMOL figures consistent with Python plots.

Includes: `mpl_blue`, `mpl_orange`, `mpl_green`, `mpl_red`, `mpl_purple`, `mpl_brown`, `mpl_pink`, `mpl_gray`, `mpl_yellow`, `mpl_cyan`

**Example:**
```python
get_colors_mpl()
color mpl_orange, helix1
```

---

### `get_colors_silas()`
Custom 8-color palette spanning cool teals through warm purples, designed for protein visualization.

Includes: `silas_teal`, `silas_seafoam`, `silas_sage`, `silas_gold`, `silas_amber`, `silas_coral`, `silas_rose`, `silas_lavender`

**Example:**
```python
get_colors_silas()
color silas_teal, chain A
color silas_coral, chain B
```

---

### `coloraf(selection="all")`
Colors a structure by B-factor using AlphaFold-style pLDDT confidence thresholds:
- **Blue** — >90 (very high confidence)
- **Cyan** — 70–90 (high confidence)
- **Yellow** — 50–70 (low confidence)
- **Orange** — ≤50 (very low confidence)

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
Sets lighting and rendering parameters for high-quality ray-traced images in the RFdiffusion style (disables specular highlights, sets ambient light, enables cartoon ring mode).

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
- `coloraf`
- `color_by_csv`
- `get_colors_rf_diffusion`
- `get_colors_okabe_ito`
- `get_colors_ibm`
- `get_colors_wong`
- `get_colors_paul_tol`
- `get_colors_tableau`
- `get_colors_mpl`
- `get_colors_silas`
- `get_lighting_rf_diffusion`

### Animation
- `spin_movie`

### Analysis
- `color_by_mutation`
- `cbce` (color by chains)
- `DNAselections`
- `DNAselections_nodisplay`
- `DNAselections_labelsonly`