# Santa 2025 – Christmas Tree Packing

**Competition**: [Kaggle Santa 2025](https://www.kaggle.com/competitions/santa-2025)

Approach (video mode):
<video src="https://github.com/user-attachments/assets/1a918075-dc95-4e46-b4f3-7d2a55c0115d" controls width="100%"></video>

---

## Problem

Pack 1 through 200 rotated Christmas tree polygons as tightly as possible. The score for each group of *n* trees is the area of their bounding square divided by *n*, so smaller and denser packings win.

## Approach

### 1. Bottom-Left Fill (BLF) placement heuristic

`blf.py` implements a BLF algorithm adapted for arbitrary polygon shapes. Given a list of trees (each with a rotation angle), BLF greedily places each tree as far left and as far down as possible within an incrementally grown bounding box.

### 2. Hybrid Genetic Algorithm (HGA)

`hga_submission.py` solves the problem incrementally: it solves for 1 tree, then reuses that solution when solving for 2, and so on up to 200.

Each individual in the GA is a vector of rotation angles. BLF is called as a decoder to translate angles into actual (x, y) placements.

- **Crossover**: arithmetic crossover (weighted average of parent angles)
- **Mutation**: angle-adjustment mutation (±30° per gene, 25% gene probability)
- **Selection**: ranked roulette-wheel selection (selective pressure 1.5)
- **Replacement**: elitist replacement
- **Parallelism**: `joblib` parallelises BLF decoding across all individuals each generation

## Files

| File | Description |
|---|---|
| `blf.py` | Bottom-Left Fill placement algorithm |
| `tree_utils.py` | `ChristmasTree` polygon definition and submission file generation |
| `genetic_algorithm_operators.py` | Crossover, mutation, selection, and replacement operators |
| `hga_submission.py` | Main HGA loop with incremental solving (1 → 200 trees) |
| `mix_simulated_annealing.ipynb` | SA mutation for HGA (not used) |
| `visualization.py` | Manim-based animation of the final packing |


### 3. References

- Original paper from which my solution is based:  
Wu, Q., Yang, W., Zhang, Q. et al. Two-dimensional nesting system based on hybrid genetic algorithm. Wuhan Univ. J. Nat. Sci. 14, 60–64 (2009). https://doi.org/10.1007/s11859-009-0113-0
- [Problem definition and discussion.](https://stackoverflow.com/questions/2675123/nesting-maximum-amount-of-shapes-on-a-surface)
- [PhD's thesis on nesting problems and algorithms](https://www.uv.es/marsyan/docs/thesis.pdf)
- [Improved No-Fit Polygon algorithm](https://www.mdpi.com/2227-7390/10/16/2941)






 

