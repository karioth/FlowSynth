
What changes as you increase ardiff_step is the number of iterations. Roughly:

iterations ≈ S + ardiff_step * (L - 1)
So for L=256 and S=20:

ardiff_step=0 → 20 iterations
ardiff_step=1 → ~275 iterations
ardiff_step=2 → ~530 iterations
ardiff_step=20 → ~5,140 iterations (pure AR)

