For AR_DIFF with EqM, it might make sense to supervise the loss post update, i.e., reconstruction objective on the clean latent, rather than the velocity. 
0.73 on slow h100 for some reason
Check how time conditioning applies to ARDIT on the original repo. Right now we pass zeros, but wouldnt that mean that for those tokens, we gate with 0 and essentially delete the conditoning??
What changes as you increase ardiff_step is the number of iterations. Roughly:

iterations ≈ S + ardiff_step * (L - 1)
So for L=256 and S=20:

ardiff_step=0 → 20 iterations
ardiff_step=1 → ~275 iterations
ardiff_step=2 → ~530 iterations
ardiff_step=20 → ~5,140 iterations (pure AR)

