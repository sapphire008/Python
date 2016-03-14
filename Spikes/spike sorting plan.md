## Spike sorting and spike classification pipeline

1. Detection of spikes via wavelet decomposition.
2. Separation of spikes from noise using two methods:
  * Generative model, such as Mixture of Gaussian
  * Superparamagnetic clustering --> advnatage: not assuming Gaussian. Maybe advantages in some cases
3. Generate spike templates based on ICA/PCA wavelet components (CWT/DWT)
4. Consider selection of wavelet basis (.e.g bior1.2/3?)
