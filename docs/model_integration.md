# Model Integration

The open-source ZeoAgent release does not bundle the private diffusion model or descriptor dataset used in the internal project.

To enable diffusion prediction, provide your own predictor backend and your own feature source.

At minimum, your integration should:

1. accept a framework identifier
2. accept optional temperature and loading overrides
3. return a scalar diffusion coefficient

The public `diffusion_predictor` module is intentionally an interface placeholder. Replace it with your own implementation if you want this capability in your deployment.
