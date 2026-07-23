# AI Engineering Standards

To ensure OmniID operates as a professional AI research repository focused on multimodal representation learning, all contributors must adhere to the following strict engineering standards:

## 1. Reproducible Experiments
- All experiments must be perfectly reproducible from scratch.
- Seeds must be fixed for data sampling, model initialization, and hardware-specific non-determinism (e.g., cuDNN).

## 2. Deterministic Training
- Ensure that the training loops and distributed setups avoid non-deterministic operations where possible.
- Document any operations that break exact determinism.

## 3. Mixed Precision
- Models must support mixed precision training (e.g., FP16, BF16) natively to optimize memory usage and throughput without sacrificing representation quality.

## 4. Dataset Versioning
- All raw, processed, and synthetic datasets must be strictly versioned using tools like DVC or internal manifests.
- Unversioned data is prohibited.

## 5. Model Versioning
- Checkpoints and final model weights must follow strict semantic versioning.

## 6. Experiment Tracking
- Every training run, fine-tuning task, or ablation study must be tracked via MLflow, Weights & Biases (W&B), or an equivalent MLOps tool.
- Hyperparameters, hardware metrics, and loss curves must be logged.

## 7. Model Cards
- Every released model or major iteration must include a detailed Model Card documenting architecture, intended use, biases, training data, and limitations.

## 8. Dataset Cards
- Every dataset created or curated for this project must feature a Dataset Card outlining collection methodology, privacy considerations, and distributions.

## 9. Benchmark Reports
- Performance metrics on standard evaluation benchmarks (e.g., retrieval, verification, latency, throughput) must be consistently generated and published as Benchmark Reports.
