# Contributing to SmartML & SmartEco

Thank you for your interest in contributing to **SmartML**, part of the **SmartEco** ecosystem.

SmartML is designed as a **fair, CPU-first, reproducible benchmarking system**.  
All contributions must align with these principles.

---

## ### SmartEco Benchmarks & Website Display

SmartEco publicly displays **benchmark results per dataset** on the SmartEco website.

For each dataset, the website will show:
- Dataset description
- Task type (classification / regression)
- Feature composition
- Models evaluated
- Performance and inference metrics

### Adding or Displaying a Dataset

If you want:
- A **new dataset** to be benchmarked
- A **dataset’s results** to appear on the SmartEco website

Please submit a **Pull Request** that includes:
- Dataset source and license
- Clear dataset description
- Task definition (classification / regression)
- Feature types (numerical / categorical)
- Target definition
- Any known caveats or biases

Datasets must be:
- Publicly usable
- Legally redistributable or referenceable
- Suitable for leakage-free benchmarking

---

## ### Adding New Models

SmartML only exposes models that:
- Run on **CPU**
- Are **deterministic or controllable**
- Can be benchmarked fairly using defaults

To add a new model, submit a **Pull Request** that includes:
- Clear model description
- Supported tasks (classification / regression)
- Dependency list
- Default hyperparameters
- Expected preprocessing requirements
- CPU compatibility confirmation

Models must:
- Follow SmartML’s availability detection rules
- Fail gracefully when dependencies are missing
- Not assume GPU, Linux-only, or proprietary environments

---

## ### Hyperparameter Tuning & Cross-Validation

SmartML uses **fixed default hyperparameters by design**.

If you want to:
- Add **hyperparameter tuning**
- Add **cross-validation (CV)**
- Introduce **task-specific optimization logic**

Please:
- Open an **Issue** first to discuss scope and design
- Or submit a **Pull Request** with a clear and isolated implementation

Tuning-related contributions must include:
- Clear definition of tuning strategy
- Reproducibility guarantees
- Separation from default benchmarking paths
- Documentation of trade-offs and limitations

SmartML will not merge tuning logic that:
- Breaks comparability
- Introduces silent leakage
- Favors specific models without justification

---

## ### Benchmarking Changes

Any change that affects benchmarking behavior must:
- Preserve determinism
- Preserve identical train/test splits
- Preserve preprocessing consistency
- Clearly document what changed and why

If results change, the reason must be explainable.

---

## ### Code Quality Expectations

All contributions should:
- Be modular and readable
- Follow existing project structure
- Avoid unnecessary abstraction
- Include comments where behavior is non-obvious
- Avoid introducing platform-specific assumptions

---

## ### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make focused, well-documented changes
4. Submit a Pull Request with:
   - Clear motivation
   - What is added or changed
   - Why it aligns with SmartML principles

Low-effort or unclear PRs may be closed without review.

---

## ### Final Notes

SmartML values:
- Transparency over cleverness
- Reproducibility over raw scores
- Fairness over optimization tricks

If you’re unsure whether a contribution fits,  
**open an issue first** — discussion is always welcome.
