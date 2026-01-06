from SmartEco.SmartML import load_dataset, run_training, SmartML_Inspect

SmartML_Inspect()


X, y = load_dataset(
    openml_id=562,
    target="usr",
    subset=None,
)

print(f"Dataset loaded: X={X.shape}, y={y.shape}")


results = run_training(
    X_df=X,
    y_ser=y,
    task="regression",
    exclude=[
        "tabnet",
        "nam",
    ],
    output_csv="results/benchmark.txt",
)

print(
    results.sort_values(
        "r2",
        ascending=False,
    )
)
