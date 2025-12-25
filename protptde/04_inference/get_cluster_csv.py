import click
import pandas as pd


@click.command()
@click.option("--mut_counts", required=True, type=int)
def main(mut_counts):
    df = pd.read_csv(f"predicted_sorted_mut_counts_{mut_counts}.csv").set_index("mut_name")

    df["positions"] = df.index.map(lambda mut_name: tuple(sorted([int(x[1:-1]) for x in mut_name.split(",")])))
    group_stats = df.groupby("positions")["pred"].agg(["mean", "std", "count"]).rename(columns={"mean": "mean_pred", "std": "std_pred", "count": "mut_count"})

    best_idx = df.groupby("positions")["pred"].idxmax()
    best = df.loc[best_idx].reset_index()
    best = best[["positions", "mut_name", "pred"]].rename(columns={"mut_name": "best_mut_name", "pred": "best_pred"})

    clustered = group_stats.reset_index().merge(best, on="positions", how="left")
    clustered = clustered.sort_values(["mean_pred", "std_pred"], ascending=[False, True])
    clustered.to_csv(f"predicted_sorted_mut_counts_{mut_counts}_clustered.csv", index=False)


if __name__ == "__main__":
    main()
