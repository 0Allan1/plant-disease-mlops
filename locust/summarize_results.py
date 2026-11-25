"""Turn results/containers_N_stats.csv into a Markdown table for the README."""
from pathlib import Path

import pandas as pd

rows = []
for f in sorted(Path("results").glob("containers_*_stats.csv")):
    n = f.stem.split("_")[1]
    df = pd.read_csv(f)
    agg = df[df["Name"] == "Aggregated"].iloc[0]
    rows.append({"API containers": n, "Requests": int(agg["Request Count"]), "Failures": int(agg["Failure Count"]),
                 "Req/s": round(agg["Requests/s"], 1), "Avg (ms)": round(agg["Average Response Time"], 1),
                 "Median (ms)": agg["50%"], "95% (ms)": agg["95%"], "99% (ms)": agg["99%"],
                 "Max (ms)": round(agg["Max Response Time"], 1)})
table = pd.DataFrame(rows)
md = table.to_markdown(index=False) if hasattr(table, "to_markdown") else table.to_string(index=False)
Path("results/summary.md").write_text(md)
print(md)
