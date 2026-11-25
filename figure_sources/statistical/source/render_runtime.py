"""Replay the archived Fig. 9 timing labels and batch-derived medians."""
import inspect
import pandas as pd
import render_figures as base


def table(name):
    if name == "model_only_attributed_archive.csv":
        data = base.read("runtime_corrected_panel_c_source.csv").set_index("variant")
        cohorts = ["baseline", "node_only_nodrop", "full_nodrop"]
        return pd.DataFrame({"ms": data.loc[cohorts, "median_ms_per_graph"].to_numpy()})
    return base.read(name)


if __name__ == "__main__":
    # Match the archived version in technical_audit_20261023/render_verified_edits.py.
    source = inspect.getsource(base.timing)
    for old, new in {
        '"(c) Forward logs"': '"(c) Batched eval logs"',
        '"Attributed ms/graph"': '"Batch-derived ms/graph"',
        '"Audit",ORANGE': '"Metrics / audit",ORANGE',
    }.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    scope = dict(vars(base), read=table)
    exec(compile(source, __file__, "exec"), scope)
    scope["timing"]()
