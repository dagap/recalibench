"""Report stage: artifacts*/ -> reports/report{,_mock}.html (one self-contained file)."""

from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from recalibench.core.paths import Paths
from recalibench.report.collect import ReportData, collect
from recalibench.report.render import render


def build_report(paths: Paths, cfg: DictConfig) -> tuple[Path, ReportData]:
    data = collect(paths)
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    page = render(data, OmegaConf.to_yaml(cfg, resolve=True), generated_at)
    out = paths.report_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out)
    return out, data
