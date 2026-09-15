from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class ReportStrategy(ABC):
    @abstractmethod
    def generate(self, results: dict, filepath: str | Path) -> None:
        """Serialise *results* and write to *filepath*."""

def _section(title: str, width: int = 80) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"

def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"

class TextReportStrategy(ReportStrategy):
    def generate(self, results: dict, filepath: str | Path) -> None:
        lines: list[str] = [
            "=" * 80,
            "          AUTOMATED EDA REPORT          ",
            "=" * 80,
        ]

        if meta := results.get("_pipeline_metadata"):
            lines.append(_section("PIPELINE METADATA"))
            lines.append(f"  Duration   : {meta.get('pipeline_duration_seconds', 'N/A')}s")
            lines.append(f"  Components : {meta.get('total_components', 'N/A')}")
            failed = meta.get("failed_components", [])
            lines.append(f"  Failed     : {', '.join(failed) if failed else 'none'}")

        if "num_rows" in results:
            lines.append(_section("DATASET OVERVIEW"))
            lines.append(f"  Rows          : {results['num_rows']:,}")
            lines.append(f"  Columns       : {results['num_cols']}")
            lines.append(f"  Memory        : {results['memory_usage_mb']:.2f} MB")
            lines.append(f"  Duplicate rows: {results.get('duplicate_rows', 'N/A')} ({results.get('duplicate_percentage', 'N/A')}%)")

        if "missing_data" in results:
            lines.append(_section("MISSING DATA"))
            missing = results["missing_data"]
            if not missing:
                lines.append("  ✓ No missing values detected.")
            else:
                lines.append(f"  {'Column':<30} {'Count':>8}  {'%':>7}")
                lines.append(f"  {'─'*30} {'─'*8}  {'─'*7}")
                for col, m in missing.items():
                    lines.append(f"  {col:<30} {m['count']:>8,}  {_fmt_pct(m['percentage']):>7}")

        if "outliers" in results:
            lines.append(_section("OUTLIERS  (IQR method)"))
            outliers = results["outliers"]
            if not outliers:
                lines.append("  ✓ No outliers detected at the current threshold.")
            else:
                lines.append(f"  {'Column':<20} {'Count':>8}  {'%':>7}  {'Lower':>12}  {'Upper':>12}")
                lines.append(f"  {'─'*20} {'─'*8}  {'─'*7}  {'─'*12}  {'─'*12}")
                for col, m in outliers.items():
                    lines.append(
                        f"  {col:<20} {m['outlier_count']:>8,}  {_fmt_pct(m['percentage']):>7}"
                        f"  {m['lower_bound']:>12.4f}  {m['upper_bound']:>12.4f}"
                    )

        if "target_distribution" in results:
            lines.append(_section("TARGET DISTRIBUTION"))
            dist = results["target_distribution"]
            if isinstance(dist, dict):
                ratio = results.get("class_imbalance_ratio")
                if ratio:
                    lines.append(f"  Class imbalance ratio: {ratio}:1")
                lines.append(f"\n  {'Class':<20} {'Count':>10}  {'%':>7}")
                lines.append(f"  {'─'*20} {'─'*10}  {'─'*7}")
                for label, stats in dist.items():
                    lines.append(f"  {str(label):<20} {stats['count']:>10,}  {_fmt_pct(stats['percentage']):>7}")
            else:
                lines.append(f"  {dist}")

        if "univariate_analysis" in results:
            lines.append(_section("UNIVARIATE ANALYSIS (Numerical)"))
            ua = results["univariate_analysis"]
            if isinstance(ua, dict):
                hdr = f"  {'Column':<20} {'Mean':>12}  {'Std':>12}  {'Skew':>9}  {'Kurt':>9}"
                lines.append(hdr)
                lines.append(f"  {'─'*20} {'─'*12}  {'─'*12}  {'─'*9}  {'─'*9}")
                for col, s in ua.items():
                    lines.append(
                        f"  {col:<20} {s['mean']:>12.4f}  {s['std']:>12.4f}"
                        f"  {s['skewness']:>9.4f}  {s['kurtosis']:>9.4f}"
                    )

        if "mutual_information_scores" in results:
            lines.append(_section("FEATURE IMPORTANCE  (Mutual Information)"))
            mi = results["mutual_information_scores"]
            if isinstance(mi, dict):
                lines.append(f"  {'Feature':<35} {'MI Score':>10}")
                lines.append(f"  {'─'*35} {'─'*10}")
                for col, score in mi.items():
                    bar = "█" * int(score * 20)
                    lines.append(f"  {col:<35} {score:>10.4f}  {bar}")
            else:
                lines.append(f"  {mi}")

        if "spearman_correlation_matrix" in results:
            lines.append(_section("HIGH CORRELATION PAIRS (Spearman)"))
            pairs = results.get("high_correlation_pairs", [])
            if not pairs:
                lines.append("  ✓ No highly correlated pairs > threshold found.")
            else:
                lines.append(f"  {'Feature A':<25} {'Feature B':<25} {'Spearman r':>12}")
                lines.append(f"  {'─'*25} {'─'*25} {'─'*12}")
                for p in pairs:
                    lines.append(f"  {p['feature_a']:<25} {p['feature_b']:<25} {p['spearman_r']:>12.4f}")

        if "bivariate_analysis" in results:
            lines.append(_section("BIVARIATE ANALYSIS (Mean by Class Excerpt)"))
            ba = results["bivariate_analysis"]
            if isinstance(ba, dict) and "mean_by_class" in ba:
                means = ba["mean_by_class"]
                diffs = ba.get("abs_mean_difference_by_feature", {})
                
                lines.append(f"  {'Feature':<20} {'Class 0 Mean':>15} {'Class 1 Mean':>15} {'Abs Diff':>15}")
                lines.append(f"  {'─'*20} {'─'*15} {'─'*15} {'─'*15}")
                
                for col, val_dict in means.items():
                    c0 = f"{val_dict.get(0, 'N/A'):.4f}" if isinstance(val_dict.get(0), (int, float)) else "N/A"
                    c1 = f"{val_dict.get(1, 'N/A'):.4f}" if isinstance(val_dict.get(1), (int, float)) else "N/A"
                    d = f"{diffs.get(col, 'N/A'):.4f}" if isinstance(diffs.get(col), (int, float)) else "N/A"
                    lines.append(f"  {col:<20} {c0:>15} {c1:>15} {d:>15}")
            else:
                lines.append(f"  {ba}")

        vis_keys = [k for k in results.keys() if k.startswith("visualizations")]
        if vis_keys:
            lines.append(_section("VISUALIZATIONS"))
            for vk in vis_keys:
                lines.append(f"  - {results[vk]}")

        lines.append("\n" + "=" * 80)
        lines.append("  END OF REPORT")
        lines.append("=" * 80 + "\n")

        Path(filepath).write_text("\n".join(lines), encoding="utf-8")


class JsonReportStrategy(ReportStrategy):
    class _SafeEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    def generate(self, results: dict, filepath: str | Path) -> None:
        Path(filepath).write_text(
            json.dumps(results, indent=4, cls=self._SafeEncoder),
            encoding="utf-8",
        )