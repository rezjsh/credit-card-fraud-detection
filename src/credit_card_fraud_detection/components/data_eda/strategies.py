from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ReportStrategy(ABC):
    @abstractmethod
    def generate(self, results: dict, filepath: str | Path) -> None:
        """Serialise *results* and write to *filepath*."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str, width: int = 60) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


# ---------------------------------------------------------------------------
# Text strategy
# ---------------------------------------------------------------------------

class TextReportStrategy(ReportStrategy):
    """
    Produces a readable, section-per-key plain-text report.

    Covers every key that the standard pipeline may produce so nothing is
    silently dropped into the output file.
    """

    def generate(self, results: dict, filepath: str | Path) -> None:
        lines: list[str] = [
            "=" * 60,
            "          AUTOMATED EDA REPORT          ",
            "=" * 60,
        ]

        # 1. Pipeline metadata
        if meta := results.get("_pipeline_metadata"):
            lines.append(_section("PIPELINE METADATA"))
            lines.append(f"  Duration   : {meta.get('pipeline_duration_seconds', 'N/A')}s")
            lines.append(f"  Components : {meta.get('total_components', 'N/A')}")
            failed = meta.get("failed_components", [])
            lines.append(f"  Failed     : {', '.join(failed) if failed else 'none'}")

        # 2. Dataset overview
        if "num_rows" in results:
            lines.append(_section("DATASET OVERVIEW"))
            lines.append(f"  Rows          : {results['num_rows']:,}")
            lines.append(f"  Columns       : {results['num_cols']}")
            lines.append(f"  Memory        : {results['memory_usage_mb']:.2f} MB")
            lines.append(f"  Duplicate rows: {results.get('duplicate_rows', 'N/A')} "
                         f"({results.get('duplicate_percentage', 'N/A')}%)")
            if dtypes := results.get("dtypes"):
                lines.append("\n  Column types:")
                for col, dtype in dtypes.items():
                    lines.append(f"    {col:<30} {dtype}")

        # 3. Missing data
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

        # 4. Outliers
        if "outliers" in results:
            lines.append(_section("OUTLIERS  (IQR method)"))
            outliers = results["outliers"]
            if not outliers:
                lines.append("  ✓ No outliers detected at the current threshold.")
            else:
                lines.append(f"  {'Column':<30} {'Count':>8}  {'%':>7}  {'Lower':>10}  {'Upper':>10}")
                lines.append(f"  {'─'*30} {'─'*8}  {'─'*7}  {'─'*10}  {'─'*10}")
                for col, m in outliers.items():
                    lines.append(
                        f"  {col:<30} {m['outlier_count']:>8,}  {_fmt_pct(m['percentage']):>7}"
                        f"  {m['lower_bound']:>10.4f}  {m['upper_bound']:>10.4f}"
                    )

        # 5. Target distribution
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

        # 6. Cardinality
        if "cardinality" in results:
            lines.append(_section("CATEGORICAL CARDINALITY"))
            card = results["cardinality"]
            if card:
                lines.append(f"  {'Column':<30} {'Unique values':>14}")
                lines.append(f"  {'─'*30} {'─'*14}")
                for col, n in sorted(card.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  {col:<30} {n:>14,}")
            high = results.get("high_cardinality_columns", [])
            # if high:
            #     lines.append(f"\n  ⚠  High-cardinality columns (>{CardinalityAnalyzer.HIGH_CARDINALITY_THRESHOLD}): "
            #                  f"{', '.join(high)}")
            const = results.get("constant_columns", [])
            if const:
                lines.append(f"  ⚠  Constant columns (useless for modelling): {', '.join(const)}")

        # 7. Univariate analysis
        if "univariate_analysis" in results:
            lines.append(_section("UNIVARIATE ANALYSIS"))
            ua = results["univariate_analysis"]

            if num := ua.get("numerical"):
                lines.append("\n  Numerical features:")
                hdr = f"  {'Column':<28} {'Mean':>10}  {'Std':>10}  {'Skew':>7}  {'Kurt':>7}"
                lines.append(hdr)
                lines.append(f"  {'─'*28} {'─'*10}  {'─'*10}  {'─'*7}  {'─'*7}")
                for col, s in num.items():
                    lines.append(
                        f"  {col:<28} {s['mean']:>10.4f}  {s['std']:>10.4f}"
                        f"  {s['skewness']:>7.2f}  {s['kurtosis']:>7.2f}"
                    )

            if cat := ua.get("categorical"):
                lines.append("\n  Categorical features (mode):")
                lines.append(f"  {'Column':<28} {'Mode':<25} {'Freq':>8}  {'%':>7}")
                lines.append(f"  {'─'*28} {'─'*25} {'─'*8}  {'─'*7}")
                for col, s in cat.items():
                    lines.append(
                        f"  {col:<28} {str(s['mode']):<25} {s['mode_frequency']:>8,}"
                        f"  {_fmt_pct(s['mode_percentage']):>7}"
                    )

        # 8. Mutual information
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

        # 9. VIF / multicollinearity
        if "vif_scores" in results:
            lines.append(_section("MULTICOLLINEARITY  (VIF scores)"))
            vif = results["vif_scores"]
            if isinstance(vif, dict):
                lines.append(f"  {'Feature':<35} {'VIF':>8}  Status")
                lines.append(f"  {'─'*35} {'─'*8}  {'─'*10}")
                for col, info in vif.items():
                    if isinstance(info, dict):
                        flag = "⚠  HIGH" if info["flag"] == "HIGH" else "✓  OK"
                        vif_val = f"{info['vif']:.2f}" if info["vif"] is not None else "N/A"
                        lines.append(f"  {col:<35} {vif_val:>8}  {flag}")
                    else:
                        lines.append(f"  {col:<35}  {info}")
            else:
                lines.append(f"  {vif}")

        # 10. Bivariate analysis (summary only — full data in JSON)
        if "bivariate_analysis" in results:
            lines.append(_section("BIVARIATE ANALYSIS  (excerpt — see JSON for full detail)"))
            ba = results["bivariate_analysis"]
            if isinstance(ba, dict):
                lines.append("  Numerical mean by target class:")
                for target_class, col_means in (ba.get("numerical_mean_by_target") or {}).items():
                    lines.append(f"    Target = {target_class}")
                    for col, mean in col_means.items():
                        lines.append(f"      {col:<30}: {mean}")
            else:
                lines.append(f"  {ba}")

        # 11. Visualisations
        if "visualizations" in results:
            lines.append(_section("VISUALIZATIONS"))
            lines.append(f"  {results['visualizations']}")

        lines.append("\n" + "=" * 60)
        lines.append("  END OF REPORT")
        lines.append("=" * 60 + "\n")

        Path(filepath).write_text("\n".join(lines), encoding="utf-8")


# Needed for the cardinality threshold reference inside the text strategy
# from credit_card_fraud_detection.components.data_eda.analyzers import CardinalityAnalyzer  # noqa: E402


# ---------------------------------------------------------------------------
# JSON strategy
# ---------------------------------------------------------------------------

class JsonReportStrategy(ReportStrategy):
    """
    Full-fidelity JSON export.

    Uses a custom encoder to handle numpy/pandas types that the stdlib encoder
    would reject.
    """

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