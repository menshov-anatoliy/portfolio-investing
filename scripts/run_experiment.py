#!/usr/bin/env python
"""CLI entry point for running portfolio experiments."""

import logging
import sys

import click
import yaml

from portfolio_investing.backtest.engine import BacktestEngine
from portfolio_investing.data.loader import DataLoader
from portfolio_investing.reporting.metrics import generate_report, print_report, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", default="config/default_config.yaml", help="Config file path")
@click.option("--output", default="output", help="Output directory")
@click.option("--start-date", default=None, help="Override start date YYYY-MM-DD")
@click.option("--end-date", default=None, help="Override end date YYYY-MM-DD")
@click.option("--n-clusters", default=None, type=int, help="Override number of clusters")
@click.option("--dry-run", is_flag=True, help="Validate config without running backtest")
def main(config, output, start_date, end_date, n_clusters, dry_run):
    """Run portfolio investing experiment across all strategy combinations."""
    logger.info("Loading config from %s", config)
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if start_date:
        cfg.setdefault("backtest", {})["start_date"] = start_date
    if end_date:
        cfg.setdefault("backtest", {})["end_date"] = end_date
    if n_clusters is not None:
        cfg.setdefault("clustering", {})["n_clusters_min"] = n_clusters
        cfg["clustering"]["n_clusters_max"] = n_clusters
    cfg.setdefault("reporting", {})["output_dir"] = output

    if dry_run:
        logger.info("Dry run mode – config validated successfully.")
        click.echo("Config OK. Tickers configured:")
        for group, tickers in cfg.get("tickers", {}).items():
            click.echo(f"  {group}: {tickers}")
        return

    # Download data
    loader = DataLoader(cfg)
    prices = loader.download()
    logger.info("Downloaded prices: %s", prices.shape)

    # Run all strategy combinations
    alloc_methods = cfg.get("allocation", {}).get(
        "methods", ["equal_weight", "inverse_vol", "risk_parity"]
    )
    reb_modes = cfg.get("rebalance", {}).get(
        "modes", ["monthly", "quarterly", "threshold", "hybrid_monthly", "hybrid_quarterly"]
    )

    engine = BacktestEngine(cfg)
    results = {}
    for alloc in alloc_methods:
        for reb in reb_modes:
            logger.info("Running: allocation=%s rebalance=%s", alloc, reb)
            try:
                result = engine.run(prices, alloc, reb)
                results[(alloc, reb)] = result
                logger.info(
                    "  -> Final value: %.4f, Rebalances: %d",
                    result["portfolio_values"].iloc[-1],
                    len(result["rebalance_dates"]),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Strategy (%s, %s) failed: %s", alloc, reb, exc)

    if not results:
        logger.error("No strategies completed successfully.")
        sys.exit(1)

    report = generate_report(results, cfg)
    print_report(report)
    save_report(report, output)
    logger.info("Done. Results saved to %s/", output)


if __name__ == "__main__":
    main()
