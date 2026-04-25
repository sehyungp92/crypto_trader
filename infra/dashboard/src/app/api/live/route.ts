import { NextResponse } from "next/server";
import pool from "@/lib/db";
import type { LiveBatchResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const [equityRes, strategyRes, positionsRes, tradesRes, healthRes, summaryRes] =
      await Promise.all([
        pool.query(
          "SELECT equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
        ),
        pool.query("SELECT * FROM v_strategy_today"),
        pool.query(
          `SELECT *,
                  EXTRACT(EPOCH FROM (now() - last_update_at)) / 60 AS stale_minutes,
                  COALESCE(EXTRACT(EPOCH FROM (now() - entry_time)) / 60, 0) AS age_minutes
           FROM positions`
        ),
        pool.query("SELECT * FROM v_today_trades LIMIT 50"),
        pool.query(
          "SELECT timestamp, assessment, uptime_sec, alerts FROM health_snapshots ORDER BY timestamp DESC LIMIT 1"
        ),
        pool.query("SELECT * FROM v_portfolio_summary"),
      ]);

    const equity = equityRes.rows[0]?.equity ?? 0;
    const summary = summaryRes.rows[0] ?? {
      open_positions: 0,
      total_unrealized_pnl: 0,
      total_heat_r: 0,
    };

    // Sum daily PnL across strategies
    const dailyPnlUsd = strategyRes.rows.reduce(
      (sum: number, s: { daily_pnl_usd: number }) => sum + (s.daily_pnl_usd ?? 0),
      0
    );

    const health = healthRes.rows[0] ?? null;

    const response: LiveBatchResponse = {
      portfolio: {
        equity,
        daily_pnl_usd: dailyPnlUsd,
        unrealized_pnl: summary.total_unrealized_pnl,
        heat_r: summary.total_heat_r,
        open_positions: summary.open_positions,
      },
      strategies: strategyRes.rows,
      positions: positionsRes.rows,
      trades: tradesRes.rows,
      health,
    };

    return NextResponse.json(response);
  } catch (err) {
    console.error("API /live error:", err);
    return NextResponse.json({ error: "Database query failed" }, { status: 500 });
  }
}
