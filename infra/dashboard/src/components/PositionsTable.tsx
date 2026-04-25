"use client";

import type { PositionRow } from "@/lib/types";
import { fmtUSD, fmtAge, colorClass } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function PositionsTable({ positions }: { positions: PositionRow[] }) {
  if (positions.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Open Positions</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-6">No open positions</p>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader><CardTitle>Open Positions</CardTitle></CardHeader>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-zinc-500 uppercase border-b border-surface-3">
              <th className="text-left py-2 pr-4">Symbol</th>
              <th className="text-left py-2 pr-4">Strategy</th>
              <th className="text-left py-2 pr-4">Dir</th>
              <th className="text-right py-2 pr-4">Qty</th>
              <th className="text-right py-2 pr-4">Entry</th>
              <th className="text-right py-2 pr-4">Unrealized</th>
              <th className="text-right py-2 pr-4">Risk R</th>
              <th className="text-right py-2 pr-4">Stop</th>
              <th className="text-right py-2">Age</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr
                key={`${p.strategy_id}-${p.symbol}`}
                className="border-b border-surface-3/50 hover:bg-surface-2/50"
              >
                <td className="py-2 pr-4 font-mono font-medium text-zinc-200">
                  {p.symbol}
                </td>
                <td className="py-2 pr-4 text-zinc-400">{p.strategy_id}</td>
                <td className="py-2 pr-4">
                  <Badge variant={p.direction === "long" ? "green" : "red"}>
                    {p.direction}
                  </Badge>
                </td>
                <td className="py-2 pr-4 text-right font-mono text-zinc-300">
                  {p.qty.toFixed(4)}
                </td>
                <td className="py-2 pr-4 text-right font-mono text-zinc-300">
                  ${p.avg_entry.toLocaleString()}
                </td>
                <td className={`py-2 pr-4 text-right font-mono ${colorClass(p.unrealized_pnl)}`}>
                  {fmtUSD(p.unrealized_pnl)}
                </td>
                <td className="py-2 pr-4 text-right font-mono text-zinc-300">
                  {p.risk_r.toFixed(2)}R
                </td>
                <td className="py-2 pr-4 text-right font-mono text-zinc-400">
                  {p.stop_price ? `$${p.stop_price.toLocaleString()}` : "—"}
                </td>
                <td className="py-2 text-right text-zinc-400">
                  {p.entry_time ? fmtAge(p.age_minutes) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
