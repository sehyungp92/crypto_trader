"use client";

import type { HealthData } from "@/lib/types";
import { fmtAge, fmtDate } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const ASSESSMENT_VARIANT: Record<string, "green" | "amber" | "red" | "neutral"> = {
  healthy: "green",
  degraded: "amber",
  critical: "red",
};

export function SystemHealth({ data }: { data: HealthData | null }) {
  if (!data) {
    return (
      <Card>
        <CardHeader><CardTitle>System Health</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-4">No health data</p>
      </Card>
    );
  }

  const uptimeStr = data.uptime_sec ? fmtAge(data.uptime_sec / 60) : "—";
  const variant = ASSESSMENT_VARIANT[data.assessment] ?? "neutral";

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Health</CardTitle>
        <Badge variant={variant}>{data.assessment}</Badge>
      </CardHeader>

      <div className="flex gap-6 text-sm">
        <div>
          <p className="text-xs text-zinc-500">Uptime</p>
          <p className="font-mono text-zinc-200">{uptimeStr}</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">Last Report</p>
          <p className="font-mono text-zinc-200">{fmtDate(data.timestamp)}</p>
        </div>
      </div>

      {data.alerts.length > 0 && (
        <div className="mt-3 pt-3 border-t border-surface-3">
          <p className="text-xs text-zinc-500 mb-1">Alerts</p>
          <div className="flex flex-wrap gap-1">
            {data.alerts.map((alert, i) => (
              <Badge key={i} variant="amber">
                {typeof alert === "string" ? alert : JSON.stringify(alert)}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
