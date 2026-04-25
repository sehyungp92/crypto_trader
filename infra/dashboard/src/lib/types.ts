export interface PortfolioData {
  equity: number;
  daily_pnl_usd: number;
  unrealized_pnl: number;
  heat_r: number;
  open_positions: number;
}

export interface StrategyData {
  strategy_id: string;
  trades_today: number;
  wins_today: number;
  losses_today: number;
  daily_pnl_r: number;
  daily_pnl_usd: number;
}

export interface PositionRow {
  strategy_id: string;
  symbol: string;
  direction: string;
  qty: number;
  avg_entry: number;
  unrealized_pnl: number;
  risk_r: number;
  stop_price: number | null;
  entry_time: string | null;
  stale_minutes: number;
  age_minutes: number;
}

export interface TradeRow {
  trade_id: string;
  strategy_id: string;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  entry_time: string;
  exit_time: string;
  pnl: number;
  net_pnl: number;
  r_multiple: number | null;
  exit_reason: string | null;
  setup_grade: string | null;
  duration_minutes: number;
}

export interface HealthData {
  assessment: string;
  uptime_sec: number | null;
  alerts: string[];
  timestamp: string;
}

export interface EquityCurvePoint {
  ts: string;
  equity: number;
}

export interface DailyPnlPoint {
  trade_date: string;
  net_pnl: number;
  total_trades: number;
}

export interface LiveBatchResponse {
  portfolio: PortfolioData;
  strategies: StrategyData[];
  positions: PositionRow[];
  trades: TradeRow[];
  health: HealthData | null;
}

export interface ChartBatchResponse {
  equity_curve: EquityCurvePoint[];
  daily_pnl: DailyPnlPoint[];
}
