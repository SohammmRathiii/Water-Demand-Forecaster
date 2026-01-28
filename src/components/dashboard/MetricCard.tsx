import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    label: string;
  };
  status?: 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}

export function MetricCard({ 
  title, 
  value, 
  unit, 
  icon: Icon, 
  trend, 
  status = 'info',
  className 
}: MetricCardProps) {
  const statusColors = {
    success: 'hsl(var(--alert-green))',
    warning: 'hsl(var(--alert-yellow))',
    danger: 'hsl(var(--alert-red))',
    info: 'hsl(var(--primary))',
  };

  const statusBgColors = {
    success: 'bg-alert-green/20 border-alert-green/40',
    warning: 'bg-alert-yellow/20 border-alert-yellow/40',
    danger: 'bg-alert-red/20 border-alert-red/40',
    info: 'bg-primary/15 border-primary/30',
  };

  return (
    <div className={cn("metric-card group", className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="data-label mb-2">{title}</p>
          <div className="flex items-baseline gap-1">
            <span className="data-value text-gradient">{value}</span>
            {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
          </div>
          {trend && (
            <div className={cn(
              "mt-2 flex items-center gap-1 text-xs font-semibold",
              trend.value >= 0 ? 'text-alert-green' : 'text-alert-red'
            )}>
              <span>{trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}%</span>
              <span className="text-muted-foreground">{trend.label}</span>
            </div>
          )}
        </div>
        <div className={cn(
          "flex h-12 w-12 items-center justify-center rounded-lg border transition-all duration-300",
          statusBgColors[status]
        )}>
          <Icon className="h-6 w-6 transition-all duration-300" style={{ color: statusColors[status] }} />
        </div>
      </div>
    </div>
  );
}
