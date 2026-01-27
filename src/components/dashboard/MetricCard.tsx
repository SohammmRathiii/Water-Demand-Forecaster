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
    success: 'text-alert-green',
    warning: 'text-alert-yellow',
    danger: 'text-alert-red',
    info: 'text-primary',
  };

  const statusBgColors = {
    success: 'bg-alert-green/10',
    warning: 'bg-alert-yellow/10',
    danger: 'bg-alert-red/10',
    info: 'bg-primary/10',
  };

  return (
    <div className={cn("metric-card", className)}>
      <div className="flex items-start justify-between">
        <div>
          <p className="data-label mb-2">{title}</p>
          <div className="flex items-baseline gap-1">
            <span className="data-value">{value}</span>
            {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
          </div>
          {trend && (
            <div className={cn(
              "mt-2 flex items-center gap-1 text-xs",
              trend.value >= 0 ? 'text-alert-green' : 'text-alert-red'
            )}>
              <span>{trend.value >= 0 ? '↑' : '↓'} {Math.abs(trend.value)}%</span>
              <span className="text-muted-foreground">{trend.label}</span>
            </div>
          )}
        </div>
        <div className={cn(
          "flex h-10 w-10 items-center justify-center rounded-lg",
          statusBgColors[status]
        )}>
          <Icon className={cn("h-5 w-5", statusColors[status])} />
        </div>
      </div>
    </div>
  );
}
