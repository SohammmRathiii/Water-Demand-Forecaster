import { cn } from '@/lib/utils';

interface StressGaugeProps {
  value: number;
  alertLevel: 'green' | 'yellow' | 'orange' | 'red';
  size?: 'sm' | 'md' | 'lg';
}

export function StressGauge({ value, alertLevel, size = 'md' }: StressGaugeProps) {
  const sizes = {
    sm: { width: 120, stroke: 8, fontSize: 'text-xl' },
    md: { width: 180, stroke: 12, fontSize: 'text-3xl' },
    lg: { width: 240, stroke: 16, fontSize: 'text-4xl' },
  };

  const { width, stroke, fontSize } = sizes[size];
  const radius = (width - stroke) / 2;
  const circumference = radius * Math.PI; // Half circle
  const progress = (value / 100) * circumference;

  const alertColors = {
    green: { primary: 'stroke-alert-green', bg: 'text-alert-green', label: 'SAFE' },
    yellow: { primary: 'stroke-alert-yellow', bg: 'text-alert-yellow', label: 'WATCH' },
    orange: { primary: 'stroke-alert-orange', bg: 'text-alert-orange', label: 'PREPARE' },
    red: { primary: 'stroke-alert-red', bg: 'text-alert-red', label: 'CRITICAL' },
  };

  const colors = alertColors[alertLevel];

  return (
    <div className="flex flex-col items-center">
      <svg width={width} height={width / 2 + 20} className="overflow-visible">
        {/* Background arc */}
        <path
          d={`M ${stroke / 2} ${width / 2} A ${radius} ${radius} 0 0 1 ${width - stroke / 2} ${width / 2}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-muted"
        />
        {/* Progress arc */}
        <path
          d={`M ${stroke / 2} ${width / 2} A ${radius} ${radius} 0 0 1 ${width - stroke / 2} ${width / 2}`}
          fill="none"
          strokeWidth={stroke}
          strokeLinecap="round"
          className={cn(colors.primary, "transition-all duration-1000")}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: circumference - progress,
          }}
        />
        {/* Center text */}
        <text
          x={width / 2}
          y={width / 2 - 10}
          textAnchor="middle"
          className={cn("font-mono font-bold fill-current", fontSize, colors.bg)}
        >
          {value}
        </text>
        <text
          x={width / 2}
          y={width / 2 + 15}
          textAnchor="middle"
          className="text-xs fill-muted-foreground uppercase tracking-widest"
        >
          / 100
        </text>
      </svg>
      <div className={cn(
        "mt-2 rounded-full px-4 py-1 text-xs font-bold uppercase tracking-wider",
        alertLevel === 'green' && "bg-alert-green/20 text-alert-green",
        alertLevel === 'yellow' && "bg-alert-yellow/20 text-alert-yellow",
        alertLevel === 'orange' && "bg-alert-orange/20 text-alert-orange",
        alertLevel === 'red' && "bg-alert-red/20 text-alert-red"
      )}>
        {colors.label}
      </div>
    </div>
  );
}
