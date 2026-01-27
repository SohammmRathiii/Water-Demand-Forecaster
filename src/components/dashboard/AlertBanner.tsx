import { AlertTriangle, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AlertBannerProps {
  level: 'green' | 'yellow' | 'orange' | 'red';
  headline: string;
  message: string;
  actions?: string[];
}

export function AlertBanner({ level, headline, message, actions }: AlertBannerProps) {
  const config = {
    green: {
      icon: CheckCircle,
      label: 'SAFE',
      bg: 'bg-alert-green/10 border-alert-green/30',
      text: 'text-alert-green',
    },
    yellow: {
      icon: AlertCircle,
      label: 'WATCH',
      bg: 'bg-alert-yellow/10 border-alert-yellow/30',
      text: 'text-alert-yellow',
    },
    orange: {
      icon: AlertTriangle,
      label: 'PREPARE',
      bg: 'bg-alert-orange/10 border-alert-orange/30',
      text: 'text-alert-orange',
    },
    red: {
      icon: XCircle,
      label: 'CRITICAL',
      bg: 'bg-alert-red/10 border-alert-red/30',
      text: 'text-alert-red',
    },
  };

  const { icon: Icon, label, bg, text } = config[level];

  return (
    <div className={cn("rounded-xl border p-4", bg)}>
      <div className="flex items-start gap-4">
        <div className={cn("flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full", bg)}>
          <Icon className={cn("h-5 w-5", text)} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className={cn("text-xs font-bold uppercase tracking-wider", text)}>
              {label}
            </span>
          </div>
          <h3 className="mt-1 font-semibold text-foreground">{headline}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{message}</p>
          {actions && actions.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-muted-foreground mb-2">Recommended Actions:</p>
              <ul className="space-y-1">
                {actions.map((action, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm text-foreground">
                    <span className={text}>•</span>
                    {action}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
