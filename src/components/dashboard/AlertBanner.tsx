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
      bg: 'bg-alert-green/15 border-alert-green/40',
      text: 'text-alert-green',
    },
    yellow: {
      icon: AlertCircle,
      label: 'WATCH',
      bg: 'bg-alert-yellow/15 border-alert-yellow/40',
      text: 'text-alert-yellow',
    },
    orange: {
      icon: AlertTriangle,
      label: 'PREPARE',
      bg: 'bg-alert-orange/15 border-alert-orange/40',
      text: 'text-alert-orange',
    },
    red: {
      icon: XCircle,
      label: 'CRITICAL',
      bg: 'bg-alert-red/15 border-alert-red/40',
      text: 'text-alert-red',
    },
  };

  const { icon: Icon, label, bg, text } = config[level];

  return (
    <div className={cn("rounded-xl border p-5 backdrop-blur-sm transition-all duration-300 hover:shadow-lg", bg)}>
      <div className="flex items-start gap-4">
        <div className={cn("flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-lg border", bg)}>
          <Icon className={cn("h-6 w-6", text)} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className={cn("text-xs font-bold uppercase tracking-wider", text)}>
              {label}
            </span>
          </div>
          <h3 className="mt-1 font-semibold text-foreground text-lg">{headline}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{message}</p>
          {actions && actions.length > 0 && (
            <div className="mt-4">
              <p className="text-xs font-semibold text-accent mb-3">RECOMMENDED ACTIONS:</p>
              <ul className="space-y-2">
                {actions.map((action, index) => (
                  <li key={index} className="flex items-start gap-3 text-sm text-foreground">
                    <span className={cn("text-lg leading-none mt-0.5", text)}>✓</span>
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
