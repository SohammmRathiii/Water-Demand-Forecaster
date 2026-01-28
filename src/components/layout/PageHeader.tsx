import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: string;
  description?: string;
  icon?: LucideIcon;
  right?: ReactNode;
  className?: string;
}

export function PageHeader({ title, description, icon: Icon, right, className }: PageHeaderProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-primary/25 bg-gradient-to-br from-card/70 to-card/50 shadow-md backdrop-blur-sm transition-all duration-300 hover:shadow-lg hover:border-primary/40",
        className,
      )}
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-30"
        style={{
          background:
            "radial-gradient(900px 260px at 20% 0%, hsl(180 100% 50% / 0.15), transparent 60%), radial-gradient(700px 220px at 80% 20%, hsl(200 100% 55% / 0.12), transparent 55%)",
        }}
      />

      <div className="relative flex flex-col gap-4 p-8 sm:flex-row sm:items-end sm:justify-between">
        <div className="flex items-start gap-4">
          {Icon ? (
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent text-primary-foreground ring-1 ring-primary/20 shadow-lg">
              <Icon className="h-6 w-6" />
            </div>
          ) : null}
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-gradient sm:text-4xl">{title}</h1>
            {description ? (
              <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">{description}</p>
            ) : null}
          </div>
        </div>

        {right ? <div className="flex items-center gap-3">{right}</div> : null}
      </div>
    </div>
  );
}


