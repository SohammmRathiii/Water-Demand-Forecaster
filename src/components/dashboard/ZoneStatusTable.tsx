import { cn } from '@/lib/utils';
import { Zone } from '@/lib/waterData';

interface ZoneAllocation {
  zoneId: string;
  allocation: number;
  priority: string;
}

interface ZoneStatusTableProps {
  zones: Zone[];
  allocations?: ZoneAllocation[];
}

export function ZoneStatusTable({ zones, allocations }: ZoneStatusTableProps) {
  const priorityColors: Record<string, string> = {
    critical: 'text-alert-red bg-alert-red/20 border-alert-red/40',
    essential: 'text-alert-orange bg-alert-orange/20 border-alert-orange/40',
    standard: 'text-accent bg-accent/20 border-accent/40',
    commercial: 'text-alert-yellow bg-alert-yellow/20 border-alert-yellow/40',
    industrial: 'text-muted-foreground bg-muted/20 border-muted/40',
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-primary/25 backdrop-blur-sm">
      <table className="w-full">
        <thead>
          <tr className="border-b border-primary/20 bg-primary/5">
            <th className="px-6 py-4 text-left text-xs font-semibold uppercase tracking-wider text-accent">Zone</th>
            <th className="px-6 py-4 text-left text-xs font-semibold uppercase tracking-wider text-accent">Priority</th>
            <th className="px-6 py-4 text-right text-xs font-semibold uppercase tracking-wider text-accent">Population</th>
            <th className="px-6 py-4 text-right text-xs font-semibold uppercase tracking-wider text-accent">Demand Range</th>
            {allocations && (
              <>
                <th className="px-6 py-4 text-right text-xs font-semibold uppercase tracking-wider text-accent">Allocation</th>
                <th className="px-6 py-4 text-right text-xs font-semibold uppercase tracking-wider text-accent">Status</th>
              </>
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-primary/10">
          {zones.map((zone) => {
            const allocation = allocations?.find(a => a.zoneId === zone.id);
            const allocationVal = allocation ? allocation.allocation : zone.maxDemand;
            const allocationPct = Math.round((allocationVal / zone.maxDemand) * 100);
            const shortfall = zone.maxDemand - allocationVal;
            
            return (
              <tr key={zone.id} className="hover:bg-primary/5 transition-colors duration-200">
                <td className="px-6 py-4">
                  <div>
                    <p className="font-semibold text-foreground">{zone.name}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{zone.description}</p>
                  </div>
                </td>
                <td className="px-6 py-4">
                  <span className={cn(
                    "inline-flex items-center rounded-lg px-3 py-1.5 text-xs font-bold capitalize border transition-all",
                    priorityColors[zone.priority]
                  )}>
                    {zone.priority}
                  </span>
                </td>
                <td className="px-6 py-4 text-right font-mono text-sm font-semibold text-foreground">
                  {zone.population.toLocaleString()}
                </td>
                <td className="px-6 py-4 text-right font-mono text-sm text-muted-foreground font-medium">
                  {zone.maxDemand} MLD
                </td>
                {allocations && (
                  <>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-3">
                        <div className="w-20 h-2.5 rounded-full bg-muted/50 overflow-hidden border border-primary/20">
                          <div 
                            className={cn(
                              "h-full rounded-full transition-all duration-500",
                              allocationPct >= 95 ? 'bg-gradient-to-r from-alert-green to-accent' :
                              allocationPct >= 75 ? 'bg-gradient-to-r from-alert-yellow to-alert-orange' :
                              allocationPct >= 50 ? 'bg-alert-orange' :
                              'bg-alert-red'
                            )}
                            style={{ width: `${allocationPct}%` }}
                          />
                        </div>
                        <span className="font-mono text-sm font-semibold text-foreground w-14 text-right">
                          {allocationVal} MLD
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                       {shortfall > 0.5 ? (
                         <span className="inline-flex items-center rounded-lg px-3 py-1.5 bg-alert-red/20 text-alert-red text-xs font-bold border border-alert-red/40">-{shortfall.toFixed(1)} MLD</span>
                       ) : (
                         <span className="inline-flex items-center rounded-lg px-3 py-1.5 bg-alert-green/20 text-alert-green text-xs font-bold border border-alert-green/40">Met</span>
                       )}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
