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
    critical: 'text-alert-red bg-alert-red/10',
    essential: 'text-alert-orange bg-alert-orange/10',
    standard: 'text-primary bg-primary/10',
    commercial: 'text-alert-yellow bg-alert-yellow/10',
    industrial: 'text-muted-foreground bg-muted/50',
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border/50">
            <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Zone</th>
            <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">Priority</th>
            <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Population</th>
            <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Demand Range</th>
            {allocations && (
              <>
                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Allocation</th>
                <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-muted-foreground">Status</th>
              </>
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-border/30">
          {zones.map((zone) => {
            const allocation = allocations?.find(a => a.zoneId === zone.id);
            const allocationVal = allocation ? allocation.allocation : zone.maxDemand;
            const allocationPct = Math.round((allocationVal / zone.maxDemand) * 100);
            const shortfall = zone.maxDemand - allocationVal;
            
            return (
              <tr key={zone.id} className="hover:bg-muted/20 transition-colors">
                <td className="px-4 py-3">
                  <div>
                    <p className="font-medium text-foreground">{zone.name}</p>
                    <p className="text-xs text-muted-foreground">{zone.description}</p>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className={cn(
                    "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium capitalize",
                    priorityColors[zone.priority]
                  )}>
                    {zone.priority}
                  </span>
                </td>
                <td className="px-4 py-3 text-right font-mono text-sm text-foreground">
                  {zone.population.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right font-mono text-sm text-muted-foreground">
                  {zone.maxDemand} MLD
                </td>
                {allocations && (
                  <>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 h-2 rounded-full bg-muted overflow-hidden">
                          <div 
                            className={cn(
                              "h-full rounded-full transition-all",
                              allocationPct >= 95 ? 'bg-alert-green' :
                              allocationPct >= 75 ? 'bg-alert-yellow' :
                              allocationPct >= 50 ? 'bg-alert-orange' :
                              'bg-alert-red'
                            )}
                            style={{ width: `${allocationPct}%` }}
                          />
                        </div>
                        <span className="font-mono text-sm text-foreground">
                          {allocationVal} MLD
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                       {shortfall > 0.5 ? (
                         <span className="text-red-500 text-xs font-medium">-{shortfall.toFixed(1)} MLD</span>
                       ) : (
                         <span className="text-green-600 text-xs font-medium">Met</span>
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
