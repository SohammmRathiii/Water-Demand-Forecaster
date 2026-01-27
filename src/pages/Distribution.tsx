import { useState, useMemo } from 'react';
import { Droplets, AlertCircle } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { ZoneStatusTable } from '@/components/dashboard/ZoneStatusTable';
import { recommendDistribution, defaultZones } from '@/lib/waterData';
import { cn } from '@/lib/utils';

export default function Distribution() {
  const [forecastedDemand, setForecastedDemand] = useState(148);
  const [reservoirCapacity] = useState(500);
  const [currentStorage, setCurrentStorage] = useState(320);
  const [maxDailySupply] = useState(165);
  const [safetyBuffer, setSafetyBuffer] = useState(10);

  const recommendation = useMemo(() => 
    recommendDistribution(forecastedDemand, reservoirCapacity, currentStorage, maxDailySupply, safetyBuffer, defaultZones),
    [forecastedDemand, currentStorage, safetyBuffer]
  );

  const storagePercent = (currentStorage / reservoirCapacity) * 100;
  const releasePercent = (recommendation.recommendedRelease / maxDailySupply) * 100;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-alert-green';
      case 'reduced': return 'text-alert-yellow';
      case 'critical': return 'text-alert-red';
      default: return 'text-muted-foreground';
    }
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'bg-alert-green';
      case 'moderate': return 'bg-alert-yellow';
      case 'low': return 'bg-alert-orange';
      case 'critical': return 'bg-alert-red';
      default: return 'bg-muted';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Water Distribution Optimizer</h1>
          <p className="text-sm text-muted-foreground">
            Zone-based allocation with priority-aware distribution
          </p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Input Parameters */}
        <div className="space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Input Parameters</span>
            </div>
            <div className="panel-body space-y-6">
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Forecasted Demand</span>
                  <span className="font-mono text-foreground">{forecastedDemand} MLD</span>
                </div>
                <Slider
                  value={[forecastedDemand]}
                  min={100}
                  max={200}
                  step={1}
                  onValueChange={([value]) => setForecastedDemand(value)}
                />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Current Storage</span>
                  <span className="font-mono text-foreground">{currentStorage} MLD ({storagePercent.toFixed(0)}%)</span>
                </div>
                <Slider
                  value={[currentStorage]}
                  min={50}
                  max={reservoirCapacity}
                  step={5}
                  onValueChange={([value]) => setCurrentStorage(value)}
                />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Safety Buffer</span>
                  <span className="font-mono text-foreground">{safetyBuffer}%</span>
                </div>
                <Slider
                  value={[safetyBuffer]}
                  min={5}
                  max={25}
                  step={1}
                  onValueChange={([value]) => setSafetyBuffer(value)}
                />
              </div>

              <div className="pt-4 border-t border-border/50 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Reservoir Capacity</span>
                  <span className="font-mono text-foreground">{reservoirCapacity} MLD</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Max Daily Supply</span>
                  <span className="font-mono text-foreground">{maxDailySupply} MLD</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Distribution Recommendation</span>
              <span className={cn(
                "text-xs font-bold uppercase tracking-wider",
                getStatusColor(recommendation.status)
              )}>
                {recommendation.status} Operations
              </span>
            </div>
            <div className="panel-body">
              <div className="grid gap-6 md:grid-cols-3 mb-6">
                <div className="text-center p-4 rounded-lg bg-primary/10">
                  <Droplets className="h-8 w-8 mx-auto mb-2 text-primary" />
                  <p className="data-label">Release Amount</p>
                  <p className="text-3xl font-bold font-mono text-primary">
                    {recommendation.recommendedRelease}
                  </p>
                  <p className="text-xs text-muted-foreground">MLD</p>
                </div>
                
                <div className="text-center p-4 rounded-lg bg-muted/30">
                  <p className="data-label">Storage After Release</p>
                  <p className="text-3xl font-bold font-mono text-foreground">
                    {recommendation.storageAfterRelease}
                  </p>
                  <p className="text-xs text-muted-foreground">MLD remaining</p>
                  <div className="mt-2 h-2 rounded-full bg-muted overflow-hidden">
                    <div 
                      className={cn("h-full rounded-full transition-all", getHealthColor(recommendation.storageHealth))}
                      style={{ width: `${(recommendation.storageAfterRelease / reservoirCapacity) * 100}%` }}
                    />
                  </div>
                </div>

                <div className="text-center p-4 rounded-lg bg-muted/30">
                  <p className="data-label">Days to Empty</p>
                  <p className={cn(
                    "text-3xl font-bold font-mono",
                    recommendation.daysToEmpty > 14 ? 'text-alert-green' :
                    recommendation.daysToEmpty > 7 ? 'text-alert-yellow' :
                    'text-alert-red'
                  )}>
                    {recommendation.daysToEmpty > 30 ? '30+' : recommendation.daysToEmpty}
                  </p>
                  <p className="text-xs text-muted-foreground">at current rate</p>
                </div>
              </div>

              <div className="rounded-lg bg-muted/30 p-4 flex items-start gap-3">
                <AlertCircle className={cn("h-5 w-5 flex-shrink-0 mt-0.5", getStatusColor(recommendation.status))} />
                <div>
                  <p className="font-medium text-foreground">{recommendation.reason}</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Storage health: <span className="capitalize">{recommendation.storageHealth}</span> • 
                    Release capacity: {releasePercent.toFixed(0)}% of maximum
                  </p>
                </div>
              </div>
            </div>
          </div>


          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Zone Allocations</span>
            </div>
            <div className="panel-body">
              <ZoneStatusTable zones={defaultZones} allocations={recommendation.zoneAllocations} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
