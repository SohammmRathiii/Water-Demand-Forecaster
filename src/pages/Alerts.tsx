import { useState, useMemo } from 'react';
import { AlertTriangle, TrendingUp, Droplets, Calendar } from 'lucide-react';
import { AlertBanner } from '@/components/dashboard/AlertBanner';
import { StressGauge } from '@/components/dashboard/StressGauge';
import { computeWaterStressIndex } from '@/lib/waterData';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';

export default function Alerts() {
  const [forecastedDemand, setForecastedDemand] = useState(155);
  const [currentStorage, setCurrentStorage] = useState(280);
  const [reservoirCapacity] = useState(500);
  const [maxDailySupply] = useState(165);
  const [inflow, setInflow] = useState(2);

  const stressResult = useMemo(() => 
    computeWaterStressIndex(forecastedDemand, currentStorage, reservoirCapacity, maxDailySupply, inflow),
    [forecastedDemand, currentStorage, inflow]
  );

  const shortageForecast = useMemo(() => {
    const days = [];
    let storage = currentStorage;
    
    for (let i = 1; i <= 7; i++) {
      const demand = forecastedDemand + (Math.random() - 0.5) * 10;
      const available = Math.min(storage + inflow, maxDailySupply);
      const shortage = Math.max(0, demand - available);
      
      storage = Math.max(0, storage + inflow - demand);
      
      days.push({
        day: i,
        demand: Math.round(demand * 10) / 10,
        available: Math.round(available * 10) / 10,
        shortage: Math.round(shortage * 10) / 10,
        storage: Math.round(storage * 10) / 10,
        status: shortage > 0 ? 'shortage' : storage < reservoirCapacity * 0.2 ? 'warning' : 'ok'
      });
    }
    
    return days;
  }, [forecastedDemand, currentStorage, inflow]);

  const daysWithShortage = shortageForecast.filter(d => d.status === 'shortage').length;
  const maxShortage = Math.max(...shortageForecast.map(d => d.shortage));

  const alertMessages: Record<string, { headline: string; message: string; actions: string[] }> = {
    green: {
      headline: 'All Systems Normal',
      message: 'Water supply is operating within safe parameters. No action required.',
      actions: ['Continue routine monitoring', 'Maintain recycling programs'],
    },
    yellow: {
      headline: 'Elevated Water Stress Detected',
      message: 'Demand approaching supply capacity. Monitor closely.',
      actions: ['Increase monitoring frequency', 'Prepare conservation notices', 'Review industrial permits'],
    },
    orange: {
      headline: 'Water Stress Warning Active',
      message: 'Supply constraints imminent. Prepare response protocols.',
      actions: ['Activate Phase 1 conservation', 'Issue public advisory', 'Coordinate tanker deployment'],
    },
    red: {
      headline: 'Critical Water Shortage',
      message: 'Emergency conditions. Immediate action required.',
      actions: ['Activate emergency rationing', 'Deploy all available tankers', 'Coordinate with state authorities'],
    },
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Risk Alerts & Management</h1>
          <p className="text-sm text-muted-foreground">
            Real-time stress monitoring and shortage prediction
          </p>
        </div>
      </div>

      {/* Current Alert */}
      <AlertBanner
        level={stressResult.alertLevel}
        headline={alertMessages[stressResult.alertLevel].headline}
        message={alertMessages[stressResult.alertLevel].message}
        actions={alertMessages[stressResult.alertLevel].actions}
      />

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Stress Gauge & Controls */}
        <div className="space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Water Stress Index</span>
            </div>
            <div className="panel-body flex flex-col items-center">
              <StressGauge 
                value={stressResult.stressIndex} 
                alertLevel={stressResult.alertLevel}
                size="lg"
              />
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Adjust Parameters</span>
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
                  <span className="font-mono text-foreground">{currentStorage} MLD</span>
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
                  <span className="text-muted-foreground">Daily Inflow</span>
                  <span className="font-mono text-foreground">{inflow} MLD</span>
                </div>
                <Slider
                  value={[inflow]}
                  min={0}
                  max={20}
                  step={0.5}
                  onValueChange={([value]) => setInflow(value)}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Stress Components */}
        <div className="space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Stress Components</span>
            </div>
            <div className="panel-body space-y-4">
              {[
                { label: 'Supply Stress', value: stressResult.components.supplyStress, icon: TrendingUp },
                { label: 'Storage Stress', value: stressResult.components.storageStress, icon: Droplets },
                { label: 'Trend Stress', value: stressResult.components.trendStress, icon: TrendingUp },
                { label: 'Buffer Stress', value: stressResult.components.bufferStress, icon: AlertTriangle },
              ].map((component) => (
                <div key={component.label} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <component.icon className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">{component.label}</span>
                    </div>
                    <span className={cn(
                      "font-mono text-sm",
                      component.value < 30 ? 'text-alert-green' :
                      component.value < 60 ? 'text-alert-yellow' :
                      component.value < 80 ? 'text-alert-orange' :
                      'text-alert-red'
                    )}>
                      {component.value}%
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div 
                      className={cn(
                        "h-full rounded-full transition-all duration-500",
                        component.value < 30 ? 'bg-alert-green' :
                        component.value < 60 ? 'bg-alert-yellow' :
                        component.value < 80 ? 'bg-alert-orange' :
                        'bg-alert-red'
                      )}
                      style={{ width: `${component.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Key Metrics</span>
            </div>
            <div className="panel-body grid grid-cols-2 gap-4">
              <div className="text-center p-3 rounded-lg bg-muted/30">
                <p className="text-xs text-muted-foreground uppercase">Days to Empty</p>
                <p className={cn(
                  "text-2xl font-bold font-mono",
                  stressResult.metrics.daysToEmpty > 14 ? 'text-alert-green' :
                  stressResult.metrics.daysToEmpty > 7 ? 'text-alert-yellow' :
                  'text-alert-red'
                )}>
                  {stressResult.metrics.daysToEmpty > 100 ? '∞' : stressResult.metrics.daysToEmpty.toFixed(1)}
                </p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/30">
                <p className="text-xs text-muted-foreground uppercase">Storage Ratio</p>
                <p className="text-2xl font-bold font-mono text-foreground">
                  {stressResult.metrics.storageRatio}%
                </p>
              </div>
              <div className="col-span-2 text-center p-3 rounded-lg bg-muted/30">
                <p className="text-xs text-muted-foreground uppercase">Supply/Demand Ratio</p>
                <p className={cn(
                  "text-2xl font-bold font-mono",
                  stressResult.metrics.supplyDemandRatio >= 1 ? 'text-alert-green' : 'text-alert-red'
                )}>
                  {stressResult.metrics.supplyDemandRatio}x
                </p>
              </div>
            </div>
          </div>
        </div>

       
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">7-Day Shortage Prediction</span>
          </div>
          <div className="panel-body">
            <div className="mb-4 grid grid-cols-2 gap-4">
              <div className="text-center p-3 rounded-lg bg-muted/30">
                <p className="text-xs text-muted-foreground">Days with Shortage</p>
                <p className={cn(
                  "text-2xl font-bold font-mono",
                  daysWithShortage === 0 ? 'text-alert-green' :
                  daysWithShortage <= 2 ? 'text-alert-yellow' :
                  'text-alert-red'
                )}>
                  {daysWithShortage}/7
                </p>
              </div>
              <div className="text-center p-3 rounded-lg bg-muted/30">
                <p className="text-xs text-muted-foreground">Max Shortage</p>
                <p className={cn(
                  "text-2xl font-bold font-mono",
                  maxShortage === 0 ? 'text-alert-green' : 'text-alert-red'
                )}>
                  {maxShortage} MLD
                </p>
              </div>
            </div>

            <div className="space-y-2">
              {shortageForecast.map((day) => (
                <div 
                  key={day.day}
                  className={cn(
                    "flex items-center justify-between p-3 rounded-lg",
                    day.status === 'shortage' ? 'bg-alert-red/10 border border-alert-red/30' :
                    day.status === 'warning' ? 'bg-alert-yellow/10 border border-alert-yellow/30' :
                    'bg-muted/30'
                  )}
                >
                  <div className="flex items-center gap-3">
                    <Calendar className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">Day {day.day}</span>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="text-muted-foreground">
                      D: <span className="font-mono text-foreground">{day.demand}</span>
                    </span>
                    <span className="text-muted-foreground">
                      S: <span className="font-mono text-foreground">{day.available}</span>
                    </span>
                    {day.shortage > 0 && (
                      <span className="text-alert-red font-medium">
                        -{day.shortage} MLD
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
