import { useState, useMemo } from 'react';
import { Droplets, AlertCircle, ArrowDown, ArrowUp, Waves } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { ZoneStatusTable } from '@/components/dashboard/ZoneStatusTable';
import { recommendDistribution, defaultZones } from '@/lib/waterData';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

export default function Distribution() {
  const [forecastedDemand, setForecastedDemand] = useState(170); 
  const [reservoirCapacity] = useState(500);
  const [currentStorage, setCurrentStorage] = useState(320);
  const [maxDailySupply] = useState(180);
  const [safetyBuffer, setSafetyBuffer] = useState(10);

  const recommendation = useMemo(() => 
    recommendDistribution(forecastedDemand, reservoirCapacity, currentStorage, maxDailySupply, safetyBuffer, defaultZones),
    [forecastedDemand, currentStorage, safetyBuffer]
  );

  const storagePercent = (currentStorage / reservoirCapacity) * 100;
  
  const dailyInflow = 2.5; 
  const dailyOutflow = recommendation.recommendedRelease;
  const netFlow = dailyInflow - dailyOutflow;

  return (
    <div className="space-y-8">
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gradient">Mumbai Zone-wise Water Distribution</h1>
          <p className="text-sm text-muted-foreground mt-2">
            Priority-based allocation & Reservoir Management
          </p>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-2xl">
              <div className="flex items-center justify-center h-10 w-10 rounded-lg bg-gradient-to-br from-primary to-accent">
                <Waves className="h-5 w-5 text-primary-foreground" />
              </div>
              Reservoir Status Panel
            </CardTitle>
            <Badge variant={recommendation.storageHealth === 'critical' ? 'destructive' : recommendation.storageHealth === 'healthy' ? 'default' : 'secondary'} className="uppercase font-bold">
              {recommendation.storageHealth} Levels
            </Badge>
          </div>
          <CardDescription className="mt-2">Real-time storage & flow monitoring</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-4 mb-8">
            <div className="space-y-2 bg-primary/5 rounded-lg p-4 border border-primary/20">
              <span className="text-xs font-semibold text-muted-foreground uppercase">Current Storage</span>
              <div className="text-3xl font-bold text-accent">
                {currentStorage} <span className="text-sm font-normal text-muted-foreground">MLD</span>
              </div>
              <div className="text-xs text-muted-foreground">
                of {reservoirCapacity} MLD Capacity
              </div>
            </div>
            <div className="space-y-2 bg-primary/5 rounded-lg p-4 border border-primary/20">
              <span className="text-xs font-semibold text-muted-foreground uppercase">Daily Outflow</span>
              <div className="text-3xl font-bold text-foreground">
                {dailyOutflow} <span className="text-sm font-normal text-muted-foreground">MLD</span>
              </div>
              <div className="text-xs text-muted-foreground">
                Released for city
              </div>
            </div>
            <div className="space-y-2 bg-primary/5 rounded-lg p-4 border border-primary/20">
              <span className="text-xs font-semibold text-muted-foreground uppercase">Net Flow</span>
              <div className={cn("text-3xl font-bold flex items-center gap-1", netFlow >= 0 ? "text-alert-green" : "text-alert-red")}>
                {netFlow > 0 ? <ArrowUp className="h-5 w-5" /> : <ArrowDown className="h-5 w-5" />}
                {Math.abs(netFlow).toFixed(1)} <span className="text-sm font-normal text-muted-foreground">MLD</span>
              </div>
            </div>
            <div className="space-y-2 bg-primary/5 rounded-lg p-4 border border-primary/20">
              <span className="text-sm text-muted-foreground">Days Remaining</span>
              <div className="text-2xl font-bold text-foreground">
                {recommendation.daysToEmpty > 365 ? '> 1 Year' : Math.round(recommendation.daysToEmpty)}
              </div>
              <div className="text-xs text-muted-foreground">
                at current release rate
              </div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium text-foreground">Storage Capacity ({storagePercent.toFixed(1)}%)</span>
              <span className="text-muted-foreground">{reservoirCapacity - currentStorage} MLD Empty Space</span>
            </div>
            <Progress value={storagePercent} className="h-3" />
            <div className="flex justify-between text-xs text-muted-foreground pt-1">
              <span>Critical Level (20%)</span>
              <span>Safe Level (60%)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">

        <div className="space-y-4 lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Simulation Controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Forecasted Demand</span>
                  <span className="font-mono text-foreground">{forecastedDemand} MLD</span>
                </div>
                <Slider
                  value={[forecastedDemand]}
                  min={100}
                  max={250}
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
              
              <div className="pt-4 border-t">
                <div className="flex items-center gap-2 mb-2">
                  <AlertCircle className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Distribution Logic</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Allocations are calculated sequentially based on priority:
                  <br/>1. Critical (Hospitals/Fire)
                  <br/>2. Essential (Residential)
                  <br/>3. Commercial
                  <br/>4. Industrial
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
             <Card className="bg-muted/30">
               <CardContent className="pt-6">
                 <div className="text-sm text-muted-foreground mb-1">Total Demand</div>
                 <div className="text-2xl font-bold">{forecastedDemand} MLD</div>
               </CardContent>
             </Card>
             <Card className={cn(
               recommendation.status === 'normal' ? 'bg-green-50 border-green-200' : 
               recommendation.status === 'reduced' ? 'bg-yellow-50 border-yellow-200' : 
               'bg-red-50 border-red-200'
             )}>
               <CardContent className="pt-6">
                 <div className="text-sm text-muted-foreground mb-1">Allocation Status</div>
                 <div className={cn("text-lg font-bold uppercase", 
                   recommendation.status === 'normal' ? 'text-green-700' : 
                   recommendation.status === 'reduced' ? 'text-yellow-700' : 
                   'text-red-700'
                 )}>{recommendation.status}</div>
               </CardContent>
             </Card>
             <Card>
               <CardContent className="pt-6">
                 <div className="text-sm text-muted-foreground mb-1">Shortfall</div>
                 <div className="text-2xl font-bold text-red-600">
                   {Math.max(0, forecastedDemand - recommendation.recommendedRelease).toFixed(1)} MLD
                 </div>
               </CardContent>
             </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Zone-wise Allocation Table</CardTitle>
              <CardDescription>
                Detailed breakdown of supply vs demand by zone priority
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ZoneStatusTable 
                zones={defaultZones} 
                allocations={recommendation.zoneAllocations} 
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
