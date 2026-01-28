import { useState, useMemo } from 'react';
import { 
  Droplets, 
  TrendingUp, 
  Gauge, 
  Thermometer,
  Users,
  Factory,
  Cloud,
  Recycle
} from 'lucide-react';
import { MetricCard } from '@/components/dashboard/MetricCard';
import { StressGauge } from '@/components/dashboard/StressGauge';
import { DemandSupplyChart } from '@/components/dashboard/DemandSupplyChart';
import { AlertBanner } from '@/components/dashboard/AlertBanner';
import { ZoneStatusTable } from '@/components/dashboard/ZoneStatusTable';
import { 
  generateForecastData, 
  computeWaterStressIndex, 
  defaultZones,
  recommendDistribution 
} from '@/lib/waterData';

export default function Dashboard() {
  const [currentStorage] = useState(320);
  const [reservoirCapacity] = useState(500);
  const [maxDailySupply] = useState(165);
  const [forecastedDemand] = useState(148);

  const forecastData = useMemo(() => generateForecastData(14), []);

  const stressResult = useMemo(() => 
    computeWaterStressIndex(forecastedDemand, currentStorage, reservoirCapacity, maxDailySupply),
    [forecastedDemand, currentStorage, reservoirCapacity, maxDailySupply]
  );

  const distribution = useMemo(() => 
    recommendDistribution(forecastedDemand, reservoirCapacity, currentStorage, maxDailySupply, 10, defaultZones),
    [forecastedDemand, reservoirCapacity, currentStorage, maxDailySupply]
  );

  const alertConfig = {
    green: {
      headline: 'Water Supply Operating Normally',
      message: 'All zones receiving full allocation. Storage levels healthy.',
      actions: ['Continue routine monitoring', 'Maintain recycling outreach programs'],
    },
    yellow: {
      headline: 'Elevated Demand Detected',
      message: 'Storage levels declining. Monitor closely over next 48 hours.',
      actions: ['Increase monitoring frequency', 'Prepare conservation advisories', 'Review industrial usage'],
    },
    orange: {
      headline: 'Water Stress Warning',
      message: 'Approaching supply constraints. Prepare rationing protocols.',
      actions: ['Activate Phase 1 conservation', 'Issue public advisory', 'Coordinate with industrial users'],
    },
    red: {
      headline: 'Critical Water Shortage',
      message: 'Immediate action required. Initiate emergency protocols.',
      actions: ['Activate emergency rationing', 'Deploy water tankers', 'Issue citywide alert'],
    },
  };

  const currentAlert = alertConfig[stressResult.alertLevel];

  return (
    <div className="space-y-8">
     
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold text-gradient">Aqua-Intel</h1>
          <p className="text-sm text-muted-foreground mt-2">
            Mumbai Metropolitan Region • Real-time monitoring & forecasting
          </p>
        </div>
        <div className="text-right bg-card/50 border border-primary/25 rounded-lg p-4 backdrop-blur-sm">
          <p className="text-xs text-muted-foreground font-semibold uppercase">Last updated</p>
          <p className="font-mono text-sm text-accent mt-1">
            {new Date().toLocaleString('en-IN', { 
              hour: '2-digit', 
              minute: '2-digit',
              day: 'numeric',
              month: 'short'
            })}
          </p>
        </div>
      </div>

    
      <AlertBanner
        level={stressResult.alertLevel}
        headline={currentAlert.headline}
        message={currentAlert.message}
        actions={currentAlert.actions}
      />

      <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Current Demand"
          value={forecastedDemand}
          unit="MLD"
          icon={TrendingUp}
          trend={{ value: 3.2, label: 'vs yesterday' }}
          status="danger"
        />
        <MetricCard
          title="Available Supply"
          value={maxDailySupply}
          unit="MLD"
          icon={Droplets}
          status="success"
        />
        <MetricCard
          title="Reservoir Storage"
          value={currentStorage}
          unit="MLD"
          icon={Gauge}
          trend={{ value: -2.1, label: 'daily change' }}
          status={stressResult.metrics.storageRatio > 50 ? 'info' : 'warning'}
        />
        <MetricCard
          title="Days to Critical"
          value={stressResult.metrics.daysToEmpty > 30 ? '30+' : stressResult.metrics.daysToEmpty.toFixed(1)}
          unit="days"
          icon={Thermometer}
          status={stressResult.metrics.daysToEmpty > 14 ? 'success' : stressResult.metrics.daysToEmpty > 7 ? 'warning' : 'danger'}
        />
      </div>

      
      <div className="grid gap-7 lg:grid-cols-3">
       
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Water Stress Index</span>
          </div>
          <div className="panel-body flex flex-col items-center justify-center">
            <StressGauge 
              value={stressResult.stressIndex} 
              alertLevel={stressResult.alertLevel}
              size="lg"
            />
            <div className="mt-6 grid w-full grid-cols-2 gap-4">
              <div className="text-center">
                <p className="data-label">Supply Stress</p>
                <p className="text-lg font-semibold">{stressResult.components.supplyStress}%</p>
              </div>
              <div className="text-center">
                <p className="data-label">Storage Stress</p>
                <p className="text-lg font-semibold">{stressResult.components.storageStress}%</p>
              </div>
              <div className="text-center">
                <p className="data-label">Trend Stress</p>
                <p className="text-lg font-semibold">{stressResult.components.trendStress}%</p>
              </div>
              <div className="text-center">
                <p className="data-label">Buffer Stress</p>
                <p className="text-lg font-semibold">{stressResult.components.bufferStress}%</p>
              </div>
            </div>
          </div>
        </div>

        
        <div className="panel lg:col-span-2">
          <div className="panel-header">
            <span className="panel-title">14-Day Demand Forecast</span>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-chart-demand" />
                <span className="text-xs text-muted-foreground">Demand</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-chart-supply" />
                <span className="text-xs text-muted-foreground">Supply</span>
              </div>
            </div>
          </div>
          <div className="panel-body">
            <div className="chart-container">
              <DemandSupplyChart data={forecastData} />
            </div>
          </div>
        </div>
      </div>

      
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Temperature"
          value="34"
          unit="°C"
          icon={Thermometer}
          trend={{ value: 2.5, label: 'above avg' }}
          status="warning"
        />
        <MetricCard
          title="Population Served"
          value="580K"
          icon={Users}
          status="info"
        />
        <MetricCard
          title="Industrial Index"
          value="92"
          unit="%"
          icon={Factory}
          status="info"
        />
        <MetricCard
          title="Recycled Water"
          value="18.5"
          unit="MLD"
          icon={Recycle}
          trend={{ value: 12, label: 'vs last month' }}
          status="success"
        />
      </div>

      
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">Zone Distribution Status</span>
          <span className={`text-xs font-medium uppercase tracking-wider ${
            distribution.status === 'normal' ? 'text-alert-green' :
            distribution.status === 'reduced' ? 'text-alert-yellow' :
            'text-alert-red'
          }`}>
            {distribution.status} Operations
          </span>
        </div>
        <div className="panel-body">
          <ZoneStatusTable zones={defaultZones} allocations={distribution.zoneAllocations} />
        </div>
      </div>
    </div>
  );
}
