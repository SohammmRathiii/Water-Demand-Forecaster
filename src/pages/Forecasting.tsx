import { useState, useMemo } from 'react';
import { Calendar, TrendingUp, Info } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DemandSupplyChart } from '@/components/dashboard/DemandSupplyChart';
import { generateForecastData, getFeatureImportance } from '@/lib/waterData';
import { cn } from '@/lib/utils';

export default function Forecasting() {
  const [forecastHorizon, setForecastHorizon] = useState<'short' | 'medium'>('short');
  
  const shortTermData = useMemo(() => generateForecastData(7), []);
  const mediumTermData = useMemo(() => generateForecastData(180), []);
  
  const featureImportance = getFeatureImportance();
  
  const currentData = forecastHorizon === 'short' ? shortTermData : mediumTermData;
  
  // Calculate summary stats
  const avgDemand = currentData.reduce((sum, d) => sum + d.demand, 0) / currentData.length;
  const maxDemand = Math.max(...currentData.map(d => d.demand));
  const minDemand = Math.min(...currentData.map(d => d.demand));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Water Demand Forecasting</h1>
          <p className="text-sm text-muted-foreground">
            AI-powered predictions using Prophet + LSTM ensemble
          </p>
        </div>
      </div>

      {/* Forecast Toggle */}
      <Tabs value={forecastHorizon} onValueChange={(v) => setForecastHorizon(v as 'short' | 'medium')}>
        <TabsList className="bg-muted/50">
          <TabsTrigger value="short" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            <Calendar className="mr-2 h-4 w-4" />
            Short-Term (1-7 Days)
          </TabsTrigger>
          <TabsTrigger value="medium" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
            <TrendingUp className="mr-2 h-4 w-4" />
            Medium-Term (1-6 Months)
          </TabsTrigger>
        </TabsList>

        <TabsContent value="short" className="space-y-6 mt-6">
          <ForecastContent 
            data={shortTermData} 
            horizon="short"
            avgDemand={avgDemand}
            maxDemand={maxDemand}
            minDemand={minDemand}
            featureImportance={featureImportance}
          />
        </TabsContent>

        <TabsContent value="medium" className="space-y-6 mt-6">
          <ForecastContent 
            data={mediumTermData} 
            horizon="medium"
            avgDemand={avgDemand}
            maxDemand={maxDemand}
            minDemand={minDemand}
            featureImportance={featureImportance}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

interface ForecastContentProps {
  data: ReturnType<typeof generateForecastData>;
  horizon: 'short' | 'medium';
  avgDemand: number;
  maxDemand: number;
  minDemand: number;
  featureImportance: ReturnType<typeof getFeatureImportance>;
}

function ForecastContent({ data, horizon, avgDemand, maxDemand, minDemand, featureImportance }: ForecastContentProps) {
  return (
    <>
      {/* Summary Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="metric-card">
          <p className="data-label">Average Demand</p>
          <p className="data-value">{avgDemand.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">MLD over forecast period</p>
        </div>
        <div className="metric-card">
          <p className="data-label">Peak Demand</p>
          <p className="data-value text-alert-red">{maxDemand.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">Maximum expected</p>
        </div>
        <div className="metric-card">
          <p className="data-label">Minimum Demand</p>
          <p className="data-value text-alert-green">{minDemand.toFixed(1)}</p>
          <p className="text-xs text-muted-foreground">Lowest expected</p>
        </div>
        <div className="metric-card">
          <p className="data-label">Model Confidence</p>
          <p className="data-value text-primary">{horizon === 'short' ? '94' : '87'}%</p>
          <p className="text-xs text-muted-foreground">Ensemble accuracy</p>
        </div>
      </div>

      {/* Main Chart */}
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">
            {horizon === 'short' ? '7-Day' : '6-Month'} Demand Forecast
          </span>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Info className="h-4 w-4" />
            <span>Shaded area shows confidence interval</span>
          </div>
        </div>
        <div className="panel-body">
          <div className={cn("w-full", horizon === 'short' ? 'h-80' : 'h-96')}>
            <DemandSupplyChart data={data} showConfidence />
          </div>
        </div>
      </div>

      {/* Explainability Panel */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Feature Importance */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Feature Importance</span>
          </div>
          <div className="panel-body space-y-4">
            {featureImportance.map((feature, index) => (
              <div key={feature.feature} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">
                    {feature.feature}
                  </span>
                  <span className={cn(
                    "text-sm font-mono",
                    feature.impact === 'positive' ? 'text-alert-red' : 'text-alert-green'
                  )}>
                    {feature.impact === 'positive' ? '+' : '-'}{feature.importance}%
                  </span>
                </div>
                <div className="h-2 rounded-full bg-muted overflow-hidden">
                  <div 
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      feature.impact === 'positive' ? 'bg-chart-demand' : 'bg-chart-supply'
                    )}
                    style={{ width: `${feature.importance * 3}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">{feature.explanation}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Plain English Explanation */}
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Forecast Explanation</span>
          </div>
          <div className="panel-body">
            <div className="rounded-lg bg-muted/30 p-4 space-y-4">
              <div className="flex items-start gap-3">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary/20 text-primary">
                  💡
                </div>
                <div>
                  <p className="font-medium text-foreground">Key Insight</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Demand is projected to {avgDemand > 150 ? 'exceed' : 'remain below'} the average baseline 
                    of 150 MLD over the forecast period.
                  </p>
                </div>
              </div>
              
              <div className="border-t border-border/50 pt-4">
                <p className="text-sm font-medium text-foreground mb-2">Contributing Factors:</p>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <span className="text-alert-red">🌡️</span>
                    Higher than average temperatures (+3°C) increasing cooling demand
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-alert-yellow">📅</span>
                    Weekday patterns showing sustained commercial activity
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-alert-green">🌧️</span>
                    Forecasted rainfall expected to partially offset irrigation needs
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-primary">🏭</span>
                    Industrial index at 92% of capacity
                  </li>
                </ul>
              </div>

              <div className="border-t border-border/50 pt-4">
                <p className="text-sm text-foreground">
                  <strong>Model Summary:</strong> The ensemble model (Prophet + LSTM) predicts 
                  {horizon === 'short' 
                    ? ' short-term demand with 94% confidence, primarily driven by temperature and day-of-week patterns.'
                    : ' medium-term trends with 87% confidence, accounting for seasonal variations and population growth.'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
