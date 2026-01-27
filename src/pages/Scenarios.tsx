import { useState, useMemo } from 'react';
import { Play, RotateCcw, Zap, CloudRain, Users, PartyPopper, Factory, Recycle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { simulateScenario, ScenarioResult } from '@/lib/waterData';
import { cn } from '@/lib/utils';

interface ScenarioConfig {
  id: string;
  name: string;
  icon: typeof Zap;
  color: string;
  params: { key: string; label: string; min: number; max: number; default: number; unit: string }[];
}

const scenarioConfigs: ScenarioConfig[] = [
  {
    id: 'heatwave',
    name: 'Heatwave',
    icon: Zap,
    color: 'text-alert-red',
    params: [
      { key: 'numDays', label: 'Duration', min: 3, max: 30, default: 7, unit: 'days' },
      { key: 'maxTemp', label: 'Max Temperature', min: 38, max: 50, default: 42, unit: '°C' },
    ],
  },
  {
    id: 'rainfall',
    name: 'Rainfall Change',
    icon: CloudRain,
    color: 'text-info',
    params: [
      { key: 'rainfallChange', label: 'Rainfall Change', min: -80, max: 80, default: -40, unit: '%' },
    ],
  },
  {
    id: 'population',
    name: 'Population Surge',
    icon: Users,
    color: 'text-alert-orange',
    params: [
      { key: 'growthPct', label: 'Growth', min: 5, max: 50, default: 15, unit: '%' },
      { key: 'durationDays', label: 'Duration', min: 7, max: 90, default: 30, unit: 'days' },
    ],
  },
  {
    id: 'festival',
    name: 'Festival Overlap',
    icon: PartyPopper,
    color: 'text-alert-yellow',
    params: [
      { key: 'numFestivals', label: 'Festivals', min: 1, max: 5, default: 2, unit: '' },
      { key: 'avgAttendees', label: 'Avg Attendees', min: 10000, max: 500000, default: 100000, unit: '' },
    ],
  },
  {
    id: 'industrial',
    name: 'Industrial Change',
    icon: Factory,
    color: 'text-muted-foreground',
    params: [
      { key: 'changePct', label: 'Activity Change', min: -50, max: 50, default: 25, unit: '%' },
    ],
  },
  {
    id: 'recycling',
    name: 'Recycling Adoption',
    icon: Recycle,
    color: 'text-alert-green',
    params: [
      { key: 'adoptionPct', label: 'Adoption Rate', min: 10, max: 80, default: 30, unit: '%' },
    ],
  },
];

export default function Scenarios() {
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const [paramValues, setParamValues] = useState<Record<string, Record<string, number>>>({});
  const [results, setResults] = useState<ScenarioResult[]>([]);
  const [baselineDemand] = useState(150);
  const [availableSupply] = useState(165);

  const toggleScenario = (id: string) => {
    setSelectedScenarios(prev => 
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    );
    // Initialize default params
    if (!paramValues[id]) {
      const config = scenarioConfigs.find(s => s.id === id);
      if (config) {
        const defaults: Record<string, number> = {};
        config.params.forEach(p => { defaults[p.key] = p.default; });
        setParamValues(prev => ({ ...prev, [id]: defaults }));
      }
    }
  };

  const updateParam = (scenarioId: string, paramKey: string, value: number) => {
    setParamValues(prev => ({
      ...prev,
      [scenarioId]: { ...prev[scenarioId], [paramKey]: value }
    }));
  };

  const runSimulation = () => {
    const newResults: ScenarioResult[] = [];
    
    selectedScenarios.forEach(scenarioId => {
      const params = paramValues[scenarioId] || {};
      const result = simulateScenario(scenarioId, baselineDemand, availableSupply, params);
      newResults.push(result);
    });

    // Calculate combined scenario
    if (newResults.length > 1) {
      const combinedDemand = newResults.reduce((acc, r) => {
        const delta = r.forecastDemand - baselineDemand;
        return acc + delta;
      }, baselineDemand);
      
      const stressRatio = combinedDemand / availableSupply;
      let riskCategory: 'low' | 'medium' | 'high' | 'critical';
      if (stressRatio < 0.8) riskCategory = 'low';
      else if (stressRatio < 1.0) riskCategory = 'medium';
      else if (stressRatio < 1.2) riskCategory = 'high';
      else riskCategory = 'critical';

      newResults.push({
        name: 'combined',
        forecastDemand: Math.round(combinedDemand * 10) / 10,
        demandChangePct: Math.round(((combinedDemand - baselineDemand) / baselineDemand) * 1000) / 10,
        stressRatio: Math.round(stressRatio * 100) / 100,
        riskCategory,
        netFreshwaterDemand: Math.round(combinedDemand * 10) / 10,
        explanation: `Combined effect of ${newResults.length} scenarios`,
      });
    }

    setResults(newResults);
  };

  const resetAll = () => {
    setSelectedScenarios([]);
    setParamValues({});
    setResults([]);
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-alert-green bg-alert-green/10';
      case 'medium': return 'text-alert-yellow bg-alert-yellow/10';
      case 'high': return 'text-alert-orange bg-alert-orange/10';
      case 'critical': return 'text-alert-red bg-alert-red/10';
      default: return 'text-muted-foreground bg-muted';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Scenario Simulator</h1>
          <p className="text-sm text-muted-foreground">
            "What-If" analysis for water demand planning
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={resetAll} disabled={selectedScenarios.length === 0}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          <Button onClick={runSimulation} disabled={selectedScenarios.length === 0}>
            <Play className="mr-2 h-4 w-4" />
            Run Simulation
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Scenario Selection */}
        <div className="lg:col-span-2 space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Select Scenarios</span>
              <span className="text-xs text-muted-foreground">
                {selectedScenarios.length} selected
              </span>
            </div>
            <div className="panel-body grid gap-4 md:grid-cols-2">
              {scenarioConfigs.map((scenario) => {
                const isSelected = selectedScenarios.includes(scenario.id);
                const params = paramValues[scenario.id] || {};
                
                return (
                  <div
                    key={scenario.id}
                    className={cn(
                      "rounded-lg border p-4 transition-all cursor-pointer",
                      isSelected 
                        ? "border-primary bg-primary/5" 
                        : "border-border/50 hover:border-border"
                    )}
                    onClick={() => toggleScenario(scenario.id)}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div className={cn(
                        "flex h-10 w-10 items-center justify-center rounded-lg",
                        isSelected ? "bg-primary/20" : "bg-muted"
                      )}>
                        <scenario.icon className={cn("h-5 w-5", scenario.color)} />
                      </div>
                      <div>
                        <p className="font-medium text-foreground">{scenario.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {scenario.params.map(p => p.label).join(' • ')}
                        </p>
                      </div>
                    </div>
                    
                    {isSelected && (
                      <div className="space-y-4 mt-4 pt-4 border-t border-border/50" onClick={e => e.stopPropagation()}>
                        {scenario.params.map((param) => (
                          <div key={param.key} className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">{param.label}</span>
                              <span className="font-mono text-foreground">
                                {params[param.key] ?? param.default}{param.unit}
                              </span>
                            </div>
                            <Slider
                              value={[params[param.key] ?? param.default]}
                              min={param.min}
                              max={param.max}
                              step={param.key === 'avgAttendees' ? 10000 : 1}
                              onValueChange={([value]) => updateParam(scenario.id, param.key, value)}
                              className="cursor-pointer"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Baseline Info */}
        <div className="space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Baseline Conditions</span>
            </div>
            <div className="panel-body space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Base Demand</span>
                <span className="font-mono text-foreground">{baselineDemand} MLD</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Available Supply</span>
                <span className="font-mono text-foreground">{availableSupply} MLD</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Supply Buffer</span>
                <span className="font-mono text-alert-green">
                  +{Math.round((availableSupply / baselineDemand - 1) * 100)}%
                </span>
              </div>
            </div>
          </div>

          {/* Results */}
          {results.length > 0 && (
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Simulation Results</span>
              </div>
              <div className="panel-body space-y-4">
                {results.map((result, index) => (
                  <div 
                    key={result.name} 
                    className={cn(
                      "rounded-lg p-4 space-y-3",
                      result.name === 'combined' ? "bg-primary/10 border border-primary/30" : "bg-muted/30"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className={cn(
                        "font-medium capitalize",
                        result.name === 'combined' ? "text-primary" : "text-foreground"
                      )}>
                        {result.name === 'combined' ? '⚡ Combined Effect' : result.name}
                      </span>
                      <span className={cn(
                        "text-xs font-medium uppercase px-2 py-1 rounded-full",
                        getRiskColor(result.riskCategory)
                      )}>
                        {result.riskCategory}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground">Demand</p>
                        <p className="font-mono text-foreground">{result.forecastDemand} MLD</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Change</p>
                        <p className={cn(
                          "font-mono",
                          result.demandChangePct > 0 ? "text-alert-red" : "text-alert-green"
                        )}>
                          {result.demandChangePct > 0 ? '+' : ''}{result.demandChangePct}%
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Stress Ratio</p>
                        <p className="font-mono text-foreground">{result.stressRatio}x</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Shortage?</p>
                        <p className={cn(
                          "font-mono",
                          result.stressRatio > 1 ? "text-alert-red" : "text-alert-green"
                        )}>
                          {result.stressRatio > 1 ? 'Yes' : 'No'}
                        </p>
                      </div>
                    </div>
                    
                    <p className="text-xs text-muted-foreground">{result.explanation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
