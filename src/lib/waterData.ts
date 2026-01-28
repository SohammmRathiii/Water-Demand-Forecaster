export interface Zone {
  id: string;
  name: string;
  priority: 'critical' | 'essential' | 'standard' | 'commercial' | 'industrial';
  minDemand: number;
  maxDemand: number;
  population: number;
  description: string;
}

export interface ForecastPoint {
  date: string;
  demand: number;
  supply: number;
  confidenceLow: number;
  confidenceHigh: number;
}

export interface WaterStressResult {
  stressIndex: number;
  alertLevel: 'green' | 'yellow' | 'orange' | 'red';
  components: {
    supplyStress: number;
    storageStress: number;
    trendStress: number;
    bufferStress: number;
  };
  metrics: {
    daysToEmpty: number;
    storageRatio: number;
    supplyDemandRatio: number;
  };
}

export interface ScenarioResult {
  name: string;
  forecastDemand: number;
  demandChangePct: number;
  stressRatio: number;
  riskCategory: 'low' | 'medium' | 'high' | 'critical';
  netFreshwaterDemand: number;
  explanation: string;
}

export interface RecyclingCategory {
  id: string;
  name: string;
  emoji: string;
  recyclability: 'yes' | 'partial' | 'no';
  recyclabilityScore: number;
  recoverablePercentage: number;
  freshwaterReduction: number;
  costPerKld: number;
  source: string;
  reusePurposes: string[];
}

export interface RecyclingEntry {
  id: string;
  waterType: string;
  volumeLiters: number;
  sourceType: 'household' | 'apartment' | 'industry' | 'institution';
  method: string;
  ward: string;
  zone: string;
  timestamp: string;
  verified: boolean;
}

export interface DistributionRecommendation {
  recommendedRelease: number;
  status: 'normal' | 'reduced' | 'critical';
  reason: string;
  storageAfterRelease: number;
  storageHealth: 'healthy' | 'moderate' | 'low' | 'critical';
  daysToEmpty: number;
  zoneAllocations: { zoneId: string; allocation: number; priority: string }[];
}

export const defaultZones: Zone[] = [
  { id: 'HOSPITAL_ZONE', name: 'Hospital Zone', priority: 'critical', minDemand: 5.0, maxDemand: 8.0, population: 50000, description: 'Hospitals, Fire Dept, Emergency' },
  { id: 'RESIDENTIAL_A', name: 'Residential North', priority: 'essential', minDemand: 30.0, maxDemand: 50.0, population: 200000, description: 'North side - drinking/sanitation' },
  { id: 'RESIDENTIAL_B', name: 'Residential South', priority: 'standard', minDemand: 25.0, maxDemand: 45.0, population: 180000, description: 'South side - full comfort' },
  { id: 'COMMERCIAL', name: 'Commercial', priority: 'commercial', minDemand: 15.0, maxDemand: 30.0, population: 100000, description: 'Offices, shops (non-essential)' },
  { id: 'INDUSTRIAL', name: 'Industrial', priority: 'industrial', minDemand: 20.0, maxDemand: 45.0, population: 50000, description: 'Factories, processing' },
];

export const recyclingCategories: RecyclingCategory[] = [
  {
    id: 'greywater',
    name: 'Greywater',
    emoji: '🚿',
    recyclability: 'yes',
    recyclabilityScore: 85,
    recoverablePercentage: 80,
    freshwaterReduction: 30,
    costPerKld: 15,
    source: 'Bathroom sinks, showers, laundry',
    reusePurposes: ['Garden irrigation', 'Toilet flushing', 'Car washing', 'Floor cleaning'],
  },
  {
    id: 'blackwater',
    name: 'Blackwater',
    emoji: '🚽',
    recyclability: 'partial',
    recyclabilityScore: 40,
    recoverablePercentage: 50,
    freshwaterReduction: 10,
    costPerKld: 45,
    source: 'Toilet waste, kitchen drains',
    reusePurposes: ['Agricultural irrigation (after treatment)', 'Industrial cooling'],
  },
  {
    id: 'industrial_wastewater',
    name: 'Industrial Wastewater',
    emoji: '🏭',
    recyclability: 'partial',
    recyclabilityScore: 55,
    recoverablePercentage: 60,
    freshwaterReduction: 20,
    costPerKld: 35,
    source: 'Manufacturing processes, cooling systems',
    reusePurposes: ['Process water recycling', 'Cooling tower makeup', 'Landscaping'],
  },
  {
    id: 'rainwater',
    name: 'Rainwater Harvesting',
    emoji: '🌧️',
    recyclability: 'yes',
    recyclabilityScore: 95,
    recoverablePercentage: 90,
    freshwaterReduction: 40,
    costPerKld: 8,
    source: 'Rooftop collection, surface runoff',
    reusePurposes: ['Drinking (with treatment)', 'Irrigation', 'Toilet flushing', 'Laundry'],
  },
  {
    id: 'treated_sewage',
    name: 'Treated Sewage',
    emoji: '♻️',
    recyclability: 'yes',
    recyclabilityScore: 70,
    recoverablePercentage: 75,
    freshwaterReduction: 35,
    costPerKld: 25,
    source: 'Municipal sewage treatment plants',
    reusePurposes: ['Industrial processes', 'Agriculture', 'Groundwater recharge', 'Urban landscaping'],
  },
];

export function generateForecastData(days: number, baseDate: Date = new Date()): ForecastPoint[] {
  const data: ForecastPoint[] = [];
  const baseDemand = 150;
  
  for (let i = 0; i < days; i++) {
    const date = new Date(baseDate);
    date.setDate(date.getDate() + i);
    
    const month = date.getMonth();
    const seasonalFactor = 1 + 0.15 * Math.sin((month - 3) * Math.PI / 6);
    
    const dow = date.getDay();
    const weekendFactor = (dow === 0 || dow === 6) ? 0.92 : 1.0;
    
    const noise = (Math.random() - 0.5) * 20;
    
    const demand = baseDemand * seasonalFactor * weekendFactor + noise;
    const supply = 165 + (Math.random() - 0.5) * 10;
    
    const uncertainty = 5 + i * 0.5;
    
    data.push({
      date: date.toISOString().split('T')[0],
      demand: Math.round(demand * 10) / 10,
      supply: Math.round(supply * 10) / 10,
      confidenceLow: Math.round((demand - uncertainty) * 10) / 10,
      confidenceHigh: Math.round((demand + uncertainty) * 10) / 10,
    });
  }
  
  return data;
}

export function computeWaterStressIndex(
  forecastedDemand: number,
  currentStorage: number,
  reservoirCapacity: number,
  maxDailySupply: number,
  actualInflow: number = 2.0
): WaterStressResult {
  
  const supplyDemandRatio = maxDailySupply / forecastedDemand;
  const supplyStress = Math.max(0, Math.min(100, (1 - supplyDemandRatio) * 100 + 50));
  
  const storageRatio = currentStorage / reservoirCapacity;
  const storageStress = Math.max(0, Math.min(100, (1 - storageRatio) * 100));
  
  const netDaily = actualInflow - forecastedDemand;
  const trendStress = netDaily < 0 ? Math.min(100, Math.abs(netDaily) / forecastedDemand * 100) : 0;
  
  const bufferDays = currentStorage / forecastedDemand;
  const bufferStress = Math.max(0, Math.min(100, (7 - bufferDays) / 7 * 100));
  
  const stressIndex = Math.round(
    supplyStress * 0.3 +
    storageStress * 0.3 +
    trendStress * 0.2 +
    bufferStress * 0.2
  );
  
  let alertLevel: 'green' | 'yellow' | 'orange' | 'red';
  if (stressIndex <= 25) alertLevel = 'green';
  else if (stressIndex <= 50) alertLevel = 'yellow';
  else if (stressIndex <= 75) alertLevel = 'orange';
  else alertLevel = 'red';
  
  const daysToEmpty = netDaily >= 0 ? 999 : currentStorage / Math.abs(netDaily);
  
  return {
    stressIndex,
    alertLevel,
    components: {
      supplyStress: Math.round(supplyStress),
      storageStress: Math.round(storageStress),
      trendStress: Math.round(trendStress),
      bufferStress: Math.round(bufferStress),
    },
    metrics: {
      daysToEmpty: Math.round(daysToEmpty * 10) / 10,
      storageRatio: Math.round(storageRatio * 100),
      supplyDemandRatio: Math.round(supplyDemandRatio * 100) / 100,
    },
  };
}

export function simulateScenario(
  scenarioType: string,
  baselineDemand: number,
  availableSupply: number,
  params: Record<string, number>
): ScenarioResult {
  let adjustedDemand = baselineDemand;
  let explanation = '';
  
  switch (scenarioType) {
    case 'heatwave':
      const tempIncrease = params.maxTemp - 35;
      const heatwaveFactor = 1 + (tempIncrease * 0.02 * params.numDays / 7);
      adjustedDemand = baselineDemand * heatwaveFactor;
      explanation = `${params.numDays}-day heatwave at ${params.maxTemp}°C increases demand by ${Math.round((heatwaveFactor - 1) * 100)}%`;
      break;
      
    case 'rainfall':
      const rainfallFactor = 1 + (params.rainfallChange / 100) * -0.3;
      adjustedDemand = baselineDemand * rainfallFactor;
      explanation = `${params.rainfallChange}% rainfall change ${params.rainfallChange < 0 ? 'increases' : 'decreases'} demand`;
      break;
      
    case 'population':
      const popFactor = 1 + (params.growthPct / 100);
      adjustedDemand = baselineDemand * popFactor;
      explanation = `${params.growthPct}% population surge over ${params.durationDays} days`;
      break;
      
    case 'festival':
      const festivalDemand = params.numFestivals * (params.avgAttendees / 10000) * 2;
      adjustedDemand = baselineDemand + festivalDemand;
      explanation = `${params.numFestivals} festivals with ${params.avgAttendees.toLocaleString()} attendees add ${festivalDemand.toFixed(1)} MLD`;
      break;
      
    case 'industrial':
      const industrialFactor = 1 + (params.changePct / 100) * 0.25;
      adjustedDemand = baselineDemand * industrialFactor;
      explanation = `${params.changePct}% industrial activity change`;
      break;
      
    case 'recycling':
      const recyclingOffset = baselineDemand * (params.adoptionPct / 100) * 0.3;
      adjustedDemand = baselineDemand - recyclingOffset;
      explanation = `${params.adoptionPct}% recycling adoption reduces demand by ${recyclingOffset.toFixed(1)} MLD`;
      break;
  }
  
  const demandChangePct = ((adjustedDemand - baselineDemand) / baselineDemand) * 100;
  const stressRatio = adjustedDemand / availableSupply;
  
  let riskCategory: 'low' | 'medium' | 'high' | 'critical';
  if (stressRatio < 0.8) riskCategory = 'low';
  else if (stressRatio < 1.0) riskCategory = 'medium';
  else if (stressRatio < 1.2) riskCategory = 'high';
  else riskCategory = 'critical';
  
  return {
    name: scenarioType,
    forecastDemand: Math.round(adjustedDemand * 10) / 10,
    demandChangePct: Math.round(demandChangePct * 10) / 10,
    stressRatio: Math.round(stressRatio * 100) / 100,
    riskCategory,
    netFreshwaterDemand: Math.round(adjustedDemand * 10) / 10,
    explanation,
  };
}

export function recommendDistribution(
  forecastedDemand: number,
  reservoirCapacity: number,
  currentStorage: number,
  maxDailySupply: number,
  safetyBufferPct: number,
  zones: Zone[]
): DistributionRecommendation {
  const safetyBuffer = reservoirCapacity * (safetyBufferPct / 100);
  const availableForRelease = currentStorage - safetyBuffer;
  
  // Logic: 
  // 1. Calculate Supply vs Demand
  // 2. Determine Stress Status
  // 3. Allocate based on Priority (Essential > Residential > Commercial > Industrial)

  let recommendedRelease = Math.min(forecastedDemand, maxDailySupply, Math.max(0, availableForRelease));
  
  let status: 'normal' | 'reduced' | 'critical';
  let reason: string;
  
  const supplyRatio = recommendedRelease / forecastedDemand;

  if (supplyRatio >= 0.98) {
    status = 'normal';
    reason = 'Sufficient supply to meet full demand';
  } else if (supplyRatio >= 0.8) {
    status = 'reduced';
    reason = 'Supply constrained - priority allocation in effect';
  } else {
    status = 'critical';
    reason = 'Critical shortage - emergency rationing required';
  }
  
  const storageAfterRelease = currentStorage - recommendedRelease;
  const storageRatio = storageAfterRelease / reservoirCapacity;
  
  let storageHealth: 'healthy' | 'moderate' | 'low' | 'critical';
  if (storageRatio > 0.6) storageHealth = 'healthy';
  else if (storageRatio > 0.4) storageHealth = 'moderate';
  else if (storageRatio > 0.2) storageHealth = 'low';
  else storageHealth = 'critical';
  
  const daysToEmpty = recommendedRelease > 0 ? currentStorage / recommendedRelease : 999;

  // Sequential Allocation Logic
  // Order: critical > essential > standard > commercial > industrial
  const priorityOrder = ['critical', 'essential', 'standard', 'commercial', 'industrial'];
  
  const zoneAllocations = zones.map(zone => {
      let allocationRatio: number;
      // In a real system, we would fill buckets sequentially. 
      // Here we simulate it by penalizing lower priorities more as status worsens.
      
      switch (zone.priority) {
        case 'critical': 
            allocationRatio = 1.0; 
            break;
        case 'essential': 
            allocationRatio = status === 'normal' ? 1.0 : 0.95; 
            break;
        case 'standard': 
            allocationRatio = status === 'normal' ? 1.0 : status === 'reduced' ? 0.85 : 0.6; 
            break;
        case 'commercial': 
            allocationRatio = status === 'normal' ? 1.0 : status === 'reduced' ? 0.75 : 0.4; 
            break;
        case 'industrial': 
            allocationRatio = status === 'normal' ? 1.0 : status === 'reduced' ? 0.5 : 0.2; 
            break;
        default: 
            allocationRatio = 0.5;
      }
      return {
        zoneId: zone.id,
        allocation: Math.round(zone.maxDemand * allocationRatio * 10) / 10,
        priority: zone.priority,
      };
    });
  
  return {
    recommendedRelease: Math.round(recommendedRelease * 10) / 10,
    status,
    reason,
    storageAfterRelease: Math.round(storageAfterRelease * 10) / 10,
    storageHealth,
    daysToEmpty: Math.round(daysToEmpty * 10) / 10,
    zoneAllocations,
  };
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  impact: 'positive' | 'negative';
  explanation: string;
}

export function getFeatureImportance(): FeatureImportance[] {
  return [
    { feature: 'Temperature', importance: 28, impact: 'positive', explanation: 'Higher temperatures increase water consumption for cooling and hydration' },
    { feature: 'Day of Week', importance: 18, impact: 'positive', explanation: 'Weekdays show higher commercial and industrial demand' },
    { feature: 'Rainfall', importance: 15, impact: 'negative', explanation: 'Rainfall reduces irrigation needs and outdoor water usage' },
    { feature: 'Festival Activity', importance: 12, impact: 'positive', explanation: 'Festivals bring additional population and consumption' },
    { feature: 'Industrial Index', importance: 11, impact: 'positive', explanation: 'Industrial activity directly correlates with process water needs' },
    { feature: 'Population Density', importance: 9, impact: 'positive', explanation: 'Higher population requires proportionally more water' },
    { feature: 'Season', importance: 7, impact: 'positive', explanation: 'Summer months show sustained higher demand patterns' },
  ];
}

export function generateAIResponse(query: string, context: {
  stressIndex: number;
  alertLevel: string;
  forecastDemand: number;
  currentStorage: number;
  recyclingRate: number;
}): string {
  const queryLower = query.toLowerCase();
  
  if (queryLower.includes('why') && queryLower.includes('demand')) {
    return `Based on our analysis, the current demand of ${context.forecastDemand} MLD is primarily driven by:
    
1. **Temperature Effect**: Current high temperatures are increasing cooling-related consumption by approximately 15%
2. **Seasonal Pattern**: We're in peak consumption season with sustained higher usage
3. **Industrial Activity**: Manufacturing sector operating at 92% capacity

The model confidence is 87% based on historical pattern matching. I recommend monitoring evening peak hours (6-9 PM) closely.`;
  }
  
  if (queryLower.includes('what if') || queryLower.includes('scenario')) {
    return `I can simulate various scenarios for you. Based on current conditions:

**If rainfall is delayed by 2 weeks:**
- Projected demand increase: +12%
- Stress index would rise from ${context.stressIndex} to ~${Math.min(100, context.stressIndex + 15)}
- Risk category: ${context.stressIndex + 15 > 75 ? 'HIGH' : 'MEDIUM'}

**If recycling adoption increases to 40%:**
- Net demand reduction: -18 MLD
- Stress index would drop to ~${Math.max(0, context.stressIndex - 12)}
- This would extend storage by 4.2 additional days

Would you like me to run a detailed simulation?`;
  }
  
  if (queryLower.includes('recommendation') || queryLower.includes('action')) {
    if (context.alertLevel === 'red' || context.alertLevel === 'orange') {
      return `⚠️ **Priority Actions Required**

Given the current stress index of ${context.stressIndex}/100:

1. **Immediate**: Activate Phase 2 water rationing for non-essential zones
2. **Short-term**: Increase treated water allocation to industrial users by 20%
3. **Communication**: Issue public advisory for voluntary conservation
4. **Monitoring**: Deploy additional sensors in high-consumption zones

Estimated impact: These measures could reduce stress index by 15-20 points within 72 hours.`;
    }
    
    return `Current operations are within normal parameters. Recommended proactive measures:

1. Continue monitoring reservoir inflow rates
2. Maintain recycling program outreach
3. Review industrial usage patterns for optimization opportunities

No immediate action required, but stay vigilant for seasonal demand increases.`;
  }
  
  return `I'm the Urban Water AI Decision Agent. I can help you with:

• **Demand Analysis**: "Why is demand increasing in Zone A?"
• **Scenario Planning**: "What if rainfall is delayed by 2 weeks?"
• **Risk Assessment**: "What's our shortage risk this week?"
• **Action Recommendations**: "What actions should we take?"

Current Status: Stress Index ${context.stressIndex}/100 (${context.alertLevel.toUpperCase()})
Storage: ${context.currentStorage} MLD | Recycling Rate: ${context.recyclingRate}%`;
}
