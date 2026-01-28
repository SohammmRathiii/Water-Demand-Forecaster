import { useState } from 'react';
import { 
  Sprout, 
  Droplets, 
  ArrowRight, 
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Info
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export default function FarmerImpact() {
  const [showExplanation, setShowExplanation] = useState(false);

  const downstreamAvailability = {
    status: 'Moderate stress',
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    indicatorColor: 'bg-yellow-500',
    irrigationSupplyAvailable: 65, // %
    changeInAllocation: -15, // %
  };

  const cropStress = {
    riskLevel: 'Medium',
    season: 'Rabi',
    stressDuration: '2 weeks',
    reason: 'Reduced releases due to urban demand spike',
  };

  const urbanEfficiency = {
    recyclingIncrease: 12, // %
    daysPreserved: 18,
    forecastingBenefit: 'Prevented 2 emergency releases',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Farmer Impact & Agricultural Security</h1>
          <p className="text-sm text-muted-foreground">
            Monitoring downstream impact of urban water decisions
          </p>
        </div>
        <Button 
          variant="outline" 
          className="gap-2"
          onClick={() => setShowExplanation(!showExplanation)}
        >
          <Info className="h-4 w-4" />
          Explain Farmer Impact
        </Button>
      </div>

      {showExplanation && (
        <Alert>
          <Info className="h-4 w-4 text-blue-500" />
          <AlertTitle className="text-blue-500">AI Agent Explanation</AlertTitle>
          <AlertDescription className="text-muted-foreground mt-2 space-y-2">
            <p><strong className="text-foreground">Why farmers are at risk:</strong> Urban water demand takes priority during shortages, reducing the volume available for downstream irrigation reservoirs.</p>
            <p><strong className="text-foreground">How urban demand caused it:</strong> A 15% increase in city consumption this month has led to a reduction in scheduled releases for the Rabi season.</p>
            <p><strong className="text-foreground">Actions to reduce impact:</strong> Increasing urban water recycling by 10% can restore ~15 days of irrigation supply. Better demand forecasting helps avoid panic-holding of water in dams.</p>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg font-medium flex items-center gap-2">
              <Droplets className="h-5 w-5 text-blue-500" />
              Downstream Availability
            </CardTitle>
            <CardDescription>Reservoir outflow for agriculture</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-medium text-2xl">{downstreamAvailability.irrigationSupplyAvailable}%</span>
                <Badge variant="outline" className={`${downstreamAvailability.bgColor} ${downstreamAvailability.color} border-none`}>
                  {downstreamAvailability.status}
                </Badge>
              </div>
              <Progress value={downstreamAvailability.irrigationSupplyAvailable} className="h-2" indicatorColor={downstreamAvailability.indicatorColor} />
              <div className="text-sm text-muted-foreground flex items-center gap-1">
                <TrendingDown className="h-4 w-4 text-red-500" />
                <span className="text-red-500 font-medium">{downstreamAvailability.changeInAllocation}%</span>
                <span>allocation vs baseline</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg font-medium flex items-center gap-2">
              <Sprout className="h-5 w-5 text-green-500" />
              Crop Stress Risk
            </CardTitle>
            <CardDescription>Qualitative planning indicator</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Risk Level</span>
                <span className="font-bold text-yellow-500">{cropStress.riskLevel}</span>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between border-b pb-1">
                  <span className="text-muted-foreground">Season</span>
                  <span>{cropStress.season}</span>
                </div>
                <div className="flex justify-between border-b pb-1">
                  <span className="text-muted-foreground">Duration</span>
                  <span>{cropStress.stressDuration}</span>
                </div>
                <div className="pt-1">
                  <span className="text-muted-foreground block mb-1">Primary Driver:</span>
                  <span className="italic text-muted-foreground">{cropStress.reason}</span>
                </div>
              </div>
              <div className="bg-muted p-2 rounded text-xs text-muted-foreground flex gap-2 items-start">
                <Info className="h-3 w-3 mt-0.5 shrink-0" />
                This is a planning indicator, not a crop yield prediction.
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg font-medium flex items-center gap-2 text-green-500">
              <ArrowRight className="h-5 w-5" />
              Urban-Rural Link
            </CardTitle>
            <CardDescription>Benefits from city efficiency</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="p-3 bg-card border rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium text-green-500">Recycling Impact</span>
                </div>
                <p className="text-2xl font-bold text-green-500">+{urbanEfficiency.daysPreserved} Days</p>
                <p className="text-xs text-muted-foreground">of irrigation supply preserved due to {urbanEfficiency.recyclingIncrease}% urban recycling increase</p>
              </div>
              
              <div className="p-3 bg-card border rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="h-4 w-4 text-blue-500" />
                  <span className="text-sm font-medium text-blue-500">Forecasting Value</span>
                </div>
                <p className="text-sm text-foreground">{urbanEfficiency.forecastingBenefit}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Regional Agriculture Status Dashboard</CardTitle>
          <CardDescription>Overview of irrigation districts dependent on city reservoirs</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-3">
               <div className="border rounded-lg p-4">
                 <h3 className="font-medium mb-2 text-sm text-muted-foreground">Current Irrigation Risk</h3>
                 <div className="flex items-center gap-2">
                   <AlertTriangle className="h-5 w-5 text-yellow-500" />
                   <span className="font-bold text-lg">Elevated</span>
                 </div>
               </div>
               <div className="border rounded-lg p-4">
                 <h3 className="font-medium mb-2 text-sm text-muted-foreground">Risk Trend</h3>
                 <div className="flex items-center gap-2">
                   <TrendingUp className="h-5 w-5 text-red-500" />
                   <span className="font-bold text-lg">Increasing</span>
                 </div>
               </div>
               <div className="border rounded-lg p-4">
                 <h3 className="font-medium mb-2 text-sm text-muted-foreground">Key Factor</h3>
                 <div className="flex items-center gap-2">
                   <Droplets className="h-5 w-5 text-blue-500" />
                   <span className="font-medium">Urban Demand Spike</span>
                 </div>
               </div>
            </div>

            <div className="border rounded-lg p-4 bg-muted/30">
              <h3 className="font-medium mb-3">Recommended Mitigation Actions</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="bg-green-100 text-green-700 px-1.5 py-0.5 rounded text-xs font-medium mt-0.5">High Impact</span>
                  <span>Accelerate leak detection program in Zone 2 to reduce urban withdrawal.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-xs font-medium mt-0.5">Medium Impact</span>
                  <span>Shift industrial water supply to recycled water in Eastern Suburbs.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded text-xs font-medium mt-0.5">Planning</span>
                  <span>Review release schedule for next month based on revised urban forecast.</span>
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
