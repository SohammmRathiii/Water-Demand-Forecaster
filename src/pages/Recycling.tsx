import { useState, useMemo } from 'react';
import { Recycle, Droplets, TrendingUp, Users, Building2, Factory, Home } from 'lucide-react';
import { recyclingCategories, RecyclingEntry } from '@/lib/waterData';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const sampleEntries: RecyclingEntry[] = [
  { id: '1', waterType: 'greywater', volumeLiters: 150, sourceType: 'household', method: 'cloth_filter', ward: 'Andheri West', zone: 'RESIDENTIAL_A', timestamp: new Date().toISOString(), verified: true },
  { id: '2', waterType: 'rainwater', volumeLiters: 500, sourceType: 'apartment', method: 'rooftop_collection', ward: 'Bandra', zone: 'RESIDENTIAL_A', timestamp: new Date().toISOString(), verified: true },
  { id: '3', waterType: 'treated_sewage', volumeLiters: 5000, sourceType: 'institution', method: 'stp', ward: 'Powai', zone: 'COMMERCIAL', timestamp: new Date().toISOString(), verified: true },
  { id: '4', waterType: 'industrial_wastewater', volumeLiters: 8000, sourceType: 'industry', method: 'membrane_bioreactor', ward: 'Andheri East', zone: 'INDUSTRIAL', timestamp: new Date().toISOString(), verified: true },
];

export default function Recycling() {
  const [entries, setEntries] = useState<RecyclingEntry[]>(sampleEntries);
  const [newEntry, setNewEntry] = useState({
    waterType: '',
    volumeLiters: '',
    sourceType: '',
    method: '',
    ward: '',
  });

  const stats = useMemo(() => {
    const totalVolume = entries.reduce((sum, e) => sum + e.volumeLiters, 0);
    const byType = recyclingCategories.map(cat => {
      const volume = entries.filter(e => e.waterType === cat.id).reduce((sum, e) => sum + e.volumeLiters, 0);
      const recoverable = volume * (cat.recoverablePercentage / 100);
      return {
        ...cat,
        volume,
        recoverable,
        percentage: totalVolume > 0 ? (volume / totalVolume) * 100 : 0,
      };
    }).filter(c => c.volume > 0);

    const totalRecoverable = byType.reduce((sum, c) => sum + c.recoverable, 0);
    const avgRecoveryRate = byType.length > 0 
      ? byType.reduce((sum, c) => sum + c.recoverablePercentage * c.volume, 0) / totalVolume 
      : 0;
    
    const cityDemand = 150 * 1000000; 
    const freshwaterOffset = (totalRecoverable / cityDemand) * 100;

    return {
      totalVolume,
      totalRecoverable,
      avgRecoveryRate,
      freshwaterOffset,
      byType,
    };
  }, [entries]);

  const pieData = stats.byType.map(c => ({
    name: c.name,
    value: c.volume,
    color: c.id === 'greywater' ? 'hsl(187, 85%, 43%)' :
           c.id === 'rainwater' ? 'hsl(199, 89%, 48%)' :
           c.id === 'treated_sewage' ? 'hsl(142, 76%, 36%)' :
           c.id === 'blackwater' ? 'hsl(25, 95%, 53%)' :
           'hsl(280, 65%, 60%)',
  }));

  const impactData = [
    { name: 'Total Input', value: stats.totalVolume / 1000, unit: 'KL' },
    { name: 'Recoverable', value: stats.totalRecoverable / 1000, unit: 'KL' },
    { name: 'Freshwater Saved', value: stats.totalRecoverable / 1000, unit: 'KL' },
  ];

  const handleSubmit = () => {
    if (!newEntry.waterType || !newEntry.volumeLiters || !newEntry.sourceType) return;
    
    const entry: RecyclingEntry = {
      id: Date.now().toString(),
      waterType: newEntry.waterType,
      volumeLiters: parseInt(newEntry.volumeLiters),
      sourceType: newEntry.sourceType as RecyclingEntry['sourceType'],
      method: newEntry.method || 'basic_filter',
      ward: newEntry.ward || 'Unknown',
      zone: 'RESIDENTIAL_A',
      timestamp: new Date().toISOString(),
      verified: false,
    };
    
    setEntries([entry, ...entries]);
    setNewEntry({ waterType: '', volumeLiters: '', sourceType: '', method: '', ward: '' });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Water Recycling Module</h1>
          <p className="text-sm text-muted-foreground">
            Classification, logging, and impact analytics
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="metric-card">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <Recycle className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="data-label">Total Recycled</p>
              <p className="data-value">{(stats.totalVolume / 1000).toFixed(1)}</p>
              <p className="text-xs text-muted-foreground">KL today</p>
            </div>
          </div>
        </div>
        
        <div className="metric-card">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-alert-green/10">
              <Droplets className="h-5 w-5 text-alert-green" />
            </div>
            <div>
              <p className="data-label">Recoverable</p>
              <p className="data-value text-alert-green">{(stats.totalRecoverable / 1000).toFixed(1)}</p>
              <p className="text-xs text-muted-foreground">KL usable</p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-info/10">
              <TrendingUp className="h-5 w-5 text-info" />
            </div>
            <div>
              <p className="data-label">Recovery Rate</p>
              <p className="data-value">{stats.avgRecoveryRate.toFixed(0)}%</p>
              <p className="text-xs text-muted-foreground">weighted avg</p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-warning/10">
              <Droplets className="h-5 w-5 text-warning" />
            </div>
            <div>
              <p className="data-label">Demand Offset</p>
              <p className="data-value">{stats.freshwaterOffset.toFixed(3)}%</p>
              <p className="text-xs text-muted-foreground">of city demand</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Water Categories</span>
          </div>
          <div className="panel-body space-y-3">
            {recyclingCategories.map((cat) => (
              <div 
                key={cat.id}
                className="flex items-start gap-3 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
              >
                <span className="text-2xl">{cat.emoji}</span>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-foreground">{cat.name}</span>
                    <span className={cn(
                      "text-xs px-2 py-0.5 rounded-full",
                      cat.recyclability === 'yes' ? 'bg-alert-green/20 text-alert-green' :
                      cat.recyclability === 'partial' ? 'bg-alert-yellow/20 text-alert-yellow' :
                      'bg-alert-red/20 text-alert-red'
                    )}>
                      {cat.recyclabilityScore}%
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{cat.source}</p>
                  <div className="flex gap-4 mt-2 text-xs">
                    <span className="text-muted-foreground">
                      Recovery: <span className="text-foreground">{cat.recoverablePercentage}%</span>
                    </span>
                    <span className="text-muted-foreground">
                      Cost: <span className="text-foreground">₹{cat.costPerKld}/KLD</span>
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Log Recycling Activity</span>
            </div>
            <div className="panel-body space-y-4">
              <div>
                <label className="text-sm text-muted-foreground">Water Type</label>
                <Select value={newEntry.waterType} onValueChange={(v) => setNewEntry({...newEntry, waterType: v})}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    {recyclingCategories.map(cat => (
                      <SelectItem key={cat.id} value={cat.id}>
                        {cat.emoji} {cat.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm text-muted-foreground">Volume (Liters)</label>
                <Input 
                  type="number"
                  placeholder="e.g., 100"
                  value={newEntry.volumeLiters}
                  onChange={(e) => setNewEntry({...newEntry, volumeLiters: e.target.value})}
                  className="mt-1"
                />
              </div>

              <div>
                <label className="text-sm text-muted-foreground">Source Type</label>
                <Select value={newEntry.sourceType} onValueChange={(v) => setNewEntry({...newEntry, sourceType: v})}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="Select source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="household"><Home className="inline h-4 w-4 mr-2" />Household</SelectItem>
                    <SelectItem value="apartment"><Building2 className="inline h-4 w-4 mr-2" />Apartment Complex</SelectItem>
                    <SelectItem value="industry"><Factory className="inline h-4 w-4 mr-2" />Industry</SelectItem>
                    <SelectItem value="institution"><Users className="inline h-4 w-4 mr-2" />Institution</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm text-muted-foreground">Ward (Optional)</label>
                <Input 
                  placeholder="e.g., Andheri West"
                  value={newEntry.ward}
                  onChange={(e) => setNewEntry({...newEntry, ward: e.target.value})}
                  className="mt-1"
                />
              </div>

              <Button onClick={handleSubmit} className="w-full">
                <Recycle className="mr-2 h-4 w-4" />
                Log Activity
              </Button>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Recent Entries</span>
            </div>
            <div className="panel-body space-y-2 max-h-64 overflow-y-auto">
              {entries.slice(0, 5).map((entry) => {
                const cat = recyclingCategories.find(c => c.id === entry.waterType);
                return (
                  <div key={entry.id} className="flex items-center justify-between p-2 rounded-lg bg-muted/30">
                    <div className="flex items-center gap-2">
                      <span>{cat?.emoji}</span>
                      <div>
                        <p className="text-sm font-medium">{cat?.name}</p>
                        <p className="text-xs text-muted-foreground">{entry.ward}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-sm">{entry.volumeLiters} L</p>
                      <p className="text-xs text-muted-foreground">{entry.sourceType}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Distribution by Type</span>
            </div>
            <div className="panel-body">
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={70}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(222, 47%, 10%)', 
                        border: '1px solid hsl(222, 30%, 20%)',
                        borderRadius: '8px'
                      }}
                      formatter={(value: number) => [`${(value / 1000).toFixed(1)} KL`, 'Volume']}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="grid grid-cols-2 gap-2 mt-4">
                {pieData.map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-xs text-muted-foreground">{item.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <span className="panel-title">Impact Summary</span>
            </div>
            <div className="panel-body">
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={impactData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 20%)" />
                    <XAxis type="number" tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 12 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 12 }} width={100} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(222, 47%, 10%)', 
                        border: '1px solid hsl(222, 30%, 20%)',
                        borderRadius: '8px'
                      }}
                      formatter={(value: number) => [`${value.toFixed(1)} KL`, 'Volume']}
                    />
                    <Bar dataKey="value" fill="hsl(187, 85%, 43%)" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
