import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts';
import { ForecastPoint } from '@/lib/waterData';

interface DemandSupplyChartProps {
  data: ForecastPoint[];
  showConfidence?: boolean;
}

export function DemandSupplyChart({ data, showConfidence = true }: DemandSupplyChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="demandGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.3} />
            <stop offset="95%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="supplyGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="hsl(142, 76%, 36%)" stopOpacity={0.3} />
            <stop offset="95%" stopColor="hsl(142, 76%, 36%)" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="hsl(187, 85%, 43%)" stopOpacity={0.2} />
            <stop offset="95%" stopColor="hsl(187, 85%, 43%)" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 20%)" />
        <XAxis 
          dataKey="date" 
          stroke="hsl(215, 20%, 55%)"
          tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 12 }}
          tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        />
        <YAxis 
          stroke="hsl(215, 20%, 55%)"
          tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 12 }}
          tickFormatter={(value) => `${value}`}
          label={{ 
            value: 'MLD', 
            angle: -90, 
            position: 'insideLeft',
            fill: 'hsl(215, 20%, 55%)',
            fontSize: 12
          }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'hsl(222, 47%, 10%)', 
            border: '1px solid hsl(222, 30%, 20%)',
            borderRadius: '8px',
            color: 'hsl(210, 40%, 98%)'
          }}
          labelFormatter={(value) => new Date(value).toLocaleDateString('en-US', { 
            weekday: 'short', 
            month: 'short', 
            day: 'numeric' 
          })}
          formatter={(value: number, name: string) => [
            `${value.toFixed(1)} MLD`,
            name.charAt(0).toUpperCase() + name.slice(1)
          ]}
        />
        <Legend 
          wrapperStyle={{ paddingTop: '20px' }}
          formatter={(value) => <span className="text-muted-foreground text-sm capitalize">{value}</span>}
        />
        
        {showConfidence && (
          <Area
            type="monotone"
            dataKey="confidenceHigh"
            stackId="confidence"
            stroke="none"
            fill="url(#confidenceGradient)"
            name="confidence"
          />
        )}
        
        <Area
          type="monotone"
          dataKey="supply"
          stroke="hsl(142, 76%, 36%)"
          strokeWidth={2}
          fill="url(#supplyGradient)"
          name="supply"
        />
        <Area
          type="monotone"
          dataKey="demand"
          stroke="hsl(0, 84%, 60%)"
          strokeWidth={2}
          fill="url(#demandGradient)"
          name="demand"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
