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
            <stop offset="5%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.35} />
            <stop offset="95%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="supplyGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="hsl(150, 80%, 40%)" stopOpacity={0.35} />
            <stop offset="95%" stopColor="hsl(150, 80%, 40%)" stopOpacity={0.05} />
          </linearGradient>
          <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="hsl(180, 100%, 50%)" stopOpacity={0.2} />
            <stop offset="95%" stopColor="hsl(180, 100%, 50%)" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(200, 40%, 18%)" opacity={0.5} />
        <XAxis 
          dataKey="date" 
          stroke="hsl(210, 30%, 55%)"
          tick={{ fill: 'hsl(210, 30%, 55%)', fontSize: 12, fontWeight: 500 }}
          tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
        />
        <YAxis 
          stroke="hsl(210, 30%, 55%)"
          tick={{ fill: 'hsl(210, 30%, 55%)', fontSize: 12, fontWeight: 500 }}
          tickFormatter={(value) => `${value}`}
          label={{ 
            value: 'MLD', 
            angle: -90, 
            position: 'insideLeft',
            fill: 'hsl(210, 30%, 55%)',
            fontSize: 12,
            fontWeight: 600
          }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'hsl(198 50% 10%)', 
            border: '1px solid hsl(180 100% 50% / 0.3)',
            borderRadius: '10px',
            color: 'hsl(210, 40%, 98%)',
            boxShadow: '0 0 20px hsl(180 100% 50% / 0.2)'
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
          formatter={(value) => <span className="text-muted-foreground text-sm capitalize font-semibold">{value}</span>}
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
          stroke="hsl(150, 80%, 40%)"
          strokeWidth={3}
          fill="url(#supplyGradient)"
          name="supply"
          isAnimationActive={true}
        />
        <Area
          type="monotone"
          dataKey="demand"
          stroke="hsl(0, 84%, 60%)"
          strokeWidth={3}
          fill="url(#demandGradient)"
          name="demand"
          isAnimationActive={true}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
