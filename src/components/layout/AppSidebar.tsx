import { 
  LayoutDashboard, 
  TrendingUp, 
  Gauge, 
  Droplets, 
  AlertTriangle, 
  Recycle, 
  Bot,
  Sprout,
  Settings,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { NavLink } from '@/components/NavLink';
import { cn } from '@/lib/utils';
import { useState } from 'react';

const navigationItems = [
  { title: 'Dashboard', icon: LayoutDashboard, path: '/' },
  { title: 'Forecasting', icon: TrendingUp, path: '/forecasting' },
  { title: 'Scenarios', icon: Gauge, path: '/scenarios' },
  { title: 'Distribution', icon: Droplets, path: '/distribution' },
  { title: 'Alerts', icon: AlertTriangle, path: '/alerts' },
  { title: 'Recycling', icon: Recycle, path: '/recycling' },
  { title: 'Farmer Impact', icon: Sprout, path: '/farmer-impact' },
  { title: 'AI Agent', icon: Bot, path: '/agent' },
];

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside 
      className={cn(
        "fixed left-0 top-0 z-40 h-screen border-r border-primary/20 bg-sidebar transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
      style={{
        background: 'linear-gradient(180deg, hsl(198 50% 7%), hsl(196 45% 5%))'
      }}
    >
      {/* Header */}
      <div className={cn(
        "flex h-16 items-center border-b border-primary/15 px-4 backdrop-blur-sm",
        collapsed ? "justify-center" : "justify-between"
      )}>
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="flex items-center justify-center h-8 w-8 rounded-lg bg-gradient-to-br from-primary to-accent">
              <Droplets className="h-4 w-4 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-gradient">AquaIntel</h1>
              <p className="text-[10px] text-muted-foreground">Urban Water</p>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-3">
        {navigationItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-all duration-200 hover:text-foreground hover:bg-primary/10 border border-transparent",
              collapsed && "justify-center px-2"
            )}
            activeClassName="bg-primary/15 text-accent border-primary/30"
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!collapsed && <span>{item.title}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <div className="border-t border-primary/15 p-3 backdrop-blur-sm">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-all duration-200 hover:text-foreground hover:bg-primary/10 border border-transparent",
            collapsed && "justify-center px-2"
          )}
        >
          {collapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <>
              <ChevronLeft className="h-5 w-5" />
              <span>Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
