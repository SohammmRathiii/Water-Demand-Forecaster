import { 
  LayoutDashboard, 
  TrendingUp, 
  Gauge, 
  Droplets, 
  AlertTriangle, 
  Recycle, 
  Bot,
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
  { title: 'AI Agent', icon: Bot, path: '/agent' },
];

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside 
      className={cn(
        "fixed left-0 top-0 z-40 h-screen border-r border-border/50 bg-sidebar transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Header */}
      <div className={cn(
        "flex h-16 items-center border-b border-border/50 px-4",
        collapsed ? "justify-center" : "justify-between"
      )}>
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div>
              <h1 className="text-sm font-bold text-foreground">AquaIntel</h1>
              <p className="text-[10px] text-muted-foreground">Urban Water Platform</p>
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
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-foreground",
              collapsed && "justify-center px-2"
            )}
            activeClassName="bg-sidebar-accent text-foreground border-l-2 border-primary"
          >
            <item.icon className="h-5 w-5 flex-shrink-0" />
            {!collapsed && <span>{item.title}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse Toggle */}
      <div className="border-t border-border/50 p-3">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-all hover:bg-sidebar-accent hover:text-foreground",
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
