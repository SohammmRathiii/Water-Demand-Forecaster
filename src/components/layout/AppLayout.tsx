import { Outlet } from 'react-router-dom';
import { AppSidebar } from './AppSidebar';

export function AppLayout() {
  return (
    <div className="min-h-screen wave-bg flex w-full">
      <AppSidebar />
      <main className="flex-1 ml-16 lg:ml-64 min-h-screen p-4 lg:p-6 transition-all duration-300">
        <Outlet />
      </main>
    </div>
  );
}
