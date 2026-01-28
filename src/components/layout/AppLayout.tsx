import { Outlet } from 'react-router-dom';
import { AppSidebar } from './AppSidebar';

export function AppLayout() {
  return (
    <div className="min-h-screen wave-bg flex w-full">
      <AppSidebar />
      <main className="flex-1 ml-16 lg:ml-64 min-h-screen p-6 lg:p-8 transition-all duration-300">
        <div className="mx-auto w-full max-w-7xl">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
