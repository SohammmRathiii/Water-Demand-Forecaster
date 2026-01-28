import { useState } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import Dashboard from "@/pages/Dashboard";
import Forecasting from "@/pages/Forecasting";
import Scenarios from "@/pages/Scenarios";
import Distribution from "@/pages/Distribution";
import Alerts from "@/pages/Alerts";
import Recycling from "@/pages/Recycling";
import Agent from "@/pages/Agent";
import FarmerImpact from "@/pages/FarmerImpact";
import NotFound from "./pages/NotFound";
import { SplashScreen } from "@/components/SplashScreen";

const queryClient = new QueryClient();

const App = () => {
  const [showSplash, setShowSplash] = useState(true);

  if (showSplash) {
    return <SplashScreen onFinish={() => setShowSplash(false)} />;
  }

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/forecasting" element={<Forecasting />} />
              <Route path="/scenarios" element={<Scenarios />} />
              <Route path="/distribution" element={<Distribution />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/recycling" element={<Recycling />} />
              <Route path="/farmer-impact" element={<FarmerImpact />} />
              <Route path="/agent" element={<Agent />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
