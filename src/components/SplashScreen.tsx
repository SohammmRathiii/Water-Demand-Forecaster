
import { useEffect, useState } from 'react';
import { Droplets } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SplashScreenProps {
  onFinish: () => void;
}

export function SplashScreen({ onFinish }: SplashScreenProps) {
  const [isFading, setIsFading] = useState(false);

  useEffect(() => {
  
    const fadeTimer = setTimeout(() => {
      setIsFading(true);
    }, 2000);

    const finishTimer = setTimeout(() => {
      onFinish();
    }, 2800);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(finishTimer);
    };
  }, [onFinish]);

  return (
    <div 
      className={cn(
        "fixed inset-0 z-50 flex flex-col items-center justify-center bg-background transition-opacity duration-700 ease-in-out",
        isFading ? "opacity-0" : "opacity-100"
      )}
    >
      <div className="flex flex-col items-center animate-in fade-in zoom-in duration-1000">
        <div className="relative mb-6">
          <div className="absolute inset-0 animate-ping rounded-full bg-blue-500/20 duration-1000" />
          <div className="relative flex h-24 w-24 items-center justify-center rounded-full bg-blue-500/10 shadow-xl ring-1 ring-blue-500/20 backdrop-blur-sm">
            <Droplets className="h-12 w-12 text-blue-600 animate-bounce" style={{ animationDuration: '2s' }} />
          </div>
        </div>
        
        <div className="space-y-2 text-center">
          <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent">
            Aqua-Intel
          </h1>
          <p className="text-lg text-muted-foreground animate-in slide-in-from-bottom-2 duration-1000 delay-300">
            Urban Water Demand Forecasting
          </p>
        </div>
      </div>
    </div>
  );
}
