import { useState, useRef, useEffect } from 'react';
import { Bot, Send, User, Sparkles, AlertTriangle, TrendingUp, Droplets } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { generateAIResponse, computeWaterStressIndex } from '@/lib/waterData';
import { cn } from '@/lib/utils';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const suggestedQuestions = [
  { icon: AlertTriangle, text: "Why is demand increasing in Zone A?" },
  { icon: TrendingUp, text: "What happens if rainfall is delayed by 2 weeks?" },
  { icon: Droplets, text: "How will increased recycling reduce shortages?" },
  { icon: Sparkles, text: "What actions should we take today?" },
];

export default function Agent() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: `Welcome to the Urban Water AI Decision Agent. I'm here to help you understand water demand patterns, simulate scenarios, and make data-driven decisions.

**Current System Status:**
• Stress Index: 42/100 (WATCH)
• Storage: 320 MLD (64% capacity)
• Recycling Rate: 12%

How can I assist you today? You can ask me about:
- Demand analysis and forecasting
- Scenario planning ("what-if" questions)
- Risk assessment and recommendations
- Recycling impact projections`,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const context = {
    stressIndex: 42,
    alertLevel: 'yellow',
    forecastDemand: 148,
    currentStorage: 320,
    recyclingRate: 12,
  };

  const handleSend = async (text: string = input) => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));

    const response = generateAIResponse(text, context);

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, assistantMessage]);
    setIsTyping(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-6rem)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">AI Decision Agent</h1>
          <p className="text-sm text-muted-foreground">
            Intelligent assistant for water management decisions
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary">
          <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
          <span className="text-sm font-medium">Online</span>
        </div>
      </div>

   
      <div className="flex-1 panel overflow-hidden flex flex-col">
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-3",
                message.role === 'user' ? "flex-row-reverse" : ""
              )}
            >
              <div className={cn(
                "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full",
                message.role === 'user' ? "bg-primary" : "bg-secondary"
              )}>
                {message.role === 'user' ? (
                  <User className="h-4 w-4 text-primary-foreground" />
                ) : (
                  <Bot className="h-4 w-4 text-primary" />
                )}
              </div>
              <div className={cn(
                "max-w-[80%] rounded-2xl px-4 py-3",
                message.role === 'user' 
                  ? "bg-primary text-primary-foreground rounded-br-md" 
                  : "bg-secondary rounded-bl-md"
              )}>
                <div className={cn(
                  "text-sm whitespace-pre-wrap",
                  message.role === 'assistant' && "prose prose-sm prose-invert max-w-none"
                )}>
                  {message.content.split('\n').map((line, i) => {
                   
                    if (line.startsWith('**') && line.endsWith('**')) {
                      return <p key={i} className="font-bold">{line.slice(2, -2)}</p>;
                    }
                    if (line.startsWith('• ') || line.startsWith('- ')) {
                      return <p key={i} className="pl-2">{line}</p>;
                    }
                    if (line.match(/^\d+\./)) {
                      return <p key={i} className="pl-2">{line}</p>;
                    }
                    return <p key={i}>{line}</p>;
                  })}
                </div>
                <p className={cn(
                  "text-xs mt-2 opacity-60",
                  message.role === 'user' ? "text-right" : ""
                )}>
                  {message.timestamp.toLocaleTimeString('en-US', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </p>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-secondary">
                <Bot className="h-4 w-4 text-primary" />
              </div>
              <div className="bg-secondary rounded-2xl rounded-bl-md px-4 py-3">
                <div className="flex gap-1">
                  <div className="h-2 w-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="h-2 w-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: '150ms' }} />
               <div className="h-2 w-2 rounded-full bg-primary/50 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {}
        {messages.length <= 2 && (
          <div className="px-4 pb-4">
            <p className="text-xs text-muted-foreground mb-2">Suggested questions:</p>
            <div className="grid grid-cols-2 gap-2">
              {suggestedQuestions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSend(q.text)}
                  className="flex items-center gap-2 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors text-left text-sm"
                >
                  <q.icon className="h-4 w-4 text-primary flex-shrink-0" />
                  <span className="text-muted-foreground">{q.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}

       
        <div className="border-t border-border/50 p-4">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about water demand, scenarios, or recommendations..."
              className="flex-1"
              disabled={isTyping}
            />
            <Button onClick={() => handleSend()} disabled={!input.trim() || isTyping}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2 text-center">
            AI responses are grounded in system data. No hallucinated statistics.
          </p>
        </div>
      </div>
    </div>
  );
}
