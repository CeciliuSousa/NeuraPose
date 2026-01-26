import { Cpu, Zap } from "lucide-react";

interface DeviceSelectorProps {
    value: string;
    onChange: (value: 'cuda' | 'cpu') => void;
    className?: string;
}

export function DeviceSelector({ value, onChange, className = "" }: DeviceSelectorProps) {
    return (
        <div className={`space-y-2 ${className}`}>
            <label className="text-sm font-medium text-muted-foreground italic">Hardware para Inferência</label>
            <div className="grid grid-cols-2 gap-2 p-1 bg-muted rounded-xl">
                <button
                    onClick={() => onChange('cuda')}
                    className={`py-2 text-xs font-bold rounded-lg transition-all flex items-center justify-center gap-2 ${value === 'cuda' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-background/50 text-muted-foreground'}`}
                >
                    <Zap className="w-4 h-4" />
                    GPU (CUDA)
                </button>
                <button
                    onClick={() => onChange('cpu')}
                    className={`py-2 text-xs font-bold rounded-lg transition-all flex items-center justify-center gap-2 ${value === 'cpu' ? 'bg-primary text-primary-foreground shadow-sm' : 'hover:bg-background/50 text-muted-foreground'}`}
                >
                    <Cpu className="w-4 h-4" />
                    CPU
                </button>
            </div>
        </div>
    );
}
