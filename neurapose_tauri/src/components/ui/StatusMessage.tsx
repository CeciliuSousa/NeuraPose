import { CheckCircle2, AlertCircle, Info } from 'lucide-react';
import { useEffect } from 'react';

interface StatusMessageProps {
    message: string;
    type?: 'success' | 'error' | 'info';
    className?: string;
    onClose?: () => void;
    autoCloseDelay?: number;
}

export function StatusMessage({ message, type = 'info', className = '', onClose, autoCloseDelay }: StatusMessageProps) {

    useEffect(() => {
        if (autoCloseDelay && onClose) {
            const timer = setTimeout(onClose, autoCloseDelay);
            return () => clearTimeout(timer);
        }
    }, [autoCloseDelay, onClose, message]);

    if (!message) return null;

    // Infer type from message content if not explicitly provided (optional legacy support)
    const effectiveType = type === 'info' && (message.includes('✅') || message.toLowerCase().includes('sucesso'))
        ? 'success'
        : type === 'info' && (message.includes('❌') || message.includes('Erro') || message.toLowerCase().includes('erro'))
            ? 'error'
            : type;

    const styles = {
        success: 'bg-green-500/10 text-green-500 border-green-500/20',
        error: 'bg-red-500/10 text-red-500 border-red-500/20',
        info: 'bg-blue-500/10 text-blue-500 border-blue-500/20'
    };

    const icons = {
        success: <CheckCircle2 className="w-5 h-5 shrink-0" />,
        error: <AlertCircle className="w-5 h-5 shrink-0" />,
        info: <Info className="w-5 h-5 shrink-0" />
    };

    return (
        <div className={`p-4 rounded-lg flex items-center gap-3 text-sm font-medium border animate-in fade-in slide-in-from-top-2 ${styles[effectiveType]} ${className}`}>
            {icons[effectiveType]}
            <span>{message}</span>
        </div>
    );
}
