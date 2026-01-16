import { CheckCircle2, AlertCircle, Info, Loader2 } from 'lucide-react';
import { useEffect } from 'react';

export type StatusType = 'success' | 'error' | 'info' | 'processing';

interface StatusMessageProps {
    message: string;
    type?: StatusType;
    className?: string;
    onClose?: () => void;
    autoCloseDelay?: number;
}

/**
 * Componente de mensagem de status reutilizável.
 * 
 * Tipos:
 * - success: Verde (operação concluída)
 * - error: Vermelho (erro)
 * - info: Azul (informação)
 * - processing: Amarelo animado (em processamento)
 */
export function StatusMessage({ message, type = 'info', className = '', onClose, autoCloseDelay }: StatusMessageProps) {

    useEffect(() => {
        // Não fecha automaticamente se for processing
        if (autoCloseDelay && onClose && type !== 'processing') {
            const timer = setTimeout(onClose, autoCloseDelay);
            return () => clearTimeout(timer);
        }
    }, [autoCloseDelay, onClose, message, type]);

    if (!message) return null;

    // Infere tipo do conteúdo da mensagem (retrocompatibilidade)
    const effectiveType = type === 'info' && (message.includes('✅') || message.toLowerCase().includes('sucesso'))
        ? 'success'
        : type === 'info' && (message.includes('❌') || message.toLowerCase().includes('erro'))
            ? 'error'
            : type === 'info' && (message.includes('⏳') || message.toLowerCase().includes('processando'))
                ? 'processing'
                : type;

    const styles: Record<StatusType, string> = {
        success: 'bg-green-500/10 text-green-500 border-green-500/20',
        error: 'bg-red-500/10 text-red-500 border-red-500/20',
        info: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
        processing: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
    };

    const icons: Record<StatusType, React.ReactNode> = {
        success: <CheckCircle2 className="w-5 h-5 shrink-0" />,
        error: <AlertCircle className="w-5 h-5 shrink-0" />,
        info: <Info className="w-5 h-5 shrink-0" />,
        processing: <Loader2 className="w-5 h-5 shrink-0 animate-spin" />
    };

    return (
        <div className={`p-4 rounded-lg flex items-center gap-3 text-sm font-medium border animate-in fade-in slide-in-from-top-2 ${styles[effectiveType]} ${className}`}>
            {icons[effectiveType]}
            <span>{message}</span>
        </div>
    );
}

