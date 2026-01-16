import { X } from 'lucide-react';
import { useEffect } from 'react';
import { Terminal } from './Terminal';

interface ProcessingModalProps {
    /** Se o modal está visível */
    isOpen: boolean;
    /** Título do modal */
    title?: string;
    /** Logs a serem exibidos no terminal */
    logs: string[];
    /** Progresso (0-100) */
    progress?: number;
    /** Estado de processamento */
    isProcessing?: boolean;
    /** Callback para fechar o modal */
    onClose?: () => void;
    /** Fechar automaticamente quando isProcessing mudar para false */
    autoCloseOnComplete?: boolean;
    /** Delay em ms antes de fechar automaticamente (default: 1500) */
    autoCloseDelay?: number;
    /** Callback para limpar logs */
    onClearLogs?: () => void;
}

/**
 * Modal de processamento com terminal integrado.
 * 
 * Usado para exibir progresso de operações longas como:
 * - Salvamento de ReID/Anotações
 * - Conversão de datasets
 * - Treinamento de modelos
 */
export function ProcessingModal({
    isOpen,
    title = 'Processando...',
    logs,
    progress,
    isProcessing = true,
    onClose,
    autoCloseOnComplete = false,
    autoCloseDelay = 1500,
    onClearLogs
}: ProcessingModalProps) {

    // Auto-fecha quando o processamento termina
    useEffect(() => {
        if (autoCloseOnComplete && !isProcessing && isOpen && onClose) {
            const timer = setTimeout(onClose, autoCloseDelay);
            return () => clearTimeout(timer);
        }
    }, [isProcessing, isOpen, onClose, autoCloseOnComplete, autoCloseDelay]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in"
                onClick={!isProcessing ? onClose : undefined}
            />

            {/* Modal */}
            <div className="relative w-full max-w-3xl mx-4 bg-card border border-border rounded-xl shadow-2xl animate-in zoom-in-95 fade-in duration-200">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-border">
                    <div className="flex items-center gap-3">
                        {isProcessing && (
                            <div className="flex gap-1">
                                <div className="w-2 h-2 rounded-full bg-yellow-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-2 h-2 rounded-full bg-yellow-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-2 h-2 rounded-full bg-yellow-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                            </div>
                        )}
                        <h2 className="text-lg font-semibold">{title}</h2>
                    </div>
                    {!isProcessing && onClose && (
                        <button
                            onClick={onClose}
                            className="p-2 hover:bg-muted rounded-lg transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    )}
                </div>

                {/* Terminal */}
                <div className="p-4">
                    <Terminal
                        logs={logs}
                        title="Progresso"
                        height="350px"
                        progress={progress}
                        isLoading={isProcessing}
                        onClear={onClearLogs}
                    />
                </div>

                {/* Footer */}
                <div className="px-6 py-4 border-t border-border flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">
                        {isProcessing
                            ? 'Aguarde enquanto o processo é executado...'
                            : '✅ Processo concluído com sucesso!'
                        }
                    </span>
                    {!isProcessing && onClose && (
                        <button
                            onClick={onClose}
                            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:brightness-110 transition-all"
                        >
                            Fechar
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
