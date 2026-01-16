import { useEffect, useRef, useState } from 'react';
import { X, Terminal as TerminalIcon } from 'lucide-react';

interface TerminalModalProps {
    /** Se o modal está visível */
    isOpen: boolean;
    /** Título do modal (ex: "Console de Re-identificação") */
    title: string;
    /** Logs a serem exibidos */
    logs: string[];
    /** Progresso (0-100) opcional */
    progress?: number;
    /** Estado de processamento */
    isProcessing?: boolean;
    /** Callback para fechar o modal */
    onClose: () => void;
    /** Delay em ms para fechar automaticamente após conclusão (default: 3000) */
    autoCloseDelay?: number;
    /** Callback quando o modal fecha automaticamente */
    onAutoClose?: () => void;
}

/**
 * Modal de Terminal que aparece no centro da tela durante processamentos.
 * 
 * Uso para ReID e Anotações:
 * 1. Ao clicar em "Salvar", abre o modal
 * 2. Exibe logs em tempo real do backend
 * 3. Após conclusão, espera autoCloseDelay (default 3s) e fecha
 * 4. Após fechar, callback onAutoClose é chamado
 */
export function TerminalModal({
    isOpen,
    title,
    logs,
    progress,
    isProcessing = true,
    onClose,
    autoCloseDelay = 3000,
    onAutoClose
}: TerminalModalProps) {
    const terminalRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);
    const [closing, setClosing] = useState(false);

    // Auto-scroll quando novos logs chegam
    useEffect(() => {
        if (terminalRef.current && autoScroll) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [logs, autoScroll]);

    // Detecta se usuário rolou manualmente
    const handleScroll = () => {
        if (terminalRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = terminalRef.current;
            const isAtBottom = scrollHeight - scrollTop - clientHeight < 10;
            setAutoScroll(isAtBottom);
        }
    };

    // Auto-fechar quando processamento termina
    useEffect(() => {
        if (!isProcessing && isOpen && !closing) {
            setClosing(true);
            const timer = setTimeout(() => {
                onClose();
                onAutoClose?.();
                setClosing(false);
            }, autoCloseDelay);
            return () => clearTimeout(timer);
        }
    }, [isProcessing, isOpen, onClose, onAutoClose, autoCloseDelay, closing]);

    // Reset closing state quando modal abre
    useEffect(() => {
        if (isOpen) {
            setClosing(false);
        }
    }, [isOpen]);

    // Determina classes de cor para cada log
    const getLogClasses = (log: string): string => {
        const isError = log.includes('[ERRO]') || log.includes('[ERROR]');
        const isOk = log.includes('[OK]') || log.includes('[SUCESSO]') || log.includes('concluído');
        const isInfo = log.includes('[INFO]');
        const isWarning = log.includes('[AVISO]') || log.includes('[WARNING]');

        if (isError) return 'text-red-400 border-red-500 bg-red-500/5';
        if (isOk) return 'text-green-400 border-green-500 bg-green-500/5';
        if (isWarning) return 'text-yellow-400 border-yellow-500 bg-yellow-500/5';
        if (isInfo) return 'text-blue-400 border-blue-500 bg-blue-500/5';
        return 'text-slate-300 border-transparent';
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/70 backdrop-blur-sm animate-in fade-in" />

            {/* Modal */}
            <div className="relative w-full max-w-3xl mx-4 animate-in zoom-in-95 fade-in duration-200">
                {/* Terminal Container */}
                <div className="flex flex-col bg-slate-950 rounded-xl border border-border shadow-2xl overflow-hidden max-h-[70vh]">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-white/5 shrink-0">
                        <div className="flex items-center gap-2">
                            <div className="flex gap-1.5">
                                <div className={`w-3 h-3 rounded-full ${isProcessing ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`} />
                                <div className="w-3 h-3 rounded-full bg-orange-500/50" />
                                <div className="w-3 h-3 rounded-full bg-slate-600" />
                            </div>
                            <TerminalIcon className="w-4 h-4 text-slate-500 ml-2" />
                            <span className="text-xs font-mono text-slate-400">{title}</span>
                        </div>
                        {!isProcessing && (
                            <button
                                onClick={onClose}
                                className="p-1 hover:bg-white/10 rounded transition-colors"
                            >
                                <X className="w-4 h-4 text-slate-400" />
                            </button>
                        )}
                    </div>

                    {/* Progress Bar (if provided) */}
                    {isProcessing && progress !== undefined && progress > 0 && (
                        <div className="px-4 py-2 bg-slate-900/50 border-b border-white/5 shrink-0">
                            <div className="flex items-center gap-3">
                                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-emerald-500 to-green-400 rounded-full transition-all duration-500"
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                                <span className="text-sm font-mono text-emerald-400 font-bold min-w-[50px] text-right">
                                    {progress}%
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Logs Area */}
                    <div
                        ref={terminalRef}
                        onScroll={handleScroll}
                        className="flex-1 p-4 font-mono text-sm overflow-y-auto space-y-1 min-h-[200px] max-h-[400px] scrollbar-thin scrollbar-thumb-white/10"
                    >
                        {logs.length === 0 ? (
                            <div className="text-slate-700 italic flex items-center justify-center h-full">
                                Iniciando processamento...
                            </div>
                        ) : (
                            logs.map((log, i) => (
                                <div
                                    key={i}
                                    className={`whitespace-pre-wrap break-all border-l-2 pl-3 py-0.5 ${getLogClasses(log)}`}
                                >
                                    {log}
                                </div>
                            ))
                        )}
                    </div>

                    {/* Footer */}
                    <div className="bg-slate-900 px-4 py-3 border-t border-white/5 flex items-center justify-between shrink-0">
                        <div className="flex items-center gap-2">
                            {isProcessing ? (
                                <>
                                    <div className="flex gap-1">
                                        <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                                        <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                                        <div className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                                    </div>
                                    <span className="text-[10px] text-yellow-500 font-mono uppercase tracking-wider">
                                        Processando...
                                    </span>
                                </>
                            ) : closing ? (
                                <span className="text-[10px] text-green-500 font-mono uppercase tracking-wider">
                                    ✓ Concluído! Fechando em {Math.ceil(autoCloseDelay / 1000)}s...
                                </span>
                            ) : (
                                <span className="text-[10px] text-green-500 font-mono uppercase tracking-wider">
                                    ✓ Concluído
                                </span>
                            )}
                        </div>
                        {!isProcessing && !closing && (
                            <button
                                onClick={onClose}
                                className="px-3 py-1.5 text-xs font-medium bg-green-600 text-white rounded-lg hover:bg-green-500 transition-colors"
                            >
                                Fechar Agora
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
