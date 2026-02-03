import { useRef, useEffect, useState, useMemo, memo } from 'react';
import { Terminal as TerminalIcon } from 'lucide-react';

interface TerminalProps {
    /** Array de mensagens de log */
    logs: string[];
    /** Título do terminal (default: "Terminal") */
    title?: string;
    /** Altura máxima/fixa do terminal */
    height?: string;
    /** Largura do terminal (default: "100%") */
    width?: string;
    /** Mostrar barra de progresso (0-100) */
    progress?: number;
    /** Estado de loading/processando */
    isLoading?: boolean;
    /** Estado pausado */
    isPaused?: boolean;
    /** Mensagem de status customizada quando loading */
    statusMessage?: string;
    /** Callback para limpar logs */
    onClear?: () => void;
    /** Filtrar logs que contenham [PROGRESSO] */
    hideProgressLogs?: boolean;
    /** Classe CSS adicional */
    className?: string;
}

/**
 * Componente Terminal reutilizável para exibir logs em tempo real.
 * 
 * Features:
 * - Auto-scroll (desativa quando usuário rola manualmente)
 * - Colorização semântica de logs ([ERRO], [OK], [INFO], [CMD])
 * - Barra de progresso opcional
 * - Indicador de status no footer
 * - Design consistente estilo macOS
 */
function TerminalBase({
    logs,
    title = 'Terminal',
    height = '500px',
    width = '100%',
    progress,
    isLoading = false,
    isPaused = false,
    statusMessage,
    onClear,
    hideProgressLogs = true,
    className = ''
}: TerminalProps) {
    const terminalRef = useRef<HTMLDivElement>(null);
    const [autoScroll, setAutoScroll] = useState(true);

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

    // Filtra logs de progresso se configurado (Memoized)
    const displayLogs = useMemo(() => {
        return hideProgressLogs
            ? logs.filter(log => !log.includes('[PROGRESSO]'))
            : logs;
    }, [logs, hideProgressLogs]);

    // Determina classes de cor para cada log
    const getLogClasses = (log: string): string => {
        // Normalização das strings para garantir matching
        const isError = log.includes('[ERRO]') || log.includes('[ERROR]') || log.includes('Critico');
        const isOk = log.includes('[OK]') || log.includes('[SUCESSO]');
        const isSkip = log.includes('[SKIP]');

        // Tags de Processo (Amarelo)
        const isProcess =
            log.includes('[NORMALIZAÇÃO]') ||
            log.includes('[YOLO]') ||
            log.includes('[YOLO Stream]') ||
            log.includes('[RTMPOSE]') ||
            log.includes('[RTMPose]') ||
            log.includes('[NUCLEO]') ||
            log.includes('[PREDIÇÃO]') ||
            log.includes('[ALERT]') ||
            log.includes('[BALANCEANDO]') ||
            log.includes('[TREINANDO]') ||
            log.includes('[PROCESSAMENTO]');

        // Info e Headers (Azul)
        // Verifica [INFO] ou padrão [1/10] ou ENCONTRADOS
        const isInfo = log.includes('[INFO]') || /\[\d+\/\d+\]/.test(log) || log.includes('ENCONTRADOS');

        if (isError) return 'text-red-400 border-red-500 bg-red-500/5 font-bold';
        if (isOk) return 'text-green-400 border-green-500 bg-green-500/5 font-bold';
        if (isSkip) return 'text-purple-400 border-purple-500 bg-purple-500/5 font-semibold';
        if (isProcess) return 'text-yellow-400 border-yellow-500 bg-yellow-500/5 font-semibold';
        if (isInfo) return 'text-blue-400 border-blue-500 bg-blue-500/5 font-semibold';

        return 'text-slate-300 border-transparent';
    };

    // Texto de status no footer
    const getStatusText = (): string => {
        if (!isLoading) return 'PRONTO';
        if (statusMessage) return statusMessage;
        if (isPaused) return 'PAUSADO';
        if (progress && progress > 0) return `PROCESSANDO: ${progress}%`;
        return 'EXECUTANDO...';
    };

    return (
        <div
            className={`flex flex-col bg-slate-950 rounded-xl border border-border shadow-2xl overflow-hidden ${className}`}
            style={{ height, width }}
        >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-white/5 shrink-0">
                <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-red-500/50" />
                        <div className="w-3 h-3 rounded-full bg-orange-500/50" />
                        <div className="w-3 h-3 rounded-full bg-green-500/50" />
                    </div>
                    <TerminalIcon className="w-4 h-4 text-slate-500 ml-2" />
                    <span className="text-xs font-mono text-slate-400">{title}</span>
                </div>
                {onClear && (
                    <button
                        onClick={onClear}
                        className="text-[10px] uppercase font-bold text-slate-500 hover:text-white transition-colors"
                    >
                        Limpar
                    </button>
                )}
            </div>

            {/* Barra de Progresso */}
            {isLoading && progress !== undefined && progress > 0 && (
                <div className="px-4 py-2 bg-slate-900/50 border-b border-white/5 shrink-0">
                    <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-emerald-500 to-green-400 rounded-full transition-all duration-500 ease-out"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <span className="text-sm font-mono text-emerald-400 font-bold min-w-[50px] text-right">
                            {progress}%
                        </span>
                    </div>
                </div>
            )}

            {/* Área de Logs */}
            <div
                ref={terminalRef}
                onScroll={handleScroll}
                className="flex-1 p-4 font-mono text-sm overflow-y-auto space-y-1 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent"
            >
                {displayLogs.length === 0 ? (
                    <div className="text-slate-700 italic flex items-center justify-center h-full">
                        Aguardando início do processo...
                    </div>
                ) : (
                    displayLogs.map((log, i) => (
                        <div
                            key={i}
                            className={`whitespace-pre-wrap break-all border-l-2 pl-3 py-0.5 ${getLogClasses(log)}`}
                        >
                            {log}
                        </div>
                    ))
                )}
            </div>

            {/* Footer com Status */}
            <div className="bg-slate-900 px-4 py-2 border-t border-white/5 flex items-center justify-between shrink-0 h-10">
                <span className="text-[10px] text-slate-500 font-mono">
                    {getStatusText()}
                </span>
                {isLoading && (
                    <div className="flex gap-1">
                        <div className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-1 h-1 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                )}
            </div>
        </div>
    );
}

export const Terminal = memo(TerminalBase);
