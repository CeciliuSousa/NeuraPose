import { useState, useEffect, useRef } from 'react';
import { Video, RefreshCw, WifiOff, Activity } from 'lucide-react';

interface VideoPreviewPanelProps {
    isVisible: boolean;
    feedUrl?: string;
    title?: string;
}

export function VideoPreviewPanel({
    isVisible,
    feedUrl = "http://localhost:8000/video_feed",
    title = "Live Preview"
}: VideoPreviewPanelProps) {
    const [currentUrl, setCurrentUrl] = useState(feedUrl);
    const [error, setError] = useState(false);
    const [retryCount, setRetryCount] = useState(0);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Reset quando torna-se visível
    useEffect(() => {
        if (isVisible) {
            setError(false);
            setRetryCount(0);
            updateUrl();
        } else {
            // Limpa timeout se esconder
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
        }
    }, [isVisible, feedUrl]);

    const updateUrl = () => {
        // Adiciona timestamp para evitar cache do navegador e forçar nova conexão
        const timestamp = new Date().getTime();
        const separator = feedUrl.includes('?') ? '&' : '?';
        setCurrentUrl(`${feedUrl}${separator}_t=${timestamp}`);
        setError(false);
    };

    const handleError = () => {
        setError(true);
        // Tenta reconectar a cada 2 segundos
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        timeoutRef.current = setTimeout(() => {
            setRetryCount(prev => prev + 1);
            updateUrl();
        }, 2000);
    };

    const handleLoad = () => {
        setError(false);
        setRetryCount(0);
    };

    if (!isVisible) return null;

    return (
        <div className="bg-card border border-border rounded-xl overflow-hidden shadow-lg animate-in fade-in zoom-in duration-300">
            {/* Header com estilo similar ao VideoPlayer */}
            <div className="relative h-12 flex items-center justify-between px-4 bg-gradient-to-r from-secondary/40 via-secondary/20 to-secondary/40 border-b border-border/50 backdrop-blur-sm">
                <div className="flex items-center gap-2">
                    <span className="relative flex h-3 w-3">
                        {!error ? (
                            <>
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                            </>
                        ) : (
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                        )}
                    </span>
                    <span className="text-xs font-bold uppercase tracking-wider text-foreground/80">
                        {title}
                    </span>
                </div>

                {error && (
                    <span className="text-xs text-red-400 flex items-center gap-1 animate-pulse">
                        <WifiOff className="w-3 h-3" />
                        Reconectando ({retryCount})...
                    </span>
                )}

                {!error && (
                    <span className="text-xs text-green-400 flex items-center gap-1">
                        <Activity className="w-3 h-3" />
                        Online
                    </span>
                )}
            </div>

            {/* Video Container */}
            <div className="aspect-video bg-black flex items-center justify-center relative group">
                <img
                    src={currentUrl}
                    alt="Live Feed"
                    className={`w-full h-full object-contain transition-opacity duration-300 ${error ? 'opacity-20' : 'opacity-100'}`}
                    onError={handleError}
                    onLoad={handleLoad}
                />

                {/* Overlay de Loading / Erro */}
                {(error) && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-3">
                        <div className="relative">
                            <Video className="w-12 h-12 opacity-20" />
                            <RefreshCw className="w-6 h-6 absolute bottom-0 right-0 animate-spin text-primary" />
                        </div>
                        <p className="text-xs font-mono opacity-60">Aguardando sinal de vídeo...</p>
                    </div>
                )}

                {/* Botão de Refresh Manual (aparece no hover) */}
                <button
                    onClick={updateUrl}
                    className="absolute top-4 right-4 p-2 bg-black/50 hover:bg-black/80 text-white rounded-full opacity-0 group-hover:opacity-100 transition-all border border-white/20"
                    title="Recarregar Stream"
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
            </div>

            {/* Footer com stats (opcional) */}
            <div className="px-4 py-1.5 bg-muted/30 border-t border-border/50 flex justify-end">
                <span className="text-[10px] font-mono text-muted-foreground/60">
                    MJPEG Stream • Low Latency
                </span>
            </div>
        </div>
    );
}
