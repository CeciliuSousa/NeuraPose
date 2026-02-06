import { useRef, useEffect, useState, useMemo } from 'react';
import { drawSkeleton } from './SkeletonUtils';
import { RotateCcw, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Play, Pause } from 'lucide-react';

interface AnnotationPlayerProps {
    src: string;
    frameData: any;
    annotations: Record<string, string>;
    classe1?: string;
    classe2?: string;
    fps?: number;
    playbackRate?: number;
    onFrameChange?: (frame: number) => void;
}

export function AnnotationPlayer({
    src,
    frameData,
    annotations,
    classe1 = 'NORMAL',
    classe2 = 'FURTO',
    fps = 30,
    playbackRate = 0.5,
    onFrameChange
}: AnnotationPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const [isPlaying, setIsPlaying] = useState(false);
    const [displayFrame, setDisplayFrame] = useState(0);
    const [editingFrame, setEditingFrame] = useState(false);
    const [frameInputValue, setFrameInputValue] = useState('');

    const frameDuration = 1 / fps;

    const framesLookup = useMemo(() => {
        return frameData?.frames || null;
    }, [frameData]);

    const syncSize = () => {
        if (videoRef.current && canvasRef.current) {
            canvasRef.current.width = videoRef.current.clientWidth;
            canvasRef.current.height = videoRef.current.clientHeight;
        }
    };

    /**
     * Calcula os parâmetros de escala considerando letterboxing (object-contain)
     * Retorna: { scaleX, scaleY, offsetX, offsetY }
     */
    const getScalingParams = (v: HTMLVideoElement, c: HTMLCanvasElement) => {
        const vidW = v.videoWidth;
        const vidH = v.videoHeight;
        const canvasW = c.width;
        const canvasH = c.height;

        if (vidW === 0 || vidH === 0 || canvasW === 0 || canvasH === 0) {
            return null;
        }

        // Aspect ratios
        const videoAspect = vidW / vidH;
        const canvasAspect = canvasW / canvasH;

        let displayWidth: number;
        let displayHeight: number;
        let offsetX = 0;
        let offsetY = 0;

        if (videoAspect > canvasAspect) {
            // Video is wider - letterbox on top/bottom
            displayWidth = canvasW;
            displayHeight = canvasW / videoAspect;
            offsetY = (canvasH - displayHeight) / 2;
        } else {
            // Video is taller - letterbox on left/right
            displayHeight = canvasH;
            displayWidth = canvasH * videoAspect;
            offsetX = (canvasW - displayWidth) / 2;
        }

        const scaleX = displayWidth / vidW;
        const scaleY = displayHeight / vidH;

        return { scaleX, scaleY, offsetX, offsetY };
    };

    const renderOverlay = () => {
        const v = videoRef.current;
        const c = canvasRef.current;
        if (!v || !c) return;

        const ctx = c.getContext('2d');
        if (!ctx) return;

        const currentFrame = Math.round(v.currentTime * fps);
        // Otimização: Evitar re-render do React a cada frame (60Hz)
        // Só atualiza o estado se o frame integer mudar
        if (currentFrame !== displayFrame) {
            setDisplayFrame(currentFrame);
            onFrameChange?.(currentFrame);
        }

        ctx.clearRect(0, 0, c.width, c.height);

        if (!framesLookup) return;

        // Lógica "Sample-and-Hold" (Segurar último estado)
        // O tracking pode ter sido feito com vid_stride > 1 (ex: a cada 3 frames)
        // Se não tiver dados para o frame atual, procura nos anteriores (até 5 frames atrás)
        let detections: any[] | null = null;
        const LOOKBACK_LIMIT = 5;

        for (let offset = 0; offset <= LOOKBACK_LIMIT; offset++) {
            const lookupFrame = currentFrame - offset;
            if (lookupFrame < 0) break;

            const data = framesLookup[String(lookupFrame)];
            if (data && Array.isArray(data)) {
                detections = data;
                break; // Encontrou dados recentes
            }
        }

        if (!detections) return; // Nenhuma detecção recente encontrada

        // Get scaling with letterbox offset
        const params = getScalingParams(v, c);
        if (!params) return;

        const { scaleX, scaleY, offsetX, offsetY } = params;

        for (let i = 0; i < detections.length; i++) {
            const item = detections[i];
            if (!item) continue;

            const pid = item.id_persistente ?? item.botsort_id ?? item.id;
            if (pid === undefined || pid === null) continue;

            let x1: number, y1: number, x2: number, y2: number;
            if (Array.isArray(item.bbox) && item.bbox.length >= 4) {
                [x1, y1, x2, y2] = item.bbox;
            } else if (item.bbox && typeof item.bbox === 'object') {
                x1 = item.bbox.x1;
                y1 = item.bbox.y1;
                x2 = item.bbox.x2;
                y2 = item.bbox.y2;
            } else {
                continue;
            }

            const classe = annotations[String(pid)] || classe1;

            let strokeColor: string;

            if (classe === classe2) {
                strokeColor = '#ef4444';
            } else {
                strokeColor = '#22c55e';
            }

            const sx1 = x1 * scaleX + offsetX;
            const sy1 = y1 * scaleY + offsetY;
            const w = (x2 - x1) * scaleX;
            const h = (y2 - y1) * scaleY;

            // --- SKELETON DRAWING ---
            const kps = item.keypoints;
            if (kps && Array.isArray(kps)) {
                drawSkeleton(ctx, kps, pid, scaleX, scaleY, offsetX, offsetY, strokeColor);
            }

            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 3;
            ctx.strokeRect(sx1, sy1, w, h);

            const label = `ID: ${pid} | Classe: ${classe}`;
            ctx.font = 'bold 12px sans-serif';
            const textWidth = ctx.measureText(label).width + 12;
            const labelHeight = 22;

            // Fundo Branco (Pedido do User)
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.fillRect(sx1, sy1 - labelHeight, textWidth, labelHeight);

            // Texto Preto (Pedido do User)
            ctx.fillStyle = 'black';
            ctx.fillText(label, sx1 + 6, sy1 - 6);
        }
    };

    useEffect(() => {
        let animationId: number;
        let isActive = true;

        const loop = () => {
            if (!isActive) return;
            renderOverlay();
            animationId = requestAnimationFrame(loop);
        };

        animationId = requestAnimationFrame(loop);

        return () => {
            isActive = false;
            cancelAnimationFrame(animationId);
        };
    }, [framesLookup, annotations, classe1, classe2, fps, onFrameChange]);

    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;

        const handleLoadedMetadata = () => {
            v.playbackRate = playbackRate;
            syncSize();
        };

        const handlePlay = () => setIsPlaying(true);
        const handlePause = () => setIsPlaying(false);

        v.addEventListener('loadedmetadata', handleLoadedMetadata);
        v.addEventListener('play', handlePlay);
        v.addEventListener('pause', handlePause);
        window.addEventListener('resize', syncSize);

        if (v.readyState >= 1) {
            handleLoadedMetadata();
        }

        return () => {
            v.removeEventListener('loadedmetadata', handleLoadedMetadata);
            v.removeEventListener('play', handlePlay);
            v.removeEventListener('pause', handlePause);
            window.removeEventListener('resize', syncSize);
        };
    }, [src, playbackRate]);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = playbackRate;
        }
    }, [playbackRate]);

    const goToFrame = (frame: number) => {
        if (videoRef.current) {
            videoRef.current.currentTime = Math.max(0, frame * frameDuration);
        }
    };

    const stepFrame = (delta: number) => {
        goToFrame(displayFrame + delta);
    };

    const resetVideo = () => goToFrame(0);

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) videoRef.current.pause();
            else videoRef.current.play();
        }
    };

    const handleFrameInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' || e.key === 'Escape') {
            const frame = parseInt(frameInputValue);
            if (e.key === 'Enter' && !isNaN(frame) && frame >= 0) {
                goToFrame(frame);
            }
            setEditingFrame(false);
            setFrameInputValue('');
        }
    };

    const startEditingFrame = () => {
        setFrameInputValue(String(displayFrame));
        setEditingFrame(true);
    };

    return (
        <div className="flex flex-col space-y-3 select-none">
            <div ref={containerRef} className="relative rounded-lg overflow-hidden bg-black aspect-video border border-border group">
                <video
                    ref={videoRef}
                    src={src}
                    className="w-full h-full object-contain"
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 pointer-events-none"
                />
            </div>

            <div className="relative h-16 flex items-center justify-center p-3 bg-gradient-to-r from-secondary/40 via-secondary/20 to-secondary/40 rounded-xl border border-border/50 backdrop-blur-sm">
                <div className="absolute left-4">
                    <button onClick={resetVideo} className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all" title="Voltar ao início">
                        <RotateCcw className="w-4 h-4 text-white/70 hover:text-white" />
                    </button>
                </div>

                <div className="flex items-center gap-2">
                    <button onClick={() => stepFrame(-10)} className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all" title="Recuar 10 frames">
                        <ChevronsLeft className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>
                    <button onClick={() => stepFrame(-1)} className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all" title="Frame anterior">
                        <ChevronLeft className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>
                    <button onClick={togglePlay} className="w-12 h-12 flex items-center justify-center border-2 border-white/30 rounded-xl hover:bg-white/10 active:scale-95 transition-all" title={isPlaying ? 'Pausar' : 'Reproduzir'}>
                        {isPlaying ? <Pause className="w-5 h-5 text-white" /> : <Play className="w-5 h-5 text-white ml-0.5" />}
                    </button>
                    <button onClick={() => stepFrame(1)} className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all" title="Próximo frame">
                        <ChevronRight className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>
                    <button onClick={() => stepFrame(10)} className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all" title="Avançar 10 frames">
                        <ChevronsRight className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>
                </div>

                <div className="absolute right-4">
                    <div className="w-32 h-9 flex items-center justify-between px-3 text-sm font-mono bg-background/60 rounded-lg border border-border/50 cursor-pointer hover:border-primary/50 transition-colors" onClick={!editingFrame ? startEditingFrame : undefined} title="Clique para editar o frame">
                        <span className="text-muted-foreground text-xs">Frame</span>
                        {editingFrame ? (
                            <input type="number" autoFocus value={frameInputValue} onChange={(e) => setFrameInputValue(e.target.value)} onKeyDown={handleFrameInputKeyDown} onBlur={() => setEditingFrame(false)} className="w-12 bg-transparent text-foreground font-bold outline-none border-b border-primary text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none" />
                        ) : (
                            <span className="text-foreground font-bold">{displayFrame}</span>
                        )}
                    </div>
                </div>
            </div>

            <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground">
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm bg-green-500"></div>
                    <span>{classe1}</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm bg-red-500"></div>
                    <span>{classe2}</span>
                </div>
            </div>
        </div>
    );
}
