
import { useRef, useEffect, useState, useMemo } from 'react';
import { RotateCcw, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Play, Pause } from 'lucide-react';

interface ReidPlayerProps {
    src: string;
    reidData: any;
    swaps: { src: number; tgt: number; start: number; end: number }[];
    deletions: { id: number; start: number; end: number }[];
    cuts: { start: number; end: number }[];
    fps?: number;
}

export function ReidPlayer({ src, reidData, swaps, deletions, cuts, fps = 30 }: ReidPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [isPlaying, setIsPlaying] = useState(false);
    const [displayFrame, setDisplayFrame] = useState(0);
    const [editingFrame, setEditingFrame] = useState(false);
    const [frameInputValue, setFrameInputValue] = useState('');

    const frameDuration = 1 / fps;

    const framesLookup = useMemo(() => {
        return reidData?.frames || null;
    }, [reidData]);

    const syncSize = () => {
        if (videoRef.current && canvasRef.current) {
            canvasRef.current.width = videoRef.current.clientWidth;
            canvasRef.current.height = videoRef.current.clientHeight;
        }
    };

    /**
     * Calcula os parâmetros de escala considerando letterboxing (object-contain)
     */
    const getScalingParams = (v: HTMLVideoElement, c: HTMLCanvasElement) => {
        const vidW = v.videoWidth;
        const vidH = v.videoHeight;
        const canvasW = c.width;
        const canvasH = c.height;

        if (vidW === 0 || vidH === 0 || canvasW === 0 || canvasH === 0) {
            return null;
        }

        const videoAspect = vidW / vidH;
        const canvasAspect = canvasW / canvasH;

        let displayWidth: number;
        let displayHeight: number;
        let offsetX = 0;
        let offsetY = 0;

        if (videoAspect > canvasAspect) {
            displayWidth = canvasW;
            displayHeight = canvasW / videoAspect;
            offsetY = (canvasH - displayHeight) / 2;
        } else {
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
        if (currentFrame !== displayFrame) {
            setDisplayFrame(currentFrame);
        }

        const canvasW = c.width;
        const canvasH = c.height;

        ctx.clearRect(0, 0, canvasW, canvasH);

        // Get scaling with letterbox offset
        const params = getScalingParams(v, c);
        if (!params) return;

        const { scaleX, scaleY, offsetX, offsetY } = params;

        // Draw CUT overlay
        const inCut = cuts.some(cut => currentFrame >= cut.start && currentFrame <= cut.end);
        if (inCut) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(0, 0, canvasW, canvasH);
            ctx.fillStyle = '#ff6b6b';
            ctx.font = 'bold 20px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`⚠ TRECHO CORTADO (Frame ${currentFrame})`, canvasW / 2, 35);
            ctx.textAlign = 'left';
        }

        // Draw BBoxes
        if (framesLookup) {
            // Lógica "Sample-and-Hold" (Segurar último estado)
            // Se não tiver dados para o frame atual, procura nos anteriores (até 5 frames atrás)
            let frameData: any[] | null = null;
            const LOOKBACK_LIMIT = 5;

            for (let offset = 0; offset <= LOOKBACK_LIMIT; offset++) {
                const lookupFrame = currentFrame - offset;
                if (lookupFrame < 0) break;

                const data = framesLookup[String(lookupFrame)];
                if (data && Array.isArray(data)) {
                    frameData = data;
                    break;
                }
            }

            if (frameData && Array.isArray(frameData)) {
                for (let i = 0; i < frameData.length; i++) {
                    const item = frameData[i];
                    if (!item) continue;

                    const pid = item.id_persistente ?? item.botsort_id ?? item.id;
                    if (pid === undefined || pid === null) continue;

                    let x1: number, y1: number, x2: number, y2: number;
                    if (Array.isArray(item.bbox) && item.bbox.length >= 4) {
                        [x1, y1, x2, y2] = item.bbox;
                    } else if (item.bbox && typeof item.bbox === 'object') {
                        x1 = item.bbox.x1; y1 = item.bbox.y1; x2 = item.bbox.x2; y2 = item.bbox.y2;
                    } else {
                        continue;
                    }

                    const itemIsDeleted = deletions.some(d => d.id === pid && currentFrame >= d.start && currentFrame <= d.end);
                    const swapRule = swaps.find(s => s.src === pid && currentFrame >= s.start && currentFrame <= s.end);

                    let strokeColor: string | null = null;
                    let bgColor: string | null = null;
                    let textColor = 'white';
                    let label: string | null = null;

                    if (itemIsDeleted) {
                        strokeColor = '#ef4444';
                        bgColor = 'rgba(239, 68, 68, 0.9)';
                        label = `❌ ID ${pid}`;
                    } else if (swapRule) {
                        strokeColor = '#fbbf24';
                        bgColor = 'rgba(251, 191, 36, 0.9)';
                        textColor = 'black';
                        label = `${pid} → ${swapRule.tgt}`;
                    }

                    if (strokeColor && label && bgColor) {
                        // Apply scaling WITH offset for letterboxing
                        const sx1 = x1 * scaleX + offsetX;
                        const sy1 = y1 * scaleY + offsetY;
                        const w = (x2 - x1) * scaleX;
                        const h = (y2 - y1) * scaleY;

                        ctx.strokeStyle = strokeColor;
                        ctx.lineWidth = 3;
                        ctx.strokeRect(sx1, sy1, w, h);

                        ctx.font = 'bold 12px sans-serif';
                        const textWidth = ctx.measureText(label).width + 12;
                        const labelHeight = 22;

                        ctx.fillStyle = bgColor;
                        ctx.fillRect(sx1, sy1 - labelHeight, textWidth, labelHeight);

                        ctx.fillStyle = textColor;
                        ctx.fillText(label, sx1 + 6, sy1 - 6);
                    }
                }
            }
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
    }, [framesLookup, swaps, deletions, cuts, fps]);

    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;

        const handleLoadedMetadata = () => {
            v.playbackRate = 0.25;
            syncSize();
        };

        const handlePlay = () => setIsPlaying(true);
        const handlePause = () => setIsPlaying(false);

        v.addEventListener('loadedmetadata', handleLoadedMetadata);
        v.addEventListener('play', handlePlay);
        v.addEventListener('pause', handlePause);
        window.addEventListener('resize', syncSize);

        if (v.readyState >= 1) handleLoadedMetadata();

        return () => {
            v.removeEventListener('loadedmetadata', handleLoadedMetadata);
            v.removeEventListener('play', handlePlay);
            v.removeEventListener('pause', handlePause);
            window.removeEventListener('resize', syncSize);
        };
    }, [src]);

    const goToFrame = (frame: number) => {
        if (videoRef.current) {
            videoRef.current.currentTime = Math.max(0, frame * frameDuration);
        }
    };

    const stepFrame = (delta: number) => goToFrame(displayFrame + delta);
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
            if (e.key === 'Enter' && !isNaN(frame) && frame >= 0) goToFrame(frame);
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
            <div className="relative rounded-lg overflow-hidden bg-black aspect-video border border-border group">
                <video ref={videoRef} src={src} className="w-full h-full object-contain" />
                <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
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
                    <div className="w-3 h-3 rounded-sm bg-red-500"></div>
                    <span>ID Excluído</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm bg-amber-400"></div>
                    <span>ID Trocado</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm bg-black/60 border border-border/50"></div>
                    <span>Trecho Cortado</span>
                </div>
            </div>
        </div>
    );
}
