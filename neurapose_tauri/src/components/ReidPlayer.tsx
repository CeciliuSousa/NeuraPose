
import { useRef, useEffect, useState } from 'react';
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
    const [currentTime, setCurrentTime] = useState(0);
    const [editingFrame, setEditingFrame] = useState(false);
    const [frameInputValue, setFrameInputValue] = useState('');

    const frameDuration = 1 / fps;

    // Calculate current frame from video time
    const currentFrame = videoRef.current ? Math.round(currentTime * fps) : 0;

    // Synchronize Canvas Layout
    useEffect(() => {
        const syncSize = () => {
            if (videoRef.current && canvasRef.current) {
                canvasRef.current.width = videoRef.current.clientWidth;
                canvasRef.current.height = videoRef.current.clientHeight;
                renderOverlay();
            }
        };
        window.addEventListener('resize', syncSize);

        const v = videoRef.current;
        if (v) {
            v.addEventListener('loadedmetadata', () => {
                syncSize();
                // Slow playback for frame-by-frame viewing
                v.playbackRate = 0.25;
            });
            v.addEventListener('timeupdate', () => {
                setCurrentTime(v.currentTime);
                renderOverlay();
            });
            v.addEventListener('play', () => setIsPlaying(true));
            v.addEventListener('pause', () => setIsPlaying(false));
        }
        return () => window.removeEventListener('resize', syncSize);
    }, [src]);

    // Draw Overlay Loop
    const renderOverlay = () => {
        try {
            const v = videoRef.current;
            const c = canvasRef.current;
            if (!v || !c) return;

            const ctx = c.getContext('2d');
            if (!ctx) return;

            // Clear
            ctx.clearRect(0, 0, c.width, c.height);

            // Frame Calc
            const frame = Math.round(v.currentTime * fps);

            // Scaling factors
            const vidW = v.videoWidth;
            const vidH = v.videoHeight;
            const canvasW = c.width;
            const canvasH = c.height;

            if (vidW === 0 || vidH === 0) return;

            const scaleX = canvasW / vidW;
            const scaleY = canvasH / vidH;

            // 1. Draw CUT segments - DARK overlay (not bright red)
            const inCut = cuts.some(cut => frame >= cut.start && frame <= cut.end);
            if (inCut) {
                // Dark semi-transparent overlay
                ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
                ctx.fillRect(0, 0, canvasW, canvasH);

                // Text label
                ctx.fillStyle = '#ff6b6b';
                ctx.font = 'bold 20px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`⚠ TRECHO CORTADO (Frame ${frame})`, canvasW / 2, 35);
                ctx.textAlign = 'left';
            }

            // 2. Draw BBoxes for ALL detections in frame
            if (reidData?.frames) {
                const frameData = reidData.frames[String(frame)];
                if (frameData && Array.isArray(frameData)) {
                    frameData.forEach((item: any) => {
                        if (!item) return;

                        const pid = item.id_persistente ?? item.botsort_id ?? item.id;
                        if (pid === undefined || pid === null) return;

                        // Handle bbox as array or object
                        let x1: number, y1: number, x2: number, y2: number;
                        if (Array.isArray(item.bbox) && item.bbox.length >= 4) {
                            [x1, y1, x2, y2] = item.bbox;
                        } else if (item.bbox && typeof item.bbox === 'object') {
                            x1 = item.bbox.x1;
                            y1 = item.bbox.y1;
                            x2 = item.bbox.x2;
                            y2 = item.bbox.y2;
                        } else {
                            return;
                        }

                        // Check rules
                        const itemIsDeleted = deletions.some(d => d.id === pid && frame >= d.start && frame <= d.end);
                        const swapRule = swaps.find(s => s.src === pid && frame >= s.start && frame <= s.end);

                        // Determine color and label
                        let strokeColor: string | null = null;
                        let bgColor: string | null = null;
                        let textColor = 'white';
                        let label: string | null = null;

                        if (itemIsDeleted) {
                            // RED for deleted IDs
                            strokeColor = '#ef4444'; // red-500
                            bgColor = 'rgba(239, 68, 68, 0.9)';
                            textColor = 'white';
                            label = `❌ ID ${pid}`;
                        } else if (swapRule) {
                            // YELLOW for swapped IDs
                            strokeColor = '#fbbf24'; // amber-400
                            bgColor = 'rgba(251, 191, 36, 0.9)';
                            textColor = 'black';
                            label = `${pid} → ${swapRule.tgt}`;
                        }

                        // Draw if has rule
                        if (strokeColor && label && bgColor) {
                            const sx1 = x1 * scaleX;
                            const sy1 = y1 * scaleY;
                            const w = (x2 - x1) * scaleX;
                            const h = (y2 - y1) * scaleY;

                            // Draw box
                            ctx.strokeStyle = strokeColor;
                            ctx.lineWidth = 3;
                            ctx.strokeRect(sx1, sy1, w, h);

                            // Draw label background
                            ctx.font = 'bold 12px sans-serif';
                            const textWidth = ctx.measureText(label).width + 12;
                            const labelHeight = 22;

                            ctx.fillStyle = bgColor;
                            ctx.fillRect(sx1, sy1 - labelHeight, textWidth, labelHeight);

                            // Draw label text
                            ctx.fillStyle = textColor;
                            ctx.fillText(label, sx1 + 6, sy1 - 6);
                        }
                    });
                }
            }

        } catch (err) {
            console.error('Error rendering overlay:', err);
        }
    };

    // Re-render when data changes
    useEffect(() => {
        renderOverlay();
    }, [swaps, deletions, cuts, reidData, currentTime]);

    // Navigation functions
    const goToFrame = (frame: number) => {
        if (videoRef.current) {
            const newTime = Math.max(0, frame * frameDuration);
            videoRef.current.currentTime = newTime;
        }
    };

    const stepFrame = (delta: number) => {
        if (videoRef.current) {
            const newFrame = Math.max(0, currentFrame + delta);
            goToFrame(newFrame);
        }
    };

    const resetVideo = () => {
        goToFrame(0);
    };

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
        setFrameInputValue(String(currentFrame));
        setEditingFrame(true);
    };

    return (
        <div className="flex flex-col space-y-3 select-none">
            {/* Video Container */}
            <div className="relative rounded-lg overflow-hidden bg-black aspect-video border border-border group">
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

            {/* Controls Bar */}
            <div className="relative h-16 flex items-center justify-center p-3 bg-gradient-to-r from-secondary/40 via-secondary/20 to-secondary/40 rounded-xl border border-border/50 backdrop-blur-sm">

                {/* Left: Reset Button */}
                <div className="absolute left-4">
                    <button
                        onClick={resetVideo}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Voltar ao início (Frame 0)"
                    >
                        <RotateCcw className="w-4 h-4 text-white/70 hover:text-white" />
                    </button>
                </div>

                {/* Center: Navigation Controls */}
                <div className="flex items-center gap-2">
                    {/* Skip -10 */}
                    <button
                        onClick={() => stepFrame(-10)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Recuar 10 frames"
                    >
                        <ChevronsLeft className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>

                    {/* Step -1 */}
                    <button
                        onClick={() => stepFrame(-1)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Frame anterior"
                    >
                        <ChevronLeft className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>

                    {/* Play/Pause */}
                    <button
                        onClick={togglePlay}
                        className="w-12 h-12 flex items-center justify-center border-2 border-white/30 rounded-xl hover:bg-white/10 active:scale-95 transition-all"
                        title={isPlaying ? 'Pausar' : 'Reproduzir'}
                    >
                        {isPlaying ? (
                            <Pause className="w-5 h-5 text-white" />
                        ) : (
                            <Play className="w-5 h-5 text-white ml-0.5" />
                        )}
                    </button>

                    {/* Step +1 */}
                    <button
                        onClick={() => stepFrame(1)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Próximo frame"
                    >
                        <ChevronRight className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>

                    {/* Skip +10 */}
                    <button
                        onClick={() => stepFrame(10)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Avançar 10 frames"
                    >
                        <ChevronsRight className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>
                </div>

                {/* Right: Frame Display with Inline Editing */}
                <div className="absolute right-4">
                    <div
                        className="w-32 h-9 flex items-center justify-between px-3 text-sm font-mono bg-background/60 rounded-lg border border-border/50 cursor-pointer hover:border-primary/50 transition-colors"
                        onClick={!editingFrame ? startEditingFrame : undefined}
                        title="Clique para editar o frame"
                    >
                        <span className="text-muted-foreground text-xs">Frame</span>
                        {editingFrame ? (
                            <input
                                type="number"
                                autoFocus
                                value={frameInputValue}
                                onChange={(e) => setFrameInputValue(e.target.value)}
                                onKeyDown={handleFrameInputKeyDown}
                                onBlur={() => setEditingFrame(false)}
                                className="w-12 bg-transparent text-foreground font-bold outline-none border-b border-primary text-center [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                            />
                        ) : (
                            <span className="text-foreground font-bold">{currentFrame}</span>
                        )}
                    </div>
                </div>
            </div>

            {/* Legend */}
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
