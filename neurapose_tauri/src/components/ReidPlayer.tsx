
import { useRef, useEffect, useState, useMemo } from 'react';

interface ReidPlayerProps {
    src: string;
    reidData: any; // The full JSON data with frames/bboxes
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

    const frameDuration = 1 / fps;

    // ================================================
    // OTIMIZAÇÃO: Pre-built lookup structures O(1)
    // ================================================

    // Set de frames cortados para lookup O(1) em vez de O(k) por frame
    const cutFrameSet = useMemo(() => {
        const set = new Set<number>();
        cuts.forEach(cut => {
            for (let f = cut.start; f <= cut.end; f++) set.add(f);
        });
        return set;
    }, [cuts]);

    // Map de (id, frame) -> true para deletions O(1) lookup
    const deletionMap = useMemo(() => {
        const map = new Map<string, boolean>();
        deletions.forEach(d => {
            for (let f = d.start; f <= d.end; f++) {
                map.set(`${d.id}:${f}`, true);
            }
        });
        return map;
    }, [deletions]);

    // Map de id -> lista de regras para swaps O(1) lookup por ID
    const swapRulesMap = useMemo(() => {
        const map = new Map<number, typeof swaps>();
        swaps.forEach(s => {
            if (!map.has(s.src)) map.set(s.src, []);
            map.get(s.src)!.push(s);
        });
        return map;
    }, [swaps]);

    // Synchronize Canvas Layout
    useEffect(() => {
        const syncSize = () => {
            if (videoRef.current && canvasRef.current) {
                canvasRef.current.width = videoRef.current.clientWidth;
                canvasRef.current.height = videoRef.current.clientHeight;
                renderOverlay(); // Force render on resize
            }
        };
        window.addEventListener('resize', syncSize);
        // Also sync when video loads metadata
        const v = videoRef.current;
        if (v) {
            v.addEventListener('loadedmetadata', syncSize);
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
        const v = videoRef.current;
        const c = canvasRef.current;
        if (!v || !c || !reidData) return;

        const ctx = c.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, c.width, c.height);

        // Frame Calc
        const currentFrame = Math.round(v.currentTime * fps);

        // Scaling factors
        // Video might be scaled in UI. Coordinates in JSON are relative to original video size.
        // We need original video dimensions.
        const vidW = v.videoWidth;
        const vidH = v.videoHeight;
        const canvasW = c.width;
        const canvasH = c.height;

        if (vidW === 0 || vidH === 0) return;

        const scaleX = canvasW / vidW;
        const scaleY = canvasH / vidH;

        // 1. Draw CUT segments (Global Overlay) - O(1) lookup
        const inCut = cutFrameSet.has(currentFrame);
        if (inCut) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            ctx.fillRect(0, 0, canvasW, canvasH);

            ctx.fillStyle = 'red';
            ctx.font = 'bold 24px monospace';
            ctx.fillText(`CORTADO (Frame ${currentFrame})`, 20, 40);
        }

        // 2. Draw BBoxes for Swaps/Deletions
        // reidData.frames[str(frame)] = list of {bbox: [x1,y1,x2,y2], id: int}
        const frameData = reidData.frames[String(currentFrame)];
        if (frameData) {
            frameData.forEach((item: any) => {
                // Prioritize id_persistente (backend standard) -> botsort_id -> id
                const pid = item.id_persistente ?? item.botsort_id ?? item.id;
                const [x1, y1, x2, y2] = item.bbox;

                // Check rules - O(1) lookups
                const isDeleted = deletionMap.has(`${pid}:${currentFrame}`);

                // Swap check: O(1) para buscar regras do ID, depois O(k) onde k = regras deste ID (tipicamente 1-2)
                let swapRule = null;
                const rulesForId = swapRulesMap.get(pid);
                if (rulesForId) {
                    swapRule = rulesForId.find(s => currentFrame >= s.start && currentFrame <= s.end);
                }

                let color = null;
                let text = null;

                if (isDeleted) {
                    color = 'red';
                    text = `DEL ${pid}`;
                } else if (swapRule) {
                    color = '#00ff00'; // Lime green
                    text = `${pid} -> ${swapRule.tgt}`;
                }

                // If modified, draw box
                if (color) {
                    const sx1 = x1 * scaleX;
                    const sy1 = y1 * scaleY;
                    const w = (x2 - x1) * scaleX;
                    const h = (y2 - y1) * scaleY;

                    ctx.strokeStyle = color;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(sx1, sy1, w, h);

                    if (text) {
                        ctx.fillStyle = color;
                        ctx.fillRect(sx1, sy1 - 20, ctx.measureText(text).width + 10, 20);
                        ctx.fillStyle = 'black';
                        ctx.font = 'bold 12px monospace';
                        ctx.fillText(text, sx1 + 5, sy1 - 5);
                    }
                }
            });
        }

        // Stats in corner
        ctx.fillStyle = 'white';
        ctx.font = '12px monospace';
        ctx.fillText(`Frame: ${currentFrame}`, canvasW - 100, 20);
    };

    // Re-render when data changes (even if paused)
    useEffect(() => {
        renderOverlay();
    }, [swaps, deletions, cuts, reidData]);

    const stepFrame = (dir: 1 | -1) => {
        if (videoRef.current) {
            videoRef.current.currentTime += dir * frameDuration;
        }
    };

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) videoRef.current.pause();
            else videoRef.current.play();
        }
    };

    return (
        <div className="flex flex-col space-y-2 select-none">
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

            {/* Custom Controls Bar - Modern Design */}
            <div className="flex items-center justify-between p-3 bg-gradient-to-r from-secondary/40 via-secondary/20 to-secondary/40 rounded-xl border border-border/50 backdrop-blur-sm">
                {/* Left spacer for centering */}
                <div className="w-28"></div>

                {/* Center: Play Controls */}
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => stepFrame(-1)}
                        className="w-10 h-10 flex items-center justify-center border border-border/50 rounded-xl hover:bg-white/10 active:scale-95 transition-all"
                        title="Frame anterior"
                    >
                        <svg className="w-5 h-5 text-white/70 hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
                        </svg>
                    </button>

                    <button
                        onClick={togglePlay}
                        className="w-14 h-14 flex items-center justify-center border-2 border-white/30 rounded-2xl hover:bg-white/10 active:scale-95 transition-all"
                    >
                        {isPlaying ? (
                            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                                <rect x="6" y="5" width="4" height="14" rx="1" />
                                <rect x="14" y="5" width="4" height="14" rx="1" />
                            </svg>
                        ) : (
                            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M6 4l15 8-15 8V4z" />
                            </svg>
                        )}
                    </button>

                    <button
                        onClick={() => stepFrame(1)}
                        className="w-10 h-10 flex items-center justify-center border border-border/50 rounded-xl hover:bg-white/10 active:scale-95 transition-all"
                        title="Próximo frame"
                    >
                        <svg className="w-5 h-5 text-white/70 hover:text-white transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
                        </svg>
                    </button>
                </div>

                {/* Right: Frame Counter */}
                <div className="w-28 flex justify-end">
                    <div className="text-sm font-mono bg-background/60 px-4 py-2 rounded-lg border border-border/50">
                        <span className="text-muted-foreground">Frame </span>
                        <span className="text-foreground font-bold text-base">{videoRef.current ? Math.round(currentTime * fps) : 0}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
