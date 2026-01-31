import { useRef, useEffect, useState, ReactNode } from 'react';
import { RotateCcw, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Play, Pause } from 'lucide-react';

interface VideoPlayerProps {
    src: string;
    fps?: number;
    onFrameChange?: (frame: number) => void;
    children?: ReactNode; // Para overlay customizado (canvas, etc)
    playbackRate?: number; // Velocidade de reprodução (controlado externamente)
    onPlaybackRateChange?: (rate: number) => void; // Callback quando velocidade muda
}

/**
 * Componente de player de vídeo reutilizável com controles de frame.
 * Usado em ReID e Anotações.
 */
export function VideoPlayer({ src, fps = 30, onFrameChange, children, playbackRate = 0.25 }: VideoPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [editingFrame, setEditingFrame] = useState(false);
    const [frameInputValue, setFrameInputValue] = useState('');

    const frameDuration = 1 / fps;
    const currentFrame = videoRef.current ? Math.round(currentTime * fps) : 0;

    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;

        const handleLoadedMetadata = () => {
            v.playbackRate = playbackRate;
        };

        const handleTimeUpdate = () => {
            setCurrentTime(v.currentTime);
            onFrameChange?.(Math.round(v.currentTime * fps));
        };

        const handlePlay = () => setIsPlaying(true);
        const handlePause = () => setIsPlaying(false);

        v.addEventListener('loadedmetadata', handleLoadedMetadata);
        v.addEventListener('timeupdate', handleTimeUpdate);
        v.addEventListener('play', handlePlay);
        v.addEventListener('pause', handlePause);

        return () => {
            v.removeEventListener('loadedmetadata', handleLoadedMetadata);
            v.removeEventListener('timeupdate', handleTimeUpdate);
            v.removeEventListener('play', handlePlay);
            v.removeEventListener('pause', handlePause);
        };
    }, [src, fps, onFrameChange, playbackRate]);

    // Atualiza playbackRate quando a prop muda
    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = playbackRate;
        }
    }, [playbackRate]);

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
                {children}
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
                    <button
                        onClick={() => stepFrame(-10)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Recuar 10 frames"
                    >
                        <ChevronsLeft className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>

                    <button
                        onClick={() => stepFrame(-1)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Frame anterior"
                    >
                        <ChevronLeft className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>

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

                    <button
                        onClick={() => stepFrame(1)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Próximo frame"
                    >
                        <ChevronRight className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>

                    <button
                        onClick={() => stepFrame(10)}
                        className="w-9 h-9 flex items-center justify-center border border-border/50 rounded-lg hover:bg-white/10 active:scale-95 transition-all"
                        title="Avançar 10 frames"
                    >
                        <ChevronsRight className="w-5 h-5 text-white/70 hover:text-white" />
                    </button>
                </div>

                {/* Right: Frame Display */}
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
        </div>
    );
}

// Exporta ref do vídeo para uso externo (overlay canvas)
export type VideoPlayerRef = {
    videoElement: HTMLVideoElement | null;
    currentFrame: number;
    goToFrame: (frame: number) => void;
};
