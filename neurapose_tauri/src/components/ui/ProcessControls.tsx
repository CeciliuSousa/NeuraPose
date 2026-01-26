import { Play, Pause, Square, RefreshCcw } from "lucide-react";

interface ProcessControlsProps {
    onStart: () => void;
    onStop: () => void;
    onPause?: () => void;
    isProcessing: boolean;
    isPaused?: boolean;
    canStart?: boolean; // If false, start button is disabled
    labels?: {
        start?: string;
        stop?: string;
        pause?: string;
        resume?: string;
    };
    loadingText?: string;
}

export function ProcessControls({
    onStart,
    onStop,
    onPause,
    isProcessing,
    isPaused = false,
    canStart = true,
    labels = {},
    loadingText
}: ProcessControlsProps) {

    // Defaults
    const txtStart = labels.start || "Iniciar";
    const txtStop = labels.stop || "Parar";
    const txtPause = labels.pause || "Pausar";
    const txtResume = labels.resume || "Continuar";

    return (
        <div className="pt-4 space-y-3">
            {!isProcessing ? (
                <button
                    onClick={onStart}
                    disabled={!canStart}
                    className={`w-full py-3 bg-primary text-primary-foreground rounded-lg font-semibold transition-all flex items-center justify-center gap-2 shadow-lg shadow-primary/20 
                        ${!canStart ? 'opacity-50 cursor-not-allowed grayscale' : 'hover:brightness-110'}`}
                >
                    <Play className="w-5 h-5 fill-current" />
                    {txtStart}
                </button>
            ) : (
                <div className="flex gap-2">
                    {/* Pause Button (Optional) */}
                    {onPause && (
                        <button
                            onClick={onPause}
                            className="flex-1 py-3 bg-orange-500 text-white rounded-lg font-semibold hover:bg-orange-600 transition-all flex items-center justify-center gap-2"
                        >
                            {isPaused ? <Play className="w-5 h-5 fill-current" /> : <Pause className="w-5 h-5 fill-current" />}
                            {isPaused ? txtResume : txtPause}
                        </button>
                    )}

                    {/* Loading Indicator if no Pause or just generic info */}
                    {!onPause && loadingText && (
                        <button disabled className="flex-1 py-3 bg-muted text-muted-foreground rounded-lg font-semibold flex items-center justify-center gap-2 cursor-wait">
                            <RefreshCcw className="w-5 h-5 animate-spin" />
                            {loadingText}
                        </button>
                    )}

                    <button
                        onClick={onStop}
                        className="flex-1 py-3 bg-red-500 text-white rounded-lg font-semibold hover:bg-red-600 transition-all flex items-center justify-center gap-2"
                    >
                        <Square className="w-5 h-5 fill-current" />
                        {txtStop}
                    </button>
                </div>
            )}
        </div>
    );
}
