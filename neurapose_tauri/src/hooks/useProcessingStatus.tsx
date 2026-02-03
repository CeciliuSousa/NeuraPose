import { useState, useEffect, useCallback, createContext, useContext, ReactNode } from 'react';
import { APIService } from '../services/api';

export type PageStatus = 'idle' | 'processing' | 'success' | 'error';

export interface ProcessingStatusContextType {
    statuses: Record<string, PageStatus>;
    setPageStatus: (page: string, status: PageStatus) => void;
    clearPageStatus: (page: string) => void;
    isAnyProcessing: boolean;
    currentProcess: string | null;  // Qual processo está rodando agora
    hardware: SystemInfo | null;    // Métricas de hardware em tempo real
}

export interface SystemInfo {
    cpu_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
    gpu_mem_used_gb: number;
    gpu_mem_total_gb: number;
    gpu_name: string;
}

const ProcessingStatusContext = createContext<ProcessingStatusContextType | null>(null);

const LOCAL_STORAGE_KEY = 'np_page_statuses';

export function ProcessingStatusProvider({ children }: { children: ReactNode }) {
    const [statuses, setStatuses] = useState<Record<string, PageStatus>>(() => {
        // Restaurar do localStorage
        const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
        return saved ? JSON.parse(saved) : {};
    });
    const [currentProcess, setCurrentProcess] = useState<string | null>(null);
    const [hardware, setHardware] = useState<SystemInfo | null>(null);

    const setPageStatus = useCallback((page: string, status: PageStatus) => {
        setStatuses(prev => {
            const next = { ...prev, [page]: status };
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(next));
            return next;
        });
    }, []);

    // Check backend health via WebSocket
    useEffect(() => {
        // Conecta ao WS de status
        import('../services/websocket').then(mod => {
            const ws = mod.default;
            ws.connectStatus();

            ws.events.on('status', (status: any) => {
                const { is_running, current_process, process_status } = status;

                // Atualiza qual processo está rodando
                setCurrentProcess(is_running ? current_process : null);

                // Atualiza hardware se disponível no payload
                if (status.hardware) {
                    setHardware(status.hardware);
                }

                setStatuses(prev => {
                    const next = { ...prev };
                    let changed = false;

                    // Se há um processo rodando e sabemos qual é
                    if (is_running && current_process && process_status) {
                        if (next[current_process] !== process_status) {
                            next[current_process] = process_status as PageStatus;
                            changed = true;
                        }
                    } else if (!is_running) {
                        // Se não há processamento, limpa os status 'processing'
                        for (const key in next) {
                            if (next[key] === 'processing') {
                                next[key] = 'idle';
                                changed = true;
                            }
                        }

                        // Atualiza estado final
                        if (current_process && process_status && process_status !== 'processing') {
                            if (next[current_process] !== process_status) {
                                next[current_process] = process_status as PageStatus;
                                changed = true;
                            }
                        }
                    }

                    if (changed) {
                        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(next));
                    }
                    return changed ? next : prev;
                });
            });
        });

        // Fallback polling (menos frequente - 30s) para garantir sincronia se WS falhar
        const interval = setInterval(async () => {
            try {
                await APIService.healthCheck();
                // Lógica de fallback simplificada se necessário
            } catch (e) { /* ignore */ }
        }, 30000);

        return () => {
            clearInterval(interval);
            // Não desconectamos o status WS pois é global
        };
    }, []);

    const clearPageStatus = useCallback((page: string) => {
        setStatuses(prev => {
            if (prev[page] === 'processing') return prev; // Don't clear if currently processing
            const next = { ...prev };
            delete next[page];
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(next));
            return next;
        });
    }, []);

    const isAnyProcessing = Object.values(statuses).some(s => s === 'processing');

    return (
        <ProcessingStatusContext.Provider value={{ statuses, setPageStatus, clearPageStatus, isAnyProcessing, currentProcess, hardware }}>
            {children}
        </ProcessingStatusContext.Provider>
    );
}

export function useProcessingStatus() {
    const context = useContext(ProcessingStatusContext);
    if (!context) {
        throw new Error('useProcessingStatus must be used within ProcessingStatusProvider');
    }
    return context;
}
