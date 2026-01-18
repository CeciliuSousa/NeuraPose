import { useState, useEffect, useCallback, createContext, useContext, ReactNode } from 'react';
import { APIService } from '../services/api';

export type PageStatus = 'idle' | 'processing' | 'success' | 'error';

export interface ProcessingStatusContextType {
    statuses: Record<string, PageStatus>;
    setPageStatus: (page: string, status: PageStatus) => void;
    clearPageStatus: (page: string) => void;
    isAnyProcessing: boolean;
    currentProcess: string | null;  // Qual processo está rodando agora
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

    const setPageStatus = useCallback((page: string, status: PageStatus) => {
        setStatuses(prev => {
            const next = { ...prev, [page]: status };
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(next));
            return next;
        });
    }, []);

    // Check backend health periodically to sync processing state
    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const res = await APIService.healthCheck();
                const { processing, current_process, process_status } = res.data;

                // Atualiza qual processo está rodando
                setCurrentProcess(processing ? current_process : null);

                setStatuses(prev => {
                    const next = { ...prev };
                    let changed = false;

                    // Se há um processo rodando e sabemos qual é
                    if (processing && current_process && process_status) {
                        if (next[current_process] !== process_status) {
                            next[current_process] = process_status as PageStatus;
                            changed = true;
                        }
                    } else if (!processing) {
                        // Se não há processamento, limpa os status 'processing'
                        for (const key in next) {
                            if (next[key] === 'processing') {
                                next[key] = 'idle'; // Ou mantém o último status se desejar
                                changed = true;
                            }
                        }

                        // Opcional: Se o processo terminou com sucesso/erro recentemente, atualiza
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
            } catch (e) { /* ignore */ }
        }, 1000); // Polling mais rápido (1s) para resposta visual melhor

        return () => clearInterval(interval);
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
        <ProcessingStatusContext.Provider value={{ statuses, setPageStatus, clearPageStatus, isAnyProcessing, currentProcess }}>
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
