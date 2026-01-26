// EventEmitter import removed since we define a simple one below

type LogCategory = 'process' | 'train' | 'test' | 'default';

export interface LogMessage {
    type: 'logs';
    category: string;
    logs: string[];
    total: number;
}

export interface StatusMessage {
    type: 'status';
    is_running: boolean;
    is_paused: boolean;
    current_process: string | null;
    process_status: string;
}

class WebSocketService {
    private logWs: WebSocket | null = null;
    private statusWs: WebSocket | null = null;
    private logUrl = 'ws://localhost:8000/ws/logs';
    private statusUrl = 'ws://localhost:8000/ws/status';

    // Event Emitter para desacoplar a lógica
    public events = new BrowserEventEmitter();

    constructor() { }

    /**
     * Conecta ao WebSocket de logs para uma categoria específica.
     */
    connectLogs(category: LogCategory = 'process') {
        if (this.logWs) {
            this.logWs.close();
        }

        this.logWs = new WebSocket(`${this.logUrl}?category=${category}`);

        this.logWs.onopen = () => {
            console.log(`[WS] Logs connected for ${category}`);
        };

        this.logWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data) as LogMessage;
                if (data.type === 'logs') {
                    // Emite o objeto completo para permitir detecção de Full Sync (via total)
                    this.events.emit('logs', data);
                }
            } catch (e) {
                console.error('[WS] Error parsing log message', e);
            }
        };

        this.logWs.onclose = () => {
            console.log('[WS] Logs disconnected');
        };

        this.logWs.onerror = (error) => {
            console.error('[WS] Logs error', error);
        };
    }

    /**
     * Conecta ao WebSocket de status global.
     */
    connectStatus() {
        if (this.statusWs && this.statusWs.readyState === WebSocket.OPEN) return;

        this.statusWs = new WebSocket(this.statusUrl);

        this.statusWs.onopen = () => {
            console.log('[WS] Status connected');
        };

        this.statusWs.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data) as StatusMessage;
                if (data.type === 'status') {
                    this.events.emit('status', data);
                }
            } catch (e) {
                console.error('[WS] Error parsing status message', e);
            }
        };

        this.statusWs.onclose = () => {
            // Tenta reconectar após 5s
            setTimeout(() => this.connectStatus(), 5000);
        };
    }

    disconnectLogs() {
        if (this.logWs) {
            this.logWs.close();
            this.logWs = null;
        }
    }
}

// Implementação simples de EventEmitter para browser
class BrowserEventEmitter {
    private listeners: Record<string, Function[]> = {};

    on(event: string, callback: Function) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    }

    off(event: string, callback: Function) {
        if (!this.listeners[event]) return;
        this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }

    emit(event: string, data: any) {
        if (!this.listeners[event]) return;
        this.listeners[event].forEach(cb => cb(data));
    }
}

// Patch para usar a classe interna se não tiver arquivo separado
const wsService = new WebSocketService();
wsService.events = new BrowserEventEmitter();

export default wsService;
