import axios from 'axios';

const api = axios.create({
    baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface ProcessRequest {
    input_path: string;
    output_path: string;
    onnx_path?: string;
    show_preview?: boolean;
}

export interface TrainRequest {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    model_name: string;
    dataset_name: string;
    temporal_model: string;
}

export interface BrowseResponse {
    current: string;
    parent: string;
    items: {
        name: string;
        path: string;
        is_dir: boolean;
    }[];
}

export interface ReIDVideo {
    id: string;
    json_path: string;
    video_path: string | null;
    processed: boolean;
}

export interface ReIDData {
    video_id: string;
    frames: Record<string, { bbox: number[], id: number }[]>;
    id_counts: Record<string, number>;
}

export const APIService = {
    healthCheck: () => api.get<any>('/health'),
    getConfig: () => api.get('/config'),

    browse: (path: string) => api.get<BrowseResponse>(`/browse?path=${encodeURIComponent(path)}`),

    startProcessing: (data: ProcessRequest) => api.post('/process', data),

    startTraining: (data: TrainRequest) => api.post('/train', data),

    listReIDVideos: (rootPath?: string) => api.get<{ videos: ReIDVideo[] }>(`/reid/list`, { params: { root_path: rootPath } }),

    getReIDData: (videoId: string, rootPath?: string) => api.get<ReIDData>(`/reid/${videoId}/data`, { params: { root_path: rootPath } }),

    applyReIDChanges: (videoId: string, data: any, rootPath?: string) => api.post(`/reid/${videoId}/apply`, data, { params: { root_path: rootPath } }),

    getLogs: () => api.get<{ logs: string[] }>('/logs'),

    pickFolder: () => api.get<{ path: string | null }>('/pick-folder'),

    // Configuração
    getAllConfig: () => api.get<any>('/config/all'),
    updateConfig: (updates: any) => api.post('/config/update', updates),

    // Controles de Processamento
    stopProcess: () => api.post('/process/stop'),
    pauseProcess: () => api.post('/process/pause'),
    resumeProcess: () => api.post('/process/resume'),
};

export default api;
