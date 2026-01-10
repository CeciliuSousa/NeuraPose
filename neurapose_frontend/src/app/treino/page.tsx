'use client';

import { useState } from 'react';
import { PageHeader } from '@/components/ui/page-header';
import { Dumbbell, Plus, BarChart3, Database, Play } from 'lucide-react';
import { APIService } from '@/services/api';

export default function TrainingPage() {
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);

    // Config State
    const [config, setConfig] = useState({
        epochs: 100,
        batchSize: 32,
        lr: 0.0003,
        modelName: 'novo-modelo-yolo',
        datasetName: 'data-labex-completo',
        device: '0',
        temporalModel: 'tft'
    });

    const handleTrain = async () => {
        setLoading(true);
        addLog(`Iniciando treinamento do modelo: ${config.modelName}`);
        try {
            const res = await APIService.startTraining({
                epochs: config.epochs,
                batch_size: config.batchSize,
                learning_rate: config.lr,
                model_name: config.modelName,
                dataset_name: config.datasetName,
                temporal_model: config.temporalModel
            });
            addLog(`Sucesso: ${res.data.status}`);
            addLog('Treinamento iniciado em background.');
        } catch (error: any) {
            console.error(error);
            addLog(`Erro: ${error.reponse?.data?.detail || error.message}`);
        } finally {
            setLoading(false);
        }
    };

    const addLog = (msg: string) => {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col">
            <PageHeader
                title="Treinamento de Modelos"
                description="Fine-tuning e treinamento de novos modelos de detecção e pose."
            />

            <div className="grid gap-6 md:grid-cols-12 flex-1">

                {/* Left Column: Configuration */}
                <div className="md:col-span-8 flex flex-col gap-6">
                    <div className="rounded-xl border border-border bg-card p-6">
                        <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                            <Dumbbell className="w-5 h-5 text-primary" />
                            Configuração do Treino
                        </h3>

                        <div className="grid grid-cols-2 gap-4 mb-4">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Nome do Modelo</label>
                                <input
                                    type="text"
                                    className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm"
                                    value={config.modelName}
                                    onChange={e => setConfig({ ...config, modelName: e.target.value })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Dataset (Nome da pasta)</label>
                                <input
                                    type="text"
                                    className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm"
                                    value={config.datasetName}
                                    onChange={e => setConfig({ ...config, datasetName: e.target.value })}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Epochs</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm"
                                    value={config.epochs}
                                    onChange={e => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Batch Size</label>
                                <input
                                    type="number"
                                    className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm"
                                    value={config.batchSize}
                                    onChange={e => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Learning Rate</label>
                                <input
                                    type="number"
                                    step="0.0001"
                                    className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm"
                                    value={config.lr}
                                    onChange={e => setConfig({ ...config, lr: parseFloat(e.target.value) })}
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-muted-foreground">Model Arch</label>
                                <select
                                    className="w-full px-3 py-2 rounded-md bg-secondary border border-border text-sm"
                                    value={config.temporalModel}
                                    onChange={e => setConfig({ ...config, temporalModel: e.target.value })}
                                >
                                    <option value="tft">TFT (Transformer)</option>
                                    <option value="lstm">LSTM</option>
                                </select>
                            </div>
                        </div>

                        <button
                            onClick={handleTrain}
                            disabled={loading}
                            className={`w-full py-3 rounded-md font-medium text-primary-foreground flex justify-center items-center gap-2
                                ${loading ? 'bg-muted cursor-not-allowed' : 'bg-primary hover:brightness-110 shadow-lg shadow-primary/20'}
                            `}
                        >
                            <Play className="w-4 h-4 fill-current" />
                            {loading ? 'Iniciando...' : 'Iniciar Treinamento'}
                        </button>
                    </div>

                    {/* Terminal Output */}
                    <div className="rounded-xl border border-border bg-black/40 backdrop-blur flex-1 flex flex-col overflow-hidden min-h-[200px]">
                        <div className="flex-1 p-4 font-mono text-xs overflow-y-auto space-y-1">
                            {logs.length === 0 && <div className="text-muted-foreground/50 italic">Logs do sistema aparecerão aqui...</div>}
                            {logs.map((log, i) => (
                                <div key={i} className="text-blue-400 border-l-2 border-blue-900/50 pl-2">{log}</div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column: History & Datasets */}
                <div className="md:col-span-4 space-y-6">
                    <div className="rounded-xl border border-border bg-card h-full p-6">
                        <div className="flex items-center gap-2 mb-4">
                            <Database className="w-5 h-5 text-primary" />
                            <h3 className="font-semibold text-lg">Histórico (Mock)</h3>
                        </div>

                        <div className="space-y-4">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="p-3 rounded-lg border border-border bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer">
                                    <div className="flex justify-between items-start mb-1">
                                        <span className="font-medium text-sm">Treino #{i}</span>
                                        <span className="text-[10px] bg-green-500/20 text-green-500 px-2 py-0.5 rounded-full">Finalizado</span>
                                    </div>
                                    <div className="text-xs text-muted-foreground flex justify-between">
                                        <span>Epochs: 100</span>
                                        <span>Hoje</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
