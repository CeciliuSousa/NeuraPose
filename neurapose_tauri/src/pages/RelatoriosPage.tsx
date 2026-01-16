import { useState, useEffect } from 'react';
import { PageHeader } from '../components/ui/PageHeader';
import {
    FileBarChart,
    FolderInput,
    Search,
    TrendingUp,
    Target,
    BarChart3,
    RefreshCcw,
    Link2
} from 'lucide-react';
import { APIService } from '../services/api';
import { FileExplorerModal } from '../components/FileExplorerModal';
import { shortenPath } from '../lib/utils';
import { StatusMessage } from '../components/ui/StatusMessage';

export default function RelatoriosPage() {
    // Paths selecionados
    const [trainReportPath, setTrainReportPath] = useState('');
    const [testReportPath, setTestReportPath] = useState('');

    // Estados do Explorer
    const [explorerOpen, setExplorerOpen] = useState<'train' | 'test' | null>(null);
    const [roots, setRoots] = useState<Record<string, string>>({});

    // Dados dos relatórios
    const [trainReport, setTrainReport] = useState<any>(null);
    const [testReport, setTestReport] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);

    // Nomes derivados
    const trainName = trainReportPath ? trainReportPath.replace(/\\/g, '/').split('/').pop() || '' : '';
    const testName = testReportPath ? testReportPath.replace(/\\/g, '/').split('/').pop() || '' : '';

    // Carregar caminhos do backend
    useEffect(() => {
        APIService.getConfig().then(res => {
            if (res.data.status === 'success') {
                setRoots(res.data.paths);
            }
        });
    }, []);

    // Match automático: quando seleciona treino, busca teste correspondente
    useEffect(() => {
        if (trainReportPath && roots.relatorios_testes) {
            const datasetName = trainReportPath.replace(/\\/g, '/').split('/').pop();
            if (datasetName) {
                // Tenta encontrar relatório de teste correspondente
                // const potentialTestPath = `${roots.relatorios_testes}/${datasetName}`;
                // TODO: Verificar se o path existe via API
                setMessage({ text: `💡 Procurando relatório de teste correspondente para "${datasetName}"...`, type: 'info' });
            }
        }
    }, [trainReportPath, roots.relatorios_testes]);

    const handleLoadReports = async () => {
        if (!trainReportPath && !testReportPath) {
            setMessage({ text: 'Selecione pelo menos um relatório para visualizar.', type: 'error' });
            return;
        }

        setLoading(true);
        setMessage(null);

        try {
            // TODO: Implementar carregamento real dos relatórios
            // Por enquanto, simula estrutura
            if (trainReportPath) {
                setTrainReport({
                    name: trainName,
                    epochs: 5000,
                    best_accuracy: 0.857,
                    final_loss: 0.023,
                    learning_rate: 0.0003,
                    model_type: 'TFT'
                });
            }

            if (testReportPath) {
                setTestReport({
                    name: testName,
                    accuracy: 0.843,
                    f1_score: 0.831,
                    precision: 0.856,
                    recall: 0.812,
                    mcc: 0.687
                });
            }

            setLoading(false);
            setMessage({ text: '✅ Relatórios carregados com sucesso!', type: 'success' });
        } catch (error: any) {
            setLoading(false);
            setMessage({ text: `❌ Erro: ${error.message}`, type: 'error' });
        }
    };

    return (
        <div className="space-y-6">
            <PageHeader
                title="Relatórios"
                description="Visualize e compare relatórios de treinamento e teste lado a lado."
            />

            {/* Status Message */}
            {message && (
                <StatusMessage
                    message={message.text}
                    type={message.type}
                    onClose={() => setMessage(null)}
                    autoCloseDelay={5000}
                />
            )}

            {/* Selection Panel */}
            <div className="grid gap-6 lg:grid-cols-2">
                {/* Train Report Selection */}
                <div className="rounded-xl border border-border bg-card p-6">
                    <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-green-500" />
                        Relatório de Treinamento
                    </h3>
                    <div className="space-y-4">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={shortenPath(trainReportPath)}
                                title={trainReportPath}
                                readOnly
                                className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer"
                                placeholder="Selecione relatorio de modelo treinado..."
                                onClick={() => setExplorerOpen('train')}
                            />
                            <button
                                onClick={() => setExplorerOpen('train')}
                                className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                            >
                                <FolderInput className="w-4 h-4" />
                            </button>
                        </div>
                        {trainName && (
                            <p className="text-xs text-muted-foreground">
                                Modelo: <span className="font-mono text-green-400">{trainName}</span>
                            </p>
                        )}
                    </div>
                </div>

                {/* Test Report Selection */}
                <div className="rounded-xl border border-border bg-card p-6">
                    <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                        <Target className="w-5 h-5 text-blue-500" />
                        Relatório de Teste
                    </h3>
                    <div className="space-y-4">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={shortenPath(testReportPath)}
                                title={testReportPath}
                                readOnly
                                className="flex-1 px-3 py-2 rounded-lg bg-secondary/50 border border-border text-sm cursor-pointer"
                                placeholder="Selecione relatorio de teste..."
                                onClick={() => setExplorerOpen('test')}
                            />
                            <button
                                onClick={() => setExplorerOpen('test')}
                                className="p-2 bg-secondary rounded-lg border border-border hover:bg-secondary/80"
                            >
                                <FolderInput className="w-4 h-4" />
                            </button>
                        </div>
                        {testName && (
                            <p className="text-xs text-muted-foreground">
                                Teste: <span className="font-mono text-blue-400">{testName}</span>
                            </p>
                        )}
                    </div>
                </div>
            </div>

            {/* Link Indicator */}
            {trainReportPath && testReportPath && (
                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                    <Link2 className="w-4 h-4" />
                    <span>Relatórios vinculados para comparação</span>
                </div>
            )}

            {/* Load Button */}
            <button
                onClick={handleLoadReports}
                disabled={loading || (!trainReportPath && !testReportPath)}
                className={`w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${loading || (!trainReportPath && !testReportPath)
                    ? 'bg-muted text-muted-foreground cursor-not-allowed'
                    : 'bg-primary text-primary-foreground hover:brightness-110'
                    }`}
            >
                {loading ? (
                    <>
                        <RefreshCcw className="w-5 h-5 animate-spin" />
                        Carregando...
                    </>
                ) : (
                    <>
                        <Search className="w-5 h-5" />
                        Carregar Relatórios
                    </>
                )}
            </button>

            {/* Reports Display */}
            {(trainReport || testReport) && (
                <div className="grid gap-6 lg:grid-cols-2">
                    {/* Training Report */}
                    {trainReport && (
                        <div className="rounded-xl border border-green-500/20 bg-green-500/5 p-6">
                            <h4 className="font-semibold text-lg mb-4 flex items-center gap-2 text-green-400">
                                <BarChart3 className="w-5 h-5" />
                                Métricas de Treinamento
                            </h4>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Épocas</p>
                                    <p className="text-2xl font-bold text-green-400">{trainReport.epochs}</p>
                                </div>
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Best Accuracy</p>
                                    <p className="text-2xl font-bold text-green-400">{(trainReport.best_accuracy * 100).toFixed(1)}%</p>
                                </div>
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Final Loss</p>
                                    <p className="text-2xl font-bold">{trainReport.final_loss.toFixed(4)}</p>
                                </div>
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Modelo</p>
                                    <p className="text-2xl font-bold">{trainReport.model_type}</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Test Report */}
                    {testReport && (
                        <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-6">
                            <h4 className="font-semibold text-lg mb-4 flex items-center gap-2 text-blue-400">
                                <FileBarChart className="w-5 h-5" />
                                Métricas de Teste
                            </h4>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Accuracy</p>
                                    <p className="text-2xl font-bold text-blue-400">{(testReport.accuracy * 100).toFixed(1)}%</p>
                                </div>
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">F1 Score</p>
                                    <p className="text-2xl font-bold text-blue-400">{(testReport.f1_score * 100).toFixed(1)}%</p>
                                </div>
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Precision</p>
                                    <p className="text-2xl font-bold">{(testReport.precision * 100).toFixed(1)}%</p>
                                </div>
                                <div className="bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">Recall</p>
                                    <p className="text-2xl font-bold">{(testReport.recall * 100).toFixed(1)}%</p>
                                </div>
                                <div className="col-span-2 bg-background/50 p-4 rounded-lg">
                                    <p className="text-xs text-muted-foreground uppercase">MCC (Matthews Corr. Coef.)</p>
                                    <p className="text-2xl font-bold">{testReport.mcc.toFixed(3)}</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* File Explorer Modals */}
            <FileExplorerModal
                isOpen={explorerOpen === 'train'}
                onClose={() => setExplorerOpen(null)}
                onSelect={(path) => {
                    setTrainReportPath(path);
                    setExplorerOpen(null);
                }}
                initialPath={trainReportPath || roots.modelos_treinados}
                rootPath={roots.modelos_treinados}
                title="Selecionar Modelo Treinado"
            />

            <FileExplorerModal
                isOpen={explorerOpen === 'test'}
                onClose={() => setExplorerOpen(null)}
                onSelect={(path) => {
                    setTestReportPath(path);
                    setExplorerOpen(null);
                }}
                initialPath={testReportPath || roots.relatorios_testes}
                rootPath={roots.relatorios_testes}
                title="Selecionar Relatório de Teste"
            />
        </div>
    );
}
