import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { APIService } from '../services/api';
import {
    Activity,
    AlertCircle,
    CheckCircle2,
    Cpu,
    Zap,
    HardDrive,
    Video,
    ScanFace,
    PenTool,
    Scissors,
    FileOutput,
    Dumbbell,
    TestTube2,
    FileBarChart,
    History
} from 'lucide-react';

interface SystemInfo {
    cpu_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
    gpu_mem_used_gb: number;
    gpu_mem_total_gb: number;
    gpu_name: string;
}

interface SystemStatus {
    status: string;
    version: string;
}

export default function HomePage() {
    const [status, setStatus] = useState<SystemStatus | null>(null);
    const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [healthRes, infoRes] = await Promise.all([
                    APIService.healthCheck(),
                    APIService.getSystemInfo()
                ]);
                setStatus(healthRes.data);
                setSysInfo(infoRes.data);
            } catch (err) {
                setStatus({ status: 'error', version: 'Unknown' });
            } finally {
                setLoading(false);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const ramPercent = sysInfo ? (sysInfo.ram_used_gb / sysInfo.ram_total_gb) * 100 : 0;
    const gpuPercent = sysInfo && sysInfo.gpu_mem_total_gb > 0
        ? (sysInfo.gpu_mem_used_gb / sysInfo.gpu_mem_total_gb) * 100
        : 0;

    return (
        <div className="space-y-10 pb-10">
            {/* Header Section */}
            <div className="flex justify-between items-center bg-card/30 p-6 rounded-2xl border border-border shadow-sm">
                <div>
                    <h2 className="text-4xl font-black tracking-tight bg-gradient-to-r from-primary to-blue-400 bg-clip-text text-transparent">
                        NeuraPose Control
                    </h2>
                    <p className="text-muted-foreground mt-1">
                        Pipeline de Visão Computacional de Alta Performance
                    </p>
                </div>

                <div className={`
          flex items-center gap-3 px-5 py-2.5 rounded-full border shadow-lg transition-all
          ${status?.status === 'healthy' ? 'bg-green-500/10 border-green-500/20 text-green-500' : 'bg-red-500/10 border-red-500/20 text-red-500'}
        `}>
                    {loading ? (
                        <Activity className="w-5 h-5 animate-spin" />
                    ) : status?.status === 'healthy' ? (
                        <CheckCircle2 className="w-5 h-5" />
                    ) : (
                        <AlertCircle className="w-5 h-5" />
                    )}
                    <span className="font-bold tracking-wide uppercase text-xs">
                        {loading ? 'Sincronizando...' : status?.status === 'healthy' ? 'Backend: online' : 'Backend offline'}
                    </span>
                </div>
            </div>

            {/* Info Cards */}
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                <StatsCard
                    title="Processamento CPU"
                    value={`${sysInfo?.cpu_percent?.toFixed(1) || '0'}%`}
                    icon={Cpu}
                    description="Carga média do processador"
                    progress={sysInfo?.cpu_percent}
                    color="blue"
                />
                <StatsCard
                    title="Memória RAM"
                    value={`${sysInfo?.ram_used_gb?.toFixed(1) || '0'} GB`}
                    icon={HardDrive}
                    description={`De ${sysInfo?.ram_total_gb?.toFixed(1) || '-'} GB`}
                    progress={ramPercent}
                    color="orange"
                />
                <StatsCard
                    title="VRAM dedicada"
                    value={sysInfo?.gpu_name ? `${sysInfo.gpu_mem_used_gb?.toFixed(1) || '0'} GB` : 'OFF'}
                    icon={Zap}
                    description={sysInfo?.gpu_name ? sysInfo.gpu_name.replace('NVIDIA GeForce ', '') : 'Sem GPU'}
                    progress={sysInfo?.gpu_name ? gpuPercent : 0}
                    color="green"
                />
                <StatsCard
                    title="Ambiente Neura"
                    value={status?.version || '0.0.0'}
                    icon={Activity}
                    description="Versão estável do sistema"
                    color="purple"
                />
            </div>

            {/* Sections Categorizadas */}
            <div className="space-y-8">
                {/* 1. Pipeline de Dados */}
                <section>
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-1 h-6 bg-blue-500 rounded-full" />
                        <h3 className="text-xl font-bold">Pipeline de Pré-processamento</h3>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        <ActionCard
                            href="/processamento"
                            title="Extração de Pose"
                            description="Executar YOLOv8 + RTMPose em vídeos brutos"
                            icon={Video}
                            gradient="from-blue-600/20 to-blue-400/10 hover:border-blue-500/30"
                        />
                        <ActionCard
                            href="/reid"
                            title="Re-identificação"
                            description="Corrigir IDs e persistir rastreamento manual"
                            icon={ScanFace}
                            gradient="from-indigo-600/20 to-indigo-400/10 hover:border-indigo-500/30"
                        />
                        <ActionCard
                            href="/anotacao"
                            title="Anotação Semiautística"
                            description="Classificar poses para treinamento de rede"
                            icon={PenTool}
                            gradient="from-cyan-600/20 to-cyan-400/10 hover:border-cyan-500/30"
                        />
                    </div>
                </section>

                {/* 2. Preparação de Dataset e Treino */}
                <section>
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-1 h-6 bg-emerald-500 rounded-full" />
                        <h3 className="text-xl font-bold">Treinamento e Modelagem</h3>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                        <ActionCard
                            href="/split"
                            title="Split Data"
                            description="Dividir Treino/Teste"
                            icon={Scissors}
                            gradient="from-emerald-600/20 to-emerald-400/10 hover:border-emerald-500/30"
                        />
                        <ActionCard
                            href="/converter"
                            title="Converter .pt"
                            description="Exportar para PyTorch"
                            icon={FileOutput}
                            gradient="from-teal-600/20 to-teal-400/10 hover:border-teal-500/30"
                        />
                        <ActionCard
                            href="/treino"
                            title="Iniciar Treino"
                            description="LSTM / TFT Models"
                            icon={Dumbbell}
                            gradient="from-green-600/20 to-green-400/10 hover:border-green-500/30"
                        />
                        <ActionCard
                            href="/testes"
                            title="Bateria de Testes"
                            description="Validar Acurácia"
                            icon={TestTube2}
                            gradient="from-lime-600/20 to-lime-400/10 hover:border-lime-500/30"
                        />
                    </div>
                </section>

                {/* 3. Analytics e Histórico */}
                <section>
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-1 h-6 bg-orange-500 rounded-full" />
                        <h3 className="text-xl font-bold">Monitoramento e Resultados</h3>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        <ActionCard
                            href="/historico"
                            title="Histórico Global"
                            description="Visualizar todos os arquivos e resultados gerados"
                            icon={History}
                            gradient="from-orange-600/20 to-orange-400/10 hover:border-orange-500/30"
                        />
                        <ActionCard
                            href="/relatorios"
                            title="Analytics"
                            description="Comparar métricas de modelos"
                            icon={FileBarChart}
                            gradient="from-amber-600/20 to-amber-400/10 hover:border-amber-500/30"
                        />
                    </div>
                </section>
            </div>
        </div>
    );
}

function StatsCard({ title, value, icon: Icon, description, progress, color }: any) {
    const colorClasses: any = {
        blue: 'text-blue-500 bg-blue-500',
        orange: 'text-orange-500 bg-orange-500',
        green: 'text-green-500 bg-green-500',
        purple: 'text-purple-500 bg-purple-500'
    };

    return (
        <div className="p-6 rounded-2xl border border-border bg-card/60 backdrop-blur-sm shadow-lg hover:border-primary/20 transition-all group">
            <div className="flex items-center justify-between space-y-0 pb-3">
                <p className="text-xs font-bold uppercase tracking-widest text-muted-foreground group-hover:text-primary transition-colors">{title}</p>
                <div className={`p-2 rounded-lg bg-muted group-hover:bg-primary/10 transition-colors`}>
                    <Icon className={`h-4 w-4 ${colorClasses[color].split(' ')[0]}`} />
                </div>
            </div>
            <div className="text-3xl font-black">{value}</div>
            <p className="text-[10px] text-muted-foreground mt-1 mb-4 italic">{description}</p>
            {progress !== undefined && (
                <div className="w-full bg-muted h-1 rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-700 ease-out ${colorClasses[color].split(' ')[1]}`}
                        style={{ width: `${progress}%` }}
                    />
                </div>
            )}
        </div>
    );
}

function ActionCard({ href, title, description, icon: Icon, gradient }: any) {
    return (
        <Link
            to={href}
            className={`
        group relative overflow-hidden rounded-2xl border border-border bg-card/40 p-6 transition-all 
        hover:shadow-2xl hover:-translate-y-1 active:scale-95
      `}
        >
            <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
            <div className="relative z-10 space-y-3">
                <div className="p-2.5 w-fit rounded-xl bg-muted border border-border group-hover:border-primary/20 transition-all">
                    <Icon className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
                <div>
                    <h3 className="font-bold text-lg group-hover:text-primary transition-colors leading-tight">{title}</h3>
                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{description}</p>
                </div>
            </div>
        </Link>
    );
}
