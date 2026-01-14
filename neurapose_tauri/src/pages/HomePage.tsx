import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { APIService } from '../services/api';
import {
    Activity,
    AlertCircle,
    CheckCircle2,
    Cpu,
    Zap,
    HardDrive
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
    neurapose_root?: string;
    device?: string;
    error?: string;
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
                setStatus({
                    status: 'error',
                    version: 'Unknown',
                    error: 'Falha ao conectar com Backend'
                });
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        // 5s interval - equilibra performance e UX
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const ramPercent = sysInfo ? (sysInfo.ram_used_gb / sysInfo.ram_total_gb) * 100 : 0;
    const gpuPercent = sysInfo && sysInfo.gpu_mem_total_gb > 0
        ? (sysInfo.gpu_mem_used_gb / sysInfo.gpu_mem_total_gb) * 100
        : 0;

    return (
        <div className="space-y-8">
            {/* Header Section */}
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
                    <p className="text-muted-foreground">
                        Bem-vindo ao NeuraPose. Gerencie seus pipelines de visão computacional.
                    </p>
                </div>

                {/* Connection Status Badge */}
                <div className={`
          flex items-center gap-2 px-4 py-2 rounded-full border border-border
          ${status?.status === 'healthy' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}
        `}>
                    {loading ? (
                        <Activity className="w-5 h-5 animate-spin" />
                    ) : status?.status === 'healthy' ? (
                        <CheckCircle2 className="w-5 h-5" />
                    ) : (
                        <AlertCircle className="w-5 h-5" />
                    )}
                    <span className="font-medium">
                        {loading ? 'Verificando...' : status?.status === 'healthy' ? 'Sistema Online' : 'Sistema Offline'}
                    </span>
                </div>
            </div>

            {/* Info Cards */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <StatsCard
                    title="Uso de CPU"
                    value={`${sysInfo?.cpu_percent?.toFixed(1) || '0'}%`}
                    icon={Cpu}
                    description={`20 núcleos detectados`}
                    progress={sysInfo?.cpu_percent}
                />
                <StatsCard
                    title="Memória RAM"
                    value={`${sysInfo?.ram_used_gb?.toFixed(1) || '0'} GB`}
                    icon={HardDrive}
                    description={`De ${sysInfo?.ram_total_gb?.toFixed(1) || '-'} GB totais`}
                    progress={ramPercent}
                />
                {sysInfo?.gpu_name ? (
                    <StatsCard
                        title="Memória GPU"
                        value={`${sysInfo.gpu_mem_used_gb?.toFixed(1) || '0'} GB`}
                        icon={Zap}
                        description={`${sysInfo.gpu_name.replace('NVIDIA GeForce ', '')} | ${sysInfo.gpu_mem_total_gb?.toFixed(1) || '-'} GB`}
                        progress={gpuPercent}
                    />
                ) : (
                    <StatsCard
                        title="GPU"
                        value="N/A"
                        icon={Zap}
                        description="Nenhuma GPU encontrada"
                    />
                )}
                <StatsCard
                    title="Versão API"
                    value={status?.version || '...'}
                    icon={Activity}
                    description="Build atual do backend"
                />
            </div>


            {/* Quick Actions */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <ActionCard
                    href="/processamento"
                    title="Novo Processamento"
                    description="Iniciar detecção e pose estimation em vídeo"
                    gradient="from-blue-500/20 to-purple-500/20"
                />
                <ActionCard
                    href="/treino"
                    title="Treinar Modelo"
                    description="Iniciar fine-tuning ou treino do zero"
                    gradient="from-emerald-500/20 to-teal-500/20"
                />
                <ActionCard
                    href="/reid"
                    title="Re-identificação"
                    description="Executar módulo de rastreamento de IDs"
                    gradient="from-orange-500/20 to-red-500/20"
                />
            </div>
        </div>
    );
}

function StatsCard({ title, value, icon: Icon, description, progress }: any) {
    return (
        <div className="p-6 rounded-xl border border-border bg-card text-card-foreground shadow-sm">
            <div className="flex items-center justify-between space-y-0 pb-2">
                <p className="text-sm font-medium text-muted-foreground">{title}</p>
                <Icon className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="text-2xl font-bold">{value}</div>
            <p className="text-xs text-muted-foreground mt-1 mb-3">{description}</p>
            {progress !== undefined && (
                <div className="w-full bg-secondary h-1.5 rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-500 ${progress > 80 ? 'bg-red-500' : progress > 50 ? 'bg-yellow-500' : 'bg-green-500'}`}
                        style={{ width: `${progress}%` }}
                    />
                </div>
            )}
        </div>
    );
}


function ActionCard({ href, title, description, gradient }: any) {
    return (
        <Link
            to={href}
            className={`
        group relative overflow-hidden rounded-xl border border-border bg-card p-6 transition-all hover:shadow-md
      `}
        >
            <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-100 transition-opacity`} />
            <div className="relative z-10">
                <h3 className="font-semibold text-lg mb-2">{title}</h3>
                <p className="text-sm text-muted-foreground">{description}</p>
            </div>
        </Link>
    );
}
