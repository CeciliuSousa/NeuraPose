'use client';

import { useEffect, useState } from 'react';
import api from '@/services/api';
import {
  Activity,
  Video,
  Database,
  AlertCircle,
  CheckCircle2,
  Cpu
} from 'lucide-react';
import Link from 'next/link';

interface SystemStatus {
  status: string;
  version: string;
  neurapose_root?: string;
  device?: string;
  error?: string;
}

export default function Home() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await api.get('/health');
        setStatus(response.data);
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

    checkHealth();
    // Poll every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-muted-foreground">
            Bem-vindo ao NeuraPose Desktop. Gerencie seus pipelines de visão computacional.
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
          title="Versão API"
          value={status?.version || '...'}
          icon={Activity}
          description="Build atual do backend"
        />
        <StatsCard
          title="Dispositivo"
          value={status?.device?.toUpperCase() || '...'}
          icon={Cpu}
          description="Processamento ativo"
        />
        {/* Placeholder stats */}
        <StatsCard
          title="Vídeos"
          value="-"
          icon={Video}
          description="Processados hoje"
        />
        <StatsCard
          title="Datasets"
          value="-"
          icon={Database}
          description="Disponíveis para treino"
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

function StatsCard({ title, value, icon: Icon, description }: any) {
  return (
    <div className="p-6 rounded-xl border border-border bg-card text-card-foreground shadow-sm">
      <div className="flex items-center justify-between space-y-0 pb-2">
        <p className="text-sm font-medium text-muted-foreground">{title}</p>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <p className="text-xs text-muted-foreground mt-1">{description}</p>
    </div>
  );
}

function ActionCard({ href, title, description, gradient }: any) {
  return (
    <Link
      href={href}
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
