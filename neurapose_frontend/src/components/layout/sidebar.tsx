'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    Video,
    ScanFace,
    PenTool,
    Scissors,
    FileOutput,
    Dumbbell,
    TestTube2,
    Settings,
    Menu,
    X,
    Cpu,
    MemoryStick,
    MonitorPlay
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { APIService } from '@/services/api';

const menuItems = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Processar Vídeo', href: '/processamento', icon: Video },
    { name: 'Re-identificação', href: '/reid', icon: ScanFace },
    { name: 'Anotação', href: '/anotacao', icon: PenTool },
    { name: 'Split Datasets', href: '/datasets', icon: Scissors },
    { name: 'Converter .pt', href: '/converter', icon: FileOutput },
    { name: 'Treinamento', href: '/treino', icon: Dumbbell },
    { name: 'Testes', href: '/testes', icon: TestTube2 },
    { name: 'Configurações', href: '/configuracao', icon: Settings },
];

interface SystemInfo {
    cpu_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
    gpu_mem_used_gb: number;
    gpu_name: string;
}

export function Sidebar() {
    const pathname = usePathname();
    const [isOpen, setIsOpen] = useState(false);
    const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
    const [isOnline, setIsOnline] = useState(false);

    // Polling de informações do sistema
    useEffect(() => {
        const fetchSystemInfo = async () => {
            try {
                const res = await APIService.getSystemInfo();
                setSystemInfo(res.data);
                setIsOnline(true);
            } catch {
                setIsOnline(false);
            }
        };

        fetchSystemInfo();
        const interval = setInterval(fetchSystemInfo, 5000); // Atualiza a cada 5 segundos
        return () => clearInterval(interval);
    }, []);

    const ramPercent = systemInfo ? (systemInfo.ram_used_gb / systemInfo.ram_total_gb) * 100 : 0;

    return (
        <>
            {/* Mobile Menu Button */}
            <button
                className="md:hidden fixed top-4 left-4 z-50 p-2 bg-secondary rounded-md"
                onClick={() => setIsOpen(!isOpen)}
            >
                {isOpen ? <X /> : <Menu />}
            </button>

            {/* Sidebar Container */}
            <aside className={`
        fixed inset-y-0 left-0 z-40 w-64 bg-card border-r border-border transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
      `}>
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="h-16 flex items-center px-6 border-b border-border">
                        <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-blue-500 bg-clip-text text-transparent">
                            NeuraPose
                        </h1>
                    </div>

                    {/* Navigation */}
                    <nav className="flex-1 overflow-y-auto py-4">
                        <ul className="space-y-1 px-3">
                            {menuItems.map((item) => {
                                const Icon = item.icon;
                                const isActive = pathname === item.href;

                                return (
                                    <li key={item.href}>
                                        <Link
                                            href={item.href}
                                            className={`
                        flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors
                        ${isActive
                                                    ? 'bg-primary/10 text-primary'
                                                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'}
                      `}
                                        >
                                            <Icon className="w-5 h-5" />
                                            {item.name}
                                        </Link>
                                    </li>
                                );
                            })}
                        </ul>
                    </nav>

                    {/* Hardware Status */}
                    {systemInfo && (
                        <div className="px-4 py-3 border-t border-border space-y-2">
                            <div className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider mb-2">Hardware</div>

                            {/* CPU */}
                            <div className="flex items-center gap-2">
                                <Cpu className="w-3.5 h-3.5 text-blue-400" />
                                <div className="flex-1">
                                    <div className="flex justify-between text-[10px]">
                                        <span className="text-muted-foreground">CPU</span>
                                        <span className="text-blue-400 font-mono">{systemInfo.cpu_percent.toFixed(0)}%</span>
                                    </div>
                                    <div className="h-1 bg-muted rounded-full mt-0.5 overflow-hidden">
                                        <div
                                            className="h-full bg-blue-500 rounded-full transition-all duration-300"
                                            style={{ width: `${systemInfo.cpu_percent}%` }}
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* RAM */}
                            <div className="flex items-center gap-2">
                                <MemoryStick className="w-3.5 h-3.5 text-orange-400" />
                                <div className="flex-1">
                                    <div className="flex justify-between text-[10px]">
                                        <span className="text-muted-foreground">RAM</span>
                                        <span className={`font-mono ${ramPercent > 90 ? 'text-red-400' : 'text-orange-400'}`}>
                                            {systemInfo.ram_used_gb.toFixed(1)}/{systemInfo.ram_total_gb.toFixed(0)}GB
                                        </span>
                                    </div>
                                    <div className="h-1 bg-muted rounded-full mt-0.5 overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-300 ${ramPercent > 90 ? 'bg-red-500' : 'bg-orange-500'}`}
                                            style={{ width: `${ramPercent}%` }}
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* GPU */}
                            {systemInfo.gpu_name && (
                                <div className="flex items-center gap-2">
                                    <MonitorPlay className="w-3.5 h-3.5 text-green-400" />
                                    <div className="flex-1">
                                        <div className="flex justify-between text-[10px]">
                                            <span className="text-muted-foreground">GPU</span>
                                            <span className="text-green-400 font-mono">{systemInfo.gpu_mem_used_gb.toFixed(1)}GB</span>
                                        </div>
                                        <div className="text-[9px] text-muted-foreground truncate" title={systemInfo.gpu_name}>
                                            {systemInfo.gpu_name.replace('NVIDIA GeForce ', '')}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Footer Status */}
                    <div className="p-4 border-t border-border">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                            Backend: {isOnline ? 'Online' : 'Offline'}
                        </div>
                    </div>
                </div>
            </aside>

            {/* Overlay for mobile */}
            {isOpen && (
                <div
                    className="md:hidden fixed inset-0 z-30 bg-black/50 backdrop-blur-sm"
                    onClick={() => setIsOpen(false)}
                />
            )}
        </>
    );
}
