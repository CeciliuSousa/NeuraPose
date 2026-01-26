import { LucideIcon } from "lucide-react";

interface ConfigCardProps {
    title?: string;
    icon?: LucideIcon;
    children: React.ReactNode;
    className?: string;
}

export function ConfigCard({ title, icon: Icon, children, className = "" }: ConfigCardProps) {
    return (
        <div className={`bg-card border border-border rounded-xl p-6 shadow-sm ${className}`}>
            {title && (
                <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                    {Icon && <Icon className="w-5 h-5 text-primary" />}
                    {title}
                </h2>
            )}
            {children}
        </div>
    );
}
