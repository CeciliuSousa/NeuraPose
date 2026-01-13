export function PageHeader({
    title,
    description,
    children
}: {
    title: string;
    description: string;
    children?: React.ReactNode;
}) {
    return (
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between pb-6 border-b border-border mb-8">
            <div>
                <h2 className="text-3xl font-bold tracking-tight">{title}</h2>
                <p className="text-muted-foreground mt-1">
                    {description}
                </p>
            </div>
            {children && (
                <div className="flex items-center gap-2">
                    {children}
                </div>
            )}
        </div>
    );
}
