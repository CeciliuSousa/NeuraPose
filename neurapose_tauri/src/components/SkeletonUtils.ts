
// COCO Keypoint Topology - 17 points
const SKELETON_CONNECTIONS = [
    [15, 13], [13, 11], [16, 14], [14, 12], // Legs
    [11, 12], // Hips
    [5, 11], [6, 12], // Torso
    [5, 6], // Shoulders
    [5, 7], [7, 9], [6, 8], [8, 10], // Arms
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4] // Head
];

// Generate unique stable color from ID
function getColorForId(id: number): string {
    const colors = [
        '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF',
        '#33FFF5', '#F5FF33', '#FF8C33', '#8CFF33', '#338CFF'
    ];
    return colors[id % colors.length];
}

export function drawSkeleton(
    ctx: CanvasRenderingContext2D,
    keypoints: any[],
    id: number,
    scaleX: number,
    scaleY: number,
    offsetX: number,
    offsetY: number,
    overrideColor?: string
) {
    const color = overrideColor || getColorForId(id);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2; // Linha fina como pedido

    // Array de coordenadas [x, y] mapeadas
    const points: Array<[number, number] | null> = keypoints.map((kp: any) => {
        // Formato esperado: [x, y, conf]
        if (!kp || kp.length < 3) return null;
        const [x, y, conf] = kp;
        if (conf < 0.5) return null; // Filtra confiança baixa
        return [x * scaleX + offsetX, y * scaleY + offsetY];
    });

    // Desenha linhas (Joints)
    SKELETON_CONNECTIONS.forEach(([i, j]) => {
        const p1 = points[i];
        const p2 = points[j];
        if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1[0], p1[1]);
            ctx.lineTo(p2[0], p2[1]);
            ctx.stroke();
        }
    });

    // Desenha pontos (Keypoints)
    points.forEach(p => {
        if (p) {
            ctx.beginPath();
            ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI); // Raio pequeno (3px)
            ctx.fill();
        }
    });
}
