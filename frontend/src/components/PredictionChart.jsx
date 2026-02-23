import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, BarChart, Bar,
    Area, AreaChart, Legend
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || !payload.length) return null
    const d = payload[0].payload
    return (
        <div style={{
            background: 'rgba(17, 24, 39, 0.95)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '10px',
            padding: '12px 16px',
            fontSize: '13px',
            color: '#f1f5f9',
            boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
        }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>{label}</div>
            <div>Open: ₹{d.open?.toLocaleString('en-IN')}</div>
            <div>High: ₹{d.high?.toLocaleString('en-IN')}</div>
            <div>Low: ₹{d.low?.toLocaleString('en-IN')}</div>
            <div style={{ fontWeight: 700 }}>Close: ₹{d.close?.toLocaleString('en-IN')}</div>
            <div style={{ color: '#94a3b8', marginTop: 4 }}>Vol: {(d.volume / 1e6).toFixed(1)}M</div>
        </div>
    )
}

export default function PredictionChart({ history, prediction }) {
    if (!history || history.length === 0) {
        return (
            <div className="chart-container" style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                No chart data available
            </div>
        )
    }

    return (
        <div className="chart-container">
            <div style={{ marginBottom: 20 }}>
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis
                            dataKey="date"
                            stroke="#64748b"
                            tick={{ fontSize: 11 }}
                            tickFormatter={(v) => v.slice(5)}
                        />
                        <YAxis
                            stroke="#64748b"
                            tick={{ fontSize: 11 }}
                            domain={['auto', 'auto']}
                            tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                            type="monotone"
                            dataKey="close"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            fill="url(#colorClose)"
                            dot={false}
                            activeDot={{ r: 4, fill: '#3b82f6' }}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>

            <ResponsiveContainer width="100%" height={80}>
                <BarChart data={history} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                    <Bar
                        dataKey="volume"
                        fill="rgba(139, 92, 246, 0.4)"
                        radius={[2, 2, 0, 0]}
                    />
                    <XAxis dataKey="date" hide />
                    <YAxis hide />
                </BarChart>
            </ResponsiveContainer>

            {prediction && (
                <div style={{
                    marginTop: 16,
                    padding: '12px 16px',
                    background: prediction.signal === 'BUY' ? 'rgba(16,185,129,0.1)' :
                        prediction.signal === 'SELL' ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)',
                    borderRadius: 10,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                }}>
                    <span style={{ fontSize: 14, fontWeight: 600 }}>
                        Next-Day Prediction: <span style={{
                            color: prediction.signal === 'BUY' ? '#10b981' :
                                prediction.signal === 'SELL' ? '#ef4444' : '#f59e0b'
                        }}>{prediction.signal}</span>
                    </span>
                    <span style={{ fontSize: 13, color: '#94a3b8' }}>
                        Probability: {(prediction.probability * 100).toFixed(1)}% · Confidence: {prediction.confidence}%
                    </span>
                </div>
            )}
        </div>
    )
}
