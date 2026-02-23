import { useState, useEffect } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:5000/api'

function sentimentColor(value) {
    if (value > 0.1) return '#10b981'   // strong positive
    if (value > 0.05) return '#34d399'  // positive
    if (value > -0.05) return '#94a3b8' // neutral
    if (value > -0.1) return '#f59e0b'  // negative
    return '#ef4444'                     // strong negative
}

function sentimentBg(value) {
    if (value > 0.1) return 'rgba(16,185,129,0.15)'
    if (value > 0.05) return 'rgba(52,211,153,0.1)'
    if (value > -0.05) return 'rgba(148,163,184,0.08)'
    if (value > -0.1) return 'rgba(245,158,11,0.15)'
    return 'rgba(239,68,68,0.15)'
}

export default function SentimentHeatmap() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [expanded, setExpanded] = useState(null)

    useEffect(() => {
        axios.get(`${API_BASE}/sentiment-heatmap`)
            .then(res => {
                if (res.data.success) setData(res.data.data)
            })
            .catch(() => { })
            .finally(() => setLoading(false))
    }, [])

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: 60, color: '#94a3b8' }}>
                <div className="loading-spinner" style={{ margin: '0 auto 16px' }} />
                Loading sentiment data across sectors...
            </div>
        )
    }

    if (!data || data.length === 0) {
        return (
            <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                No sentiment data available. Run the pipeline first.
            </div>
        )
    }

    return (
        <div>
            <div style={{
                background: 'rgba(17, 24, 39, 0.8)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 16, padding: 24, marginBottom: 24,
            }}>
                <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 6 }}>
                    🌡️ Sector Sentiment Heatmap
                </h2>
                <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20 }}>
                    News sentiment aggregated by sector — shows where the market buzz is focused
                </p>

                {/* Heatmap Grid */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                    gap: 12,
                }}>
                    {data.map(sector => (
                        <div
                            key={sector.sector}
                            onClick={() => setExpanded(expanded === sector.sector ? null : sector.sector)}
                            style={{
                                background: sentimentBg(sector.avg_sentiment),
                                border: `1px solid ${sentimentColor(sector.avg_sentiment)}33`,
                                borderLeft: `4px solid ${sentimentColor(sector.avg_sentiment)}`,
                                borderRadius: 12,
                                padding: 16,
                                cursor: 'pointer',
                                transition: 'all 0.3s ease',
                                transform: expanded === sector.sector ? 'scale(1.02)' : 'scale(1)',
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                                <div style={{ fontSize: 15, fontWeight: 700, color: '#f1f5f9' }}>
                                    {sector.sector}
                                </div>
                                <span style={{
                                    padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
                                    background: sentimentColor(sector.avg_sentiment) + '22',
                                    color: sentimentColor(sector.avg_sentiment),
                                }}>
                                    {sector.verdict}
                                </span>
                            </div>

                            <div style={{ fontSize: 24, fontWeight: 800, color: sentimentColor(sector.avg_sentiment), marginBottom: 6 }}>
                                {sector.avg_sentiment >= 0 ? '+' : ''}{(sector.avg_sentiment * 100).toFixed(1)}
                            </div>

                            <div style={{ display: 'flex', gap: 12, fontSize: 12, color: '#94a3b8' }}>
                                <span>🟢 {sector.positive}</span>
                                <span>⚪ {sector.neutral}</span>
                                <span>🔴 {sector.negative}</span>
                                <span style={{ marginLeft: 'auto' }}>{sector.stock_count} stocks</span>
                            </div>

                            {/* Expanded: show individual stocks */}
                            {expanded === sector.sector && (
                                <div style={{ marginTop: 12, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10 }}>
                                    {sector.stocks.map(stock => (
                                        <div key={stock.ticker} style={{
                                            display: 'flex', justifyContent: 'space-between',
                                            padding: '5px 0', fontSize: 12,
                                        }}>
                                            <span style={{ color: '#cbd5e1' }}>{stock.company}</span>
                                            <span style={{
                                                fontWeight: 700,
                                                color: sentimentColor(stock.sentiment),
                                            }}>
                                                {stock.sentiment >= 0 ? '+' : ''}{(stock.sentiment * 100).toFixed(1)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Legend */}
            <div style={{
                display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap',
                fontSize: 12, color: '#64748b', padding: '0 0 20px',
            }}>
                <span><span style={{ color: '#10b981' }}>■</span> Strong Bullish (&gt;+10)</span>
                <span><span style={{ color: '#34d399' }}>■</span> Bullish (+5 to +10)</span>
                <span><span style={{ color: '#94a3b8' }}>■</span> Neutral (−5 to +5)</span>
                <span><span style={{ color: '#f59e0b' }}>■</span> Bearish (−10 to −5)</span>
                <span><span style={{ color: '#ef4444' }}>■</span> Strong Bearish (&lt;−10)</span>
            </div>
        </div>
    )
}
