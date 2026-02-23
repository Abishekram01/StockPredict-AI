import { useState, useEffect } from 'react'
import axios from 'axios'
import NewsPanel from './NewsPanel'

const API_BASE = 'http://localhost:5000/api'

function InsightCard({ item, type }) {
    const [expanded, setExpanded] = useState(false)
    const pred = item.prediction
    const ins = item.insights
    const isBuy = type === 'buy'

    const accentColor = isBuy ? '#10b981' : '#ef4444'
    const accentGlow = isBuy ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)'
    const signalEmoji = isBuy ? '🟢' : '🔴'

    return (
        <div style={{
            background: 'rgba(17, 24, 39, 0.8)',
            backdropFilter: 'blur(20px)',
            border: `1px solid rgba(255,255,255,0.08)`,
            borderLeft: `4px solid ${accentColor}`,
            borderRadius: 16,
            padding: 24,
            boxShadow: '0 4px 30px rgba(0,0,0,0.4)',
        }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
                <div>
                    <div style={{ fontSize: 22, fontWeight: 800 }}>
                        {signalEmoji} {pred.company}
                    </div>
                    <div style={{ color: '#94a3b8', fontSize: 14, marginTop: 2 }}>
                        {pred.ticker.replace('.NS', '')} · {ins.profile?.sector || 'N/A'} · {ins.profile?.industry || 'N/A'}
                    </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                    <span style={{
                        padding: '6px 16px',
                        borderRadius: 20,
                        fontSize: 13,
                        fontWeight: 700,
                        background: accentGlow,
                        color: accentColor,
                        boxShadow: `0 0 15px ${accentGlow}`,
                    }}>
                        {pred.signal} · {pred.confidence}%
                    </span>
                    <div style={{ fontSize: 24, fontWeight: 700, marginTop: 8 }}>
                        ₹{pred.current_price?.toLocaleString('en-IN')}
                    </div>
                    <div style={{
                        fontSize: 13, fontWeight: 600,
                        color: pred.price_change_pct >= 0 ? '#10b981' : '#ef4444'
                    }}>
                        {pred.price_change_pct >= 0 ? '+' : ''}{pred.price_change_pct}%
                    </div>
                </div>
            </div>

            {/* Key Metrics Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))',
                gap: 10,
                marginBottom: 20,
            }}>
                <MetricBadge label="Market Cap" value={ins.financials?.market_cap_formatted} />
                <MetricBadge label="P/E Ratio" value={ins.valuation?.pe_ratio} />
                <MetricBadge label="Forward P/E" value={ins.valuation?.forward_pe} />
                <MetricBadge label="P/B Ratio" value={ins.valuation?.price_to_book} />
                <MetricBadge label="ROE" value={ins.financials?.return_on_equity} />
                <MetricBadge label="Profit Margin" value={ins.financials?.profit_margin} />
                <MetricBadge label="Debt/Equity" value={ins.financials?.debt_to_equity} />
                <MetricBadge label="Div Yield" value={ins.dividend?.dividend_yield} />
            </div>

            {/* Price Range */}
            <div style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 10,
                padding: 16,
                marginBottom: 16,
            }}>
                <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                    52-Week Range
                </div>
                <PriceRange
                    low={ins.price_info?.fifty_two_week_low}
                    high={ins.price_info?.fifty_two_week_high}
                    current={pred.current_price}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, fontSize: 13 }}>
                    <span style={{ color: '#94a3b8' }}>50-Day Avg: <b style={{ color: '#f1f5f9' }}>₹{ins.price_info?.fifty_day_avg?.toLocaleString('en-IN')}</b></span>
                    <span style={{ color: '#94a3b8' }}>200-Day Avg: <b style={{ color: '#f1f5f9' }}>₹{ins.price_info?.two_hundred_day_avg?.toLocaleString('en-IN')}</b></span>
                    <span style={{ color: '#94a3b8' }}>Beta: <b style={{ color: '#f1f5f9' }}>{ins.price_info?.beta}</b></span>
                </div>
            </div>

            {/* Analyst Target */}
            {ins.analyst?.target_mean > 0 && (
                <div style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: 10,
                    padding: 16,
                    marginBottom: 16,
                }}>
                    <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                        Analyst Consensus ({ins.analyst?.num_analysts} analysts)
                    </div>
                    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
                        <Stat label="Recommendation" value={ins.analyst?.recommendation?.toUpperCase()} color={
                            ins.analyst?.recommendation === 'buy' || ins.analyst?.recommendation === 'strong_buy' ? '#10b981' :
                                ins.analyst?.recommendation === 'sell' ? '#ef4444' : '#f59e0b'
                        } />
                        <Stat label="Target Low" value={`₹${ins.analyst?.target_low?.toLocaleString('en-IN')}`} />
                        <Stat label="Target Mean" value={`₹${ins.analyst?.target_mean?.toLocaleString('en-IN')}`} color="#3b82f6" />
                        <Stat label="Target High" value={`₹${ins.analyst?.target_high?.toLocaleString('en-IN')}`} />
                        {pred.current_price > 0 && ins.analyst?.target_mean > 0 && (
                            <Stat
                                label="Upside"
                                value={`${(((ins.analyst.target_mean - pred.current_price) / pred.current_price) * 100).toFixed(1)}%`}
                                color={ins.analyst.target_mean > pred.current_price ? '#10b981' : '#ef4444'}
                            />
                        )}
                    </div>
                </div>
            )}

            {/* News Sentiment Summary */}
            <div style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 10,
                padding: 16,
                marginBottom: 16,
            }}>
                <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                    News Sentiment ({ins.news_summary?.total_articles} articles)
                </div>
                <div style={{ display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
                    <Stat label="Verdict" value={ins.news_summary?.sentiment_verdict} color={
                        ins.news_summary?.sentiment_verdict === 'Bullish' ? '#10b981' :
                            ins.news_summary?.sentiment_verdict === 'Bearish' ? '#ef4444' : '#f59e0b'
                    } />
                    <Stat label="Positive" value={ins.news_summary?.positive} color="#10b981" />
                    <Stat label="Negative" value={ins.news_summary?.negative} color="#ef4444" />
                    <Stat label="Neutral" value={ins.news_summary?.neutral} color="#64748b" />
                    <Stat label="Avg Score" value={ins.news_summary?.average_sentiment?.toFixed(3)} />
                </div>
            </div>

            {/* Expand / Collapse for details */}
            <button
                onClick={() => setExpanded(!expanded)}
                style={{
                    width: '100%',
                    padding: '10px 0',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 8,
                    color: '#94a3b8',
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                }}
            >
                {expanded ? '▲ Hide Details' : '▼ Show Company Profile & News'}
            </button>

            {expanded && (
                <div style={{ marginTop: 20 }}>
                    {/* Company Description */}
                    {ins.profile?.description && ins.profile.description !== 'N/A' && (
                        <div style={{ marginBottom: 20 }}>
                            <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
                                About {ins.profile?.name}
                            </div>
                            <p style={{ fontSize: 14, color: '#cbd5e1', lineHeight: 1.7 }}>
                                {ins.profile.description}
                            </p>
                            <div style={{ display: 'flex', gap: 16, marginTop: 10, fontSize: 13, color: '#94a3b8' }}>
                                {ins.profile?.employees > 0 && <span>👥 {ins.profile.employees.toLocaleString()} employees</span>}
                                {ins.profile?.website && ins.profile.website !== 'N/A' && (
                                    <a href={ins.profile.website} target="_blank" rel="noopener" style={{ color: '#3b82f6' }}>🌐 Website</a>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Detailed Financials */}
                    <div style={{ marginBottom: 20 }}>
                        <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                            Financial Details
                        </div>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                            gap: 8,
                        }}>
                            <MetricBadge label="Revenue" value={ins.financials?.revenue_formatted} />
                            <MetricBadge label="Net Income" value={ins.financials?.net_income_formatted} />
                            <MetricBadge label="EBITDA" value={ins.financials?.ebitda} />
                            <MetricBadge label="Free Cash Flow" value={ins.financials?.free_cash_flow} />
                            <MetricBadge label="Operating Margin" value={ins.financials?.operating_margin} />
                            <MetricBadge label="Gross Margin" value={ins.financials?.gross_margin} />
                            <MetricBadge label="ROA" value={ins.financials?.return_on_assets} />
                            <MetricBadge label="Current Ratio" value={ins.financials?.current_ratio} />
                            <MetricBadge label="EV/EBITDA" value={ins.valuation?.ev_to_ebitda} />
                            <MetricBadge label="EV/Revenue" value={ins.valuation?.ev_to_revenue} />
                            <MetricBadge label="PEG Ratio" value={ins.valuation?.peg_ratio} />
                            <MetricBadge label="P/S Ratio" value={ins.valuation?.price_to_sales} />
                        </div>
                    </div>

                    {/* News */}
                    <div>
                        <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                            Latest News & Headlines
                        </div>
                        <NewsPanel news={ins.news?.slice(0, 10)} />
                    </div>
                </div>
            )}
        </div>
    )
}


function MetricBadge({ label, value }) {
    return (
        <div style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 8,
            padding: '10px 12px',
        }}>
            <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
            <div style={{ fontSize: 15, fontWeight: 700, marginTop: 4, color: '#f1f5f9' }}>
                {value ?? 'N/A'}
            </div>
        </div>
    )
}

function Stat({ label, value, color }) {
    return (
        <div>
            <div style={{ fontSize: 11, color: '#64748b' }}>{label}</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: color || '#f1f5f9' }}>{value ?? 'N/A'}</div>
        </div>
    )
}

function PriceRange({ low, high, current }) {
    if (!low || !high || !current) return null
    const pct = Math.min(100, Math.max(0, ((current - low) / (high - low)) * 100))
    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
                <span style={{ color: '#ef4444' }}>₹{low?.toLocaleString('en-IN')}</span>
                <span style={{ color: '#10b981' }}>₹{high?.toLocaleString('en-IN')}</span>
            </div>
            <div style={{
                height: 8,
                background: 'rgba(255,255,255,0.1)',
                borderRadius: 4,
                position: 'relative',
                overflow: 'visible',
            }}>
                <div style={{
                    height: '100%',
                    width: `${pct}%`,
                    background: 'linear-gradient(90deg, #ef4444, #f59e0b, #10b981)',
                    borderRadius: 4,
                }} />
                <div style={{
                    position: 'absolute',
                    top: -4,
                    left: `${pct}%`,
                    transform: 'translateX(-50%)',
                    width: 16,
                    height: 16,
                    background: '#3b82f6',
                    borderRadius: '50%',
                    border: '2px solid #0a0e1a',
                    boxShadow: '0 0 10px rgba(59,130,246,0.5)',
                }} />
            </div>
            <div style={{ textAlign: 'center', marginTop: 8, fontSize: 12, color: '#94a3b8' }}>
                Current: <b style={{ color: '#3b82f6' }}>₹{current?.toLocaleString('en-IN')}</b>
                {' '}({pct.toFixed(0)}% from low)
            </div>
        </div>
    )
}


export default function TopPicks() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        const fetchTopPicks = async () => {
            try {
                setLoading(true)
                const res = await axios.get(`${API_BASE}/top-picks`)
                if (res.data.success) {
                    setData(res.data.data)
                }
            } catch (err) {
                setError('Failed to load top picks')
            } finally {
                setLoading(false)
            }
        }
        fetchTopPicks()
    }, [])

    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: 60 }}>
                <div className="loading-spinner" style={{ margin: '0 auto 16px' }} />
                <div style={{ color: '#94a3b8' }}>Analyzing top picks & fetching company insights...</div>
                <div style={{ color: '#64748b', fontSize: 13, marginTop: 6 }}>This may take a moment</div>
            </div>
        )
    }

    if (error || !data) {
        return (
            <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                {error || 'No data available'}
            </div>
        )
    }

    return (
        <div>
            {/* Top BUY Picks */}
            {data.buy?.length > 0 && (
                <div style={{ marginBottom: 32 }}>
                    <h2 style={{
                        fontSize: 20,
                        fontWeight: 800,
                        marginBottom: 16,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                    }}>
                        <span style={{
                            background: 'rgba(16,185,129,0.15)',
                            color: '#10b981',
                            padding: '4px 12px',
                            borderRadius: 8,
                            fontSize: 14,
                        }}>🟢 TOP BUY</span>
                        Stocks Recommended to Buy
                    </h2>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                        {data.buy.map((item, idx) => (
                            <InsightCard key={idx} item={item} type="buy" />
                        ))}
                    </div>
                </div>
            )}

            {/* Top SELL Picks */}
            {data.sell?.length > 0 && (
                <div>
                    <h2 style={{
                        fontSize: 20,
                        fontWeight: 800,
                        marginBottom: 16,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 10,
                    }}>
                        <span style={{
                            background: 'rgba(239,68,68,0.15)',
                            color: '#ef4444',
                            padding: '4px 12px',
                            borderRadius: 8,
                            fontSize: 14,
                        }}>🔴 TOP SELL</span>
                        Stocks Recommended to Sell / Avoid
                    </h2>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                        {data.sell.map((item, idx) => (
                            <InsightCard key={idx} item={item} type="sell" />
                        ))}
                    </div>
                </div>
            )}

            {data.buy?.length === 0 && data.sell?.length === 0 && (
                <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>
                    No strong BUY or SELL signals at this time
                </div>
            )}
        </div>
    )
}
