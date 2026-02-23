import { useState } from 'react'
import axios from 'axios'
import PredictionChart from './PredictionChart'
import ModelComparison from './ModelComparison'
import NewsPanel from './NewsPanel'

const API_BASE = 'http://localhost:5000/api'

const PROGRESS_STEPS = [
    'Scraping stock data...',
    'Fetching news & sentiment...',
    'Engineering features...',
    'Training 6 ML models...',
    'Generating prediction...',
    'Fetching company insights...',
    'Analysis complete!',
]

function MetricBadge({ label, value }) {
    return (
        <div style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 8,
            padding: '10px 12px',
        }}>
            <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
            <div style={{ fontSize: 15, fontWeight: 700, marginTop: 4, color: '#f1f5f9' }}>{value ?? 'N/A'}</div>
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
            <div style={{ height: 8, background: 'rgba(255,255,255,0.1)', borderRadius: 4, position: 'relative' }}>
                <div style={{ height: '100%', width: `${pct}%`, background: 'linear-gradient(90deg, #ef4444, #f59e0b, #10b981)', borderRadius: 4 }} />
                <div style={{
                    position: 'absolute', top: -4, left: `${pct}%`, transform: 'translateX(-50%)',
                    width: 16, height: 16, background: '#3b82f6', borderRadius: '50%',
                    border: '2px solid #0a0e1a', boxShadow: '0 0 10px rgba(59,130,246,0.5)',
                }} />
            </div>
            <div style={{ textAlign: 'center', marginTop: 8, fontSize: 12, color: '#94a3b8' }}>
                Current: <b style={{ color: '#3b82f6' }}>₹{current?.toLocaleString('en-IN')}</b> ({pct.toFixed(0)}% from low)
            </div>
        </div>
    )
}


export default function CustomSearch() {
    const [ticker, setTicker] = useState('')
    const [companyName, setCompanyName] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [result, setResult] = useState(null)
    const [progress, setProgress] = useState(0)
    const [showDetails, setShowDetails] = useState(false)

    const handleAnalyze = async () => {
        if (!ticker.trim()) return

        setLoading(true)
        setError(null)
        setResult(null)
        setProgress(0)

        // Simulate progress
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= PROGRESS_STEPS.length - 1) {
                    clearInterval(interval)
                    return prev
                }
                return prev + 1
            })
        }, 3000)

        try {
            const res = await axios.post(`${API_BASE}/analyze`, {
                ticker: ticker.trim(),
                company_name: companyName.trim() || null,
            }, { timeout: 300000 }) // 5 min timeout

            clearInterval(interval)
            setProgress(PROGRESS_STEPS.length - 1)

            if (res.data.success) {
                setResult(res.data.data)
            } else {
                setError(res.data.error || 'Analysis failed')
            }
        } catch (err) {
            clearInterval(interval)
            setError(err.response?.data?.error || 'Failed to analyze stock. Check the ticker symbol.')
        } finally {
            setLoading(false)
        }
    }

    const pred = result?.prediction
    const ins = result?.insights
    const signalColor = pred?.signal === 'BUY' ? '#10b981' : pred?.signal === 'SELL' ? '#ef4444' : '#f59e0b'
    const signalGlow = pred?.signal === 'BUY' ? 'rgba(16,185,129,0.15)' : pred?.signal === 'SELL' ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)'

    return (
        <div>
            {/* Search Input */}
            <div style={{
                background: 'rgba(17, 24, 39, 0.8)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 16,
                padding: 28,
                marginBottom: 24,
                boxShadow: '0 4px 30px rgba(0,0,0,0.4)',
            }}>
                <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 6 }}>
                    🔍 Analyze Any Stock
                </h2>
                <p style={{ color: '#94a3b8', fontSize: 14, marginBottom: 20 }}>
                    Enter any NSE/BSE ticker — we'll scrape data, train 6 ML models, evaluate on a held-out test set, and give you a prediction with full company insights.
                </p>

                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    <input
                        type="text"
                        placeholder="Ticker (e.g. ZOMATO, PAYTM, IRCTC)"
                        value={ticker}
                        onChange={e => setTicker(e.target.value.toUpperCase())}
                        onKeyDown={e => e.key === 'Enter' && !loading && handleAnalyze()}
                        disabled={loading}
                        style={{
                            flex: '1 1 200px',
                            padding: '12px 16px',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: 10,
                            color: '#f1f5f9',
                            fontSize: 15,
                            fontWeight: 600,
                            outline: 'none',
                            transition: 'all 0.3s ease',
                        }}
                    />
                    <input
                        type="text"
                        placeholder="Company name (optional)"
                        value={companyName}
                        onChange={e => setCompanyName(e.target.value)}
                        disabled={loading}
                        style={{
                            flex: '1 1 200px',
                            padding: '12px 16px',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: 10,
                            color: '#f1f5f9',
                            fontSize: 14,
                            outline: 'none',
                        }}
                    />
                    <button
                        onClick={handleAnalyze}
                        disabled={loading || !ticker.trim()}
                        style={{
                            padding: '12px 28px',
                            background: loading ? 'rgba(255,255,255,0.1)' : 'linear-gradient(135deg, #667eea, #764ba2)',
                            color: 'white',
                            border: 'none',
                            borderRadius: 10,
                            fontSize: 14,
                            fontWeight: 700,
                            cursor: loading ? 'not-allowed' : 'pointer',
                            transition: 'all 0.3s ease',
                            whiteSpace: 'nowrap',
                        }}
                    >
                        {loading ? '⏳ Analyzing...' : '🚀 Analyze'}
                    </button>
                </div>

                <div style={{ marginTop: 10, fontSize: 12, color: '#64748b' }}>
                    💡 Use NSE tickers like ZOMATO, PAYTM, IRCTC, DMART, PNB, NYKAA — the <code>.NS</code> suffix is added automatically.
                </div>
            </div>

            {/* Loading Progress */}
            {loading && (
                <div style={{
                    background: 'rgba(17, 24, 39, 0.8)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 16,
                    padding: 32,
                    marginBottom: 24,
                    textAlign: 'center',
                }}>
                    <div className="loading-spinner" style={{ margin: '0 auto 20px' }} />
                    <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>
                        Training ML models for {ticker}...
                    </div>
                    <div style={{ maxWidth: 400, margin: '0 auto' }}>
                        {PROGRESS_STEPS.map((step, idx) => (
                            <div key={idx} style={{
                                display: 'flex', alignItems: 'center', gap: 10, padding: '6px 0',
                                opacity: idx <= progress ? 1 : 0.3,
                                transition: 'opacity 0.4s ease',
                            }}>
                                <span style={{
                                    width: 20, height: 20, borderRadius: '50%', display: 'flex',
                                    alignItems: 'center', justifyContent: 'center', fontSize: 11,
                                    background: idx < progress ? '#10b981' : idx === progress ? '#3b82f6' : 'rgba(255,255,255,0.1)',
                                    color: 'white', fontWeight: 700, flexShrink: 0,
                                }}>
                                    {idx < progress ? '✓' : idx + 1}
                                </span>
                                <span style={{ fontSize: 13, color: idx <= progress ? '#f1f5f9' : '#64748b' }}>{step}</span>
                            </div>
                        ))}
                    </div>
                    <div style={{ marginTop: 16, fontSize: 12, color: '#64748b' }}>
                        This may take 1-3 minutes depending on the stock
                    </div>
                </div>
            )}

            {/* Error */}
            {error && (
                <div style={{
                    background: 'rgba(239,68,68,0.1)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: 12,
                    padding: 20,
                    marginBottom: 24,
                    color: '#fca5a5',
                    fontSize: 14,
                }}>
                    ❌ {error}
                </div>
            )}

            {/* Results */}
            {result && (
                <div>
                    {/* Prediction Signal Hero */}
                    <div style={{
                        background: 'rgba(17, 24, 39, 0.8)',
                        border: `1px solid rgba(255,255,255,0.08)`,
                        borderLeft: `4px solid ${signalColor}`,
                        borderRadius: 16,
                        padding: 28,
                        marginBottom: 20,
                        boxShadow: '0 4px 30px rgba(0,0,0,0.4)',
                    }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
                            <div>
                                <div style={{ fontSize: 24, fontWeight: 800 }}>
                                    {pred?.signal === 'BUY' ? '🟢' : pred?.signal === 'SELL' ? '🔴' : '🟡'} {pred?.company || pred?.ticker}
                                </div>
                                <div style={{ color: '#94a3b8', fontSize: 14, marginTop: 2 }}>
                                    {pred?.ticker} · {ins?.profile?.sector || 'N/A'} · {ins?.profile?.industry || 'N/A'}
                                </div>
                                <div style={{ marginTop: 12, fontSize: 13, color: '#94a3b8' }}>
                                    Split: {result.split_info?.train} train / {result.split_info?.val} val / {result.split_info?.test} test
                                    {' '}· Best model: <b style={{ color: '#f1f5f9' }}>{pred?.best_model}</b>
                                </div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <span style={{
                                    padding: '8px 20px', borderRadius: 20, fontSize: 16, fontWeight: 800,
                                    background: signalGlow, color: signalColor,
                                    boxShadow: `0 0 20px ${signalGlow}`,
                                }}>
                                    {pred?.signal}
                                </span>
                                <div style={{ fontSize: 28, fontWeight: 700, marginTop: 10 }}>
                                    ₹{pred?.current_price?.toLocaleString('en-IN')}
                                </div>
                                <div style={{ fontSize: 14, fontWeight: 600, color: pred?.price_change_pct >= 0 ? '#10b981' : '#ef4444' }}>
                                    {pred?.price_change_pct >= 0 ? '+' : ''}{pred?.price_change_pct}%
                                </div>
                                <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
                                    Confidence: <b style={{ color: signalColor }}>{pred?.confidence}%</b>
                                    {' '}· Val AUC: {pred?.best_val_auc} · Test AUC: {pred?.best_test_auc}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Insights Metrics */}
                    {ins && (
                        <div style={{
                            background: 'rgba(17, 24, 39, 0.8)',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: 16,
                            padding: 24,
                            marginBottom: 20,
                        }}>
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

                            {/* 52-week range */}
                            {ins.price_info && (
                                <div style={{ marginBottom: 16 }}>
                                    <div style={{ fontSize: 12, color: '#64748b', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
                                        52-Week Range
                                    </div>
                                    <PriceRange
                                        low={ins.price_info.fifty_two_week_low}
                                        high={ins.price_info.fifty_two_week_high}
                                        current={pred?.current_price}
                                    />
                                </div>
                            )}

                            {/* Analyst */}
                            {ins.analyst?.target_mean > 0 && (
                                <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginTop: 16 }}>
                                    <Stat label="Analyst Rating" value={ins.analyst?.recommendation?.toUpperCase()} color={
                                        ['buy', 'strong_buy'].includes(ins.analyst?.recommendation) ? '#10b981' :
                                            ins.analyst?.recommendation === 'sell' ? '#ef4444' : '#f59e0b'
                                    } />
                                    <Stat label="Target Mean" value={`₹${ins.analyst?.target_mean?.toLocaleString('en-IN')}`} color="#3b82f6" />
                                    <Stat label="Target High" value={`₹${ins.analyst?.target_high?.toLocaleString('en-IN')}`} />
                                    {pred?.current_price > 0 && ins.analyst?.target_mean > 0 && (
                                        <Stat
                                            label="Upside"
                                            value={`${(((ins.analyst.target_mean - pred.current_price) / pred.current_price) * 100).toFixed(1)}%`}
                                            color={ins.analyst.target_mean > pred.current_price ? '#10b981' : '#ef4444'}
                                        />
                                    )}
                                    <Stat label="News Sentiment" value={ins.news_summary?.sentiment_verdict} color={
                                        ins.news_summary?.sentiment_verdict === 'Bullish' ? '#10b981' :
                                            ins.news_summary?.sentiment_verdict === 'Bearish' ? '#ef4444' : '#f59e0b'
                                    } />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Chart */}
                    <div style={{
                        background: 'rgba(17, 24, 39, 0.8)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: 16,
                        padding: 24,
                        marginBottom: 20,
                    }}>
                        <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
                            Price History (90 Days)
                        </div>
                        <PredictionChart history={result.history} prediction={pred} />
                    </div>

                    {/* Model Comparison */}
                    <div style={{
                        background: 'rgba(17, 24, 39, 0.8)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: 16,
                        padding: 24,
                        marginBottom: 20,
                    }}>
                        <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
                            Model Comparison
                        </div>
                        <ModelComparison
                            modelMetrics={result.val_metrics}
                            testMetrics={result.test_metrics}
                            bestModel={pred?.best_model}
                        />
                    </div>

                    {/* Expandable: Company Profile + News */}
                    <button
                        onClick={() => setShowDetails(!showDetails)}
                        style={{
                            width: '100%', padding: '12px 0',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: 12, color: '#94a3b8',
                            fontSize: 14, fontWeight: 600, cursor: 'pointer',
                            marginBottom: 20,
                        }}
                    >
                        {showDetails ? '▲ Hide Company Profile & News' : '▼ Show Company Profile & News'}
                    </button>

                    {showDetails && (
                        <div style={{
                            background: 'rgba(17, 24, 39, 0.8)',
                            border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: 16,
                            padding: 24,
                        }}>
                            {/* Description */}
                            {ins?.profile?.description && ins.profile.description !== 'N/A' && (
                                <div style={{ marginBottom: 24 }}>
                                    <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
                                        About {ins.profile.name}
                                    </div>
                                    <p style={{ fontSize: 14, color: '#cbd5e1', lineHeight: 1.7 }}>{ins.profile.description}</p>
                                    <div style={{ display: 'flex', gap: 16, marginTop: 10, fontSize: 13, color: '#94a3b8' }}>
                                        {ins.profile?.employees > 0 && <span>👥 {ins.profile.employees.toLocaleString()} employees</span>}
                                        {ins.profile?.website && ins.profile.website !== 'N/A' && (
                                            <a href={ins.profile.website} target="_blank" rel="noopener" style={{ color: '#3b82f6' }}>🌐 Website</a>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Detailed Financials */}
                            {ins && (
                                <div style={{ marginBottom: 24 }}>
                                    <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                                        Detailed Financials
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 8 }}>
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
                            )}

                            {/* Company News */}
                            <div style={{ marginBottom: 24 }}>
                                <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                                    📰 Company News
                                </div>
                                <NewsPanel news={result.news} />
                            </div>

                            {/* Industry / Domain News */}
                            {result.industry_news?.length > 0 && (
                                <div>
                                    <div style={{ fontSize: 13, color: '#64748b', fontWeight: 600, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
                                        🏭 Industry & Domain News
                                    </div>
                                    <div style={{ fontSize: 12, color: '#475569', marginBottom: 12 }}>
                                        {result.sector && <span>Sector: <b style={{ color: '#94a3b8' }}>{result.sector}</b></span>}
                                        {result.industry && <span> · Industry: <b style={{ color: '#94a3b8' }}>{result.industry}</b></span>}
                                        {' '}— macro-level news that may affect this stock
                                    </div>
                                    <NewsPanel news={result.industry_news} />
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
