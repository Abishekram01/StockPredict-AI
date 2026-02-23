import { useState, useEffect } from 'react'
import axios from 'axios'
import StockCard from './StockCard'
import PredictionChart from './PredictionChart'
import NewsPanel from './NewsPanel'
import ModelComparison from './ModelComparison'
import TopPicks from './TopPicks'
import CustomSearch from './CustomSearch'
import SentimentHeatmap from './SentimentHeatmap'

const API_BASE = 'http://localhost:5000/api'

export default function Dashboard() {
    const [stocks, setStocks] = useState([])
    const [summary, setSummary] = useState({ total: 0, buy: 0, hold: 0, sell: 0 })
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [refreshing, setRefreshing] = useState(false)
    const [selectedStock, setSelectedStock] = useState(null)
    const [stockDetail, setStockDetail] = useState(null)
    const [detailLoading, setDetailLoading] = useState(false)
    const [activeTab, setActiveTab] = useState('top-picks')
    const [searchQuery, setSearchQuery] = useState('')

    // Fetch all stocks
    const fetchStocks = async () => {
        try {
            setLoading(true)
            setError(null)
            const res = await axios.get(`${API_BASE}/stocks`)
            if (res.data.success) {
                setStocks(res.data.stocks)
                setSummary(res.data.summary)
            }
        } catch (err) {
            setError('Failed to connect to the API server. Make sure the backend is running on port 5000.')
        } finally {
            setLoading(false)
        }
    }

    // Fetch stock detail
    const fetchStockDetail = async (ticker) => {
        try {
            setDetailLoading(true)
            const res = await axios.get(`${API_BASE}/stocks/${ticker}`)
            if (res.data.success) {
                setStockDetail(res.data.data)
            }
        } catch (err) {
            console.error('Error fetching stock details:', err)
        } finally {
            setDetailLoading(false)
        }
    }

    // Refresh data
    const handleRefresh = async () => {
        try {
            setRefreshing(true)
            await axios.post(`${API_BASE}/refresh`)
            await fetchStocks()
        } catch (err) {
            setError('Failed to refresh data')
        } finally {
            setRefreshing(false)
        }
    }

    // Handle stock card click
    const handleStockClick = (ticker) => {
        setSelectedStock(ticker)
        fetchStockDetail(ticker)
    }

    // Close modal
    const closeModal = () => {
        setSelectedStock(null)
        setStockDetail(null)
    }

    useEffect(() => {
        fetchStocks()
    }, [])

    // Filter stocks
    const filteredStocks = stocks.filter(stock => {
        const matchesTab = activeTab === 'all' || stock.signal.toLowerCase() === activeTab
        const matchesSearch = searchQuery === '' ||
            stock.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
            stock.company.toLowerCase().includes(searchQuery.toLowerCase())
        return matchesTab && matchesSearch
    })

    if (loading) {
        return (
            <div className="loading-screen">
                <div className="loading-spinner" />
                <div className="loading-text">Loading predictions...</div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="error-state">
                <div className="icon">⚠️</div>
                <h2>Connection Error</h2>
                <p>{error}</p>
                <button className="retry-btn" onClick={fetchStocks}>Retry</button>
            </div>
        )
    }

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <div className="header-left">
                    <h1>📊 StockPredict AI</h1>
                    <p>NIFTY 50 · ML-Powered Predictions · Live News Sentiment</p>
                </div>
                <div className="header-actions">
                    <button
                        className={`btn-refresh ${refreshing ? 'spinning' : ''}`}
                        onClick={handleRefresh}
                        disabled={refreshing}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 12a9 9 0 11-6.219-8.56" />
                            <polyline points="21 3 21 12 12 12" />
                        </svg>
                        {refreshing ? 'Refreshing...' : 'Refresh Data'}
                    </button>
                </div>
            </header>

            {/* Summary Bar */}
            <div className="summary-bar">
                <div className="summary-card total">
                    <div className="label">Total Stocks</div>
                    <div className="value">{summary.total}</div>
                </div>
                <div className="summary-card buy">
                    <div className="label">Buy Signals</div>
                    <div className="value">{summary.buy}</div>
                </div>
                <div className="summary-card hold">
                    <div className="label">Hold Signals</div>
                    <div className="value">{summary.hold}</div>
                </div>
                <div className="summary-card sell">
                    <div className="label">Sell Signals</div>
                    <div className="value">{summary.sell}</div>
                </div>
            </div>

            {/* Tabs + Filters */}
            <div className="tabs">
                {['top-picks', 'all', 'buy', 'hold', 'sell', 'heatmap', 'search'].map(tab => (
                    <button
                        key={tab}
                        className={`tab ${activeTab === tab ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab)}
                    >
                        {tab === 'top-picks' ? `🏆 Top Picks` :
                            tab === 'heatmap' ? `🌡️ Heatmap` :
                                tab === 'search' ? `🔍 Search Stock` :
                                    tab === 'all' ? `All (${summary.total})` :
                                        tab === 'buy' ? `🟢 Buy (${summary.buy})` :
                                            tab === 'hold' ? `🟡 Hold (${summary.hold})` :
                                                `🔴 Sell (${summary.sell})`}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'top-picks' ? (
                <TopPicks />
            ) : activeTab === 'heatmap' ? (
                <SentimentHeatmap />
            ) : activeTab === 'search' ? (
                <CustomSearch />
            ) : (
                <>
                    <div className="filter-bar">
                        <input
                            className="search-input"
                            type="text"
                            placeholder="🔍  Search by ticker or company name..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>

                    {/* Stock Grid */}
                    <div className="stock-grid">
                        {filteredStocks.map(stock => (
                            <StockCard
                                key={stock.ticker}
                                stock={stock}
                                onClick={handleStockClick}
                            />
                        ))}
                    </div>

                    {filteredStocks.length === 0 && (
                        <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>
                            <div style={{ fontSize: 40, marginBottom: 12 }}>🔍</div>
                            <p>No stocks match your filters</p>
                        </div>
                    )}
                </>
            )}

            {/* Stock Detail Modal */}
            {selectedStock && (
                <div className="modal-overlay" onClick={(e) => {
                    if (e.target === e.currentTarget) closeModal()
                }}>
                    <div className="modal-content">
                        <div className="modal-header">
                            <h2>
                                {stockDetail?.prediction?.company || selectedStock.replace('.NS', '')}
                                <span style={{ color: '#64748b', fontWeight: 400, fontSize: 16, marginLeft: 10 }}>
                                    {selectedStock.replace('.NS', '')}
                                </span>
                            </h2>
                            <button className="close-btn" onClick={closeModal}>✕</button>
                        </div>

                        <div className="modal-body">
                            {detailLoading ? (
                                <div className="loading-screen" style={{ minHeight: '30vh' }}>
                                    <div className="loading-spinner" />
                                </div>
                            ) : stockDetail ? (
                                <>
                                    {/* Chart */}
                                    <div className="modal-section">
                                        <h3>Price History (90 Days)</h3>
                                        <PredictionChart
                                            history={stockDetail.history}
                                            prediction={stockDetail.prediction}
                                        />
                                    </div>

                                    {/* Model Comparison */}
                                    <div className="modal-section">
                                        <h3>Model Comparison</h3>
                                        <ModelComparison
                                            modelMetrics={stockDetail.model_metrics}
                                            testMetrics={stockDetail.test_metrics}
                                            bestModel={stockDetail.prediction?.best_model}
                                            bestModelOnTest={stockDetail.best_model_on_test}
                                        />
                                    </div>

                                    {/* News */}
                                    <div className="modal-section">
                                        <h3>Latest News & Sentiment</h3>
                                        <NewsPanel news={stockDetail.news} />
                                    </div>
                                </>
                            ) : (
                                <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                                    Failed to load details
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
