import { useState } from 'react'

export default function StockCard({ stock, onClick }) {
  const signalClass = stock.signal.toLowerCase()
  const priceChangeClass = stock.price_change_pct >= 0 ? 'positive' : 'negative'
  const priceChangeSign = stock.price_change_pct >= 0 ? '+' : ''

  return (
    <div
      className={`stock-card ${signalClass}-signal`}
      onClick={() => onClick(stock.ticker)}
    >
      <div className="stock-card-header">
        <div>
          <div className="ticker">{stock.ticker.replace('.NS', '')}</div>
          <div className="company">{stock.company}</div>
        </div>
        <span className={`signal-badge ${signalClass}`}>
          {stock.signal}
        </span>
      </div>

      <div className="stock-card-body">
        <div className="price-info">
          <div className="price">₹{stock.current_price?.toLocaleString('en-IN')}</div>
          <div className={`change ${priceChangeClass}`}>
            {priceChangeSign}{stock.price_change_pct}%
          </div>
        </div>

        <div className="confidence-info">
          <div className="label">Confidence</div>
          <div className="confidence-bar">
            <div
              className={`fill ${signalClass}`}
              style={{ width: `${stock.confidence}%` }}
            />
          </div>
          <div className="confidence-value">{stock.confidence}%</div>
        </div>
      </div>

      <div className="model-tag">
        <span className="dot" />
        {stock.best_model} · AUC {stock.best_auc?.toFixed(2)}
      </div>
    </div>
  )
}
