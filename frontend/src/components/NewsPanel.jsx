export default function NewsPanel({ news }) {
    if (!news || news.length === 0) {
        return (
            <div style={{ textAlign: 'center', padding: 30, color: '#64748b' }}>
                No news available for this stock
            </div>
        )
    }

    return (
        <div className="news-list">
            {news.map((item, idx) => (
                <div key={idx} className="news-item">
                    <div className={`news-sentiment-dot ${item.sentiment_label}`} />
                    <div className="news-content">
                        <div className="headline">
                            <a href={item.url} target="_blank" rel="noopener noreferrer">
                                {item.headline}
                            </a>
                        </div>
                        <div className="news-meta">
                            <span className={`score ${item.sentiment_label}`}>
                                {item.sentiment_score > 0 ? '+' : ''}{item.sentiment_score?.toFixed(2)}
                            </span>
                            <span>{item.source}</span>
                            <span>{new Date(item.published_at).toLocaleDateString('en-IN', {
                                day: 'numeric', month: 'short', year: 'numeric'
                            })}</span>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    )
}
