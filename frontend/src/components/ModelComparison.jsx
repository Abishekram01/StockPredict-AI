import { useState } from 'react'

export default function ModelComparison({ modelMetrics, testMetrics, bestModel, bestModelOnTest }) {
    const [viewMode, setViewMode] = useState('test') // 'val' or 'test'

    const metrics = viewMode === 'test' && testMetrics && Object.keys(testMetrics).length > 0
        ? testMetrics
        : modelMetrics

    const currentBest = viewMode === 'test' ? (bestModelOnTest || bestModel) : bestModel

    if (!metrics || Object.keys(metrics).length === 0) {
        return (
            <div style={{ textAlign: 'center', padding: 30, color: '#64748b' }}>
                No model metrics available
            </div>
        )
    }

    const getMetricClass = (value) => {
        if (value >= 0.7) return 'metric-high'
        if (value >= 0.5) return 'metric-mid'
        return 'metric-low'
    }

    const models = Object.entries(metrics)

    return (
        <div>
            {/* Toggle between Validation and Test results */}
            <div style={{
                display: 'flex',
                gap: 6,
                marginBottom: 16,
                background: 'rgba(17, 24, 39, 0.8)',
                padding: 4,
                borderRadius: 10,
                border: '1px solid rgba(255,255,255,0.08)',
                width: 'fit-content',
            }}>
                <button
                    onClick={() => setViewMode('val')}
                    style={{
                        padding: '8px 18px',
                        border: 'none',
                        borderRadius: 8,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        background: viewMode === 'val'
                            ? 'linear-gradient(135deg, #667eea, #764ba2)'
                            : 'transparent',
                        color: viewMode === 'val' ? 'white' : '#94a3b8',
                    }}
                >
                    Validation (20%)
                </button>
                <button
                    onClick={() => setViewMode('test')}
                    style={{
                        padding: '8px 18px',
                        border: 'none',
                        borderRadius: 8,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        background: viewMode === 'test'
                            ? 'linear-gradient(135deg, #f59e0b, #ef4444)'
                            : 'transparent',
                        color: viewMode === 'test' ? 'white' : '#94a3b8',
                    }}
                >
                    🧪 Held-Out Test (20%)
                </button>
            </div>

            <div style={{
                fontSize: 12,
                color: '#64748b',
                marginBottom: 12,
                lineHeight: 1.5,
            }}>
                {viewMode === 'test'
                    ? '⚠️ These metrics come from the held-out test set (last 20% of data) — completely unseen during training.'
                    : 'These metrics come from the validation set (60-80% of data) — used to select the best model.'}
            </div>

            <table className="model-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>ROC AUC</th>
                    </tr>
                </thead>
                <tbody>
                    {models.map(([name, m]) => (
                        <tr key={name} className={name === currentBest ? 'best-model' : ''}>
                            <td style={{ fontWeight: name === currentBest ? 700 : 400 }}>
                                {name === currentBest ? '★ ' : ''}{name}
                            </td>
                            <td className={getMetricClass(m.accuracy)}>
                                {(m.accuracy * 100).toFixed(1)}%
                            </td>
                            <td className={getMetricClass(m.precision)}>
                                {(m.precision * 100).toFixed(1)}%
                            </td>
                            <td className={getMetricClass(m.recall)}>
                                {(m.recall * 100).toFixed(1)}%
                            </td>
                            <td className={getMetricClass(m.f1)}>
                                {(m.f1 * 100).toFixed(1)}%
                            </td>
                            <td className={getMetricClass(m.roc_auc)}>
                                {(m.roc_auc * 100).toFixed(1)}%
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}
