"""Web dashboard for monitoring trading strategies and performance."""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import json
import os
from datetime import datetime, timedelta


class DashboardServer:
    """Web dashboard server for trading monitoring."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False
    ):
        """Initialize dashboard server.
        
        Args:
            host: Server host
            port: Server port
            debug: Debug mode
        """
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.host = host
        self.port = port
        self.debug = debug
        
        # Data storage
        self.performance_data: List[Dict] = []
        self.positions_data: Dict[str, float] = {}
        self.orders_data: List[Dict] = []
        self.risk_metrics: Dict = {}
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get performance data."""
            return jsonify(self.performance_data)
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions."""
            return jsonify(self.positions_data)
        
        @self.app.route('/api/orders')
        def get_orders():
            """Get order history."""
            limit = request.args.get('limit', 100, type=int)
            return jsonify(self.orders_data[-limit:])
        
        @self.app.route('/api/risk')
        def get_risk_metrics():
            """Get risk metrics."""
            return jsonify(self.risk_metrics)
        
        @self.app.route('/api/summary')
        def get_summary():
            """Get portfolio summary."""
            if not self.performance_data:
                return jsonify({})
            
            latest = self.performance_data[-1]
            
            summary = {
                "portfolio_value": latest.get('portfolio_value', 0),
                "cash": latest.get('cash', 0),
                "pnl": latest.get('pnl', 0),
                "return_pct": latest.get('return_pct', 0),
                "num_positions": len([p for p in self.positions_data.values() if p != 0]),
                "num_orders": len(self.orders_data),
                "last_update": latest.get('timestamp', '')
            }
            
            return jsonify(summary)
        
        @self.app.route('/api/update', methods=['POST'])
        def update_data():
            """Update dashboard data."""
            data = request.json
            
            if 'performance' in data:
                self.performance_data.append(data['performance'])
            
            if 'positions' in data:
                self.positions_data = data['positions']
            
            if 'orders' in data:
                self.orders_data.extend(data['orders'])
            
            if 'risk' in data:
                self.risk_metrics = data['risk']
            
            return jsonify({"status": "success"})
    
    def run(self) -> None:
        """Start the dashboard server."""
        print(f"Starting dashboard server at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)
    
    def load_data(self, filepath: str) -> None:
        """Load data from file.
        
        Args:
            filepath: Path to JSON data file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'performance_history' in data:
            self.performance_data = data['performance_history']
        
        if 'summary' in data and 'positions' in data['summary']:
            self.positions_data = data['summary']['positions']
        
        if 'trades' in data:
            self.orders_data = data['trades']


def create_dashboard_html() -> str:
    """Create HTML template for dashboard.
    
    Returns:
        HTML template string
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f1419;
            color: #ffffff;
            padding: 20px;
        }
        
        .header {
            background: #1a1f2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #1a1f2e;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .metric-label {
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
        }
        
        .metric-value.positive {
            color: #00ff88;
        }
        
        .metric-value.negative {
            color: #ff4444;
        }
        
        .chart-container {
            background: #1a1f2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
            background: #1a1f2e;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .positions-table th {
            background: #252d3d;
            padding: 12px;
            text-align: left;
            color: #8b949e;
            font-weight: 600;
        }
        
        .positions-table td {
            padding: 12px;
            border-top: 1px solid #2d3748;
        }
        
        .positions-table tr:hover {
            background: #252d3d;
        }
        
        .refresh-btn {
            background: #00d4ff;
            color: #0f1419;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        .refresh-btn:hover {
            background: #00b8e6;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Dashboard</h1>
        <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
    </div>
    
    <div class="summary-grid" id="summary">
        <!-- Summary metrics will be inserted here -->
    </div>
    
    <div class="chart-container">
        <h2>Portfolio Performance</h2>
        <div id="performance-chart"></div>
    </div>
    
    <div class="chart-container">
        <h2>Current Positions</h2>
        <table class="positions-table" id="positions-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Quantity</th>
                    <th>Price</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
    
    <script>
        async function fetchData() {
            try {
                const [summary, performance, positions] = await Promise.all([
                    fetch('/api/summary').then(r => r.json()),
                    fetch('/api/performance').then(r => r.json()),
                    fetch('/api/positions').then(r => r.json())
                ]);
                
                return { summary, performance, positions };
            } catch (error) {
                console.error('Error fetching data:', error);
                return null;
            }
        }
        
        function updateSummary(summary) {
            const summaryDiv = document.getElementById('summary');
            
            const pnlClass = summary.pnl >= 0 ? 'positive' : 'negative';
            const returnClass = summary.return_pct >= 0 ? 'positive' : 'negative';
            
            summaryDiv.innerHTML = `
                <div class="metric-card">
                    <div class="metric-label">Portfolio Value</div>
                    <div class="metric-value">$${summary.portfolio_value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cash</div>
                    <div class="metric-value">$${summary.cash.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">P&L</div>
                    <div class="metric-value ${pnlClass}">$${summary.pnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Return</div>
                    <div class="metric-value ${returnClass}">${summary.return_pct.toFixed(2)}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Positions</div>
                    <div class="metric-value">${summary.num_positions}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Orders</div>
                    <div class="metric-value">${summary.num_orders}</div>
                </div>
            `;
        }
        
        function updatePerformanceChart(performance) {
            const timestamps = performance.map(p => p.timestamp);
            const values = performance.map(p => p.portfolio_value);
            
            const trace = {
                x: timestamps,
                y: values,
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#00d4ff',
                    width: 2
                },
                fill: 'tozeroy',
                fillcolor: 'rgba(0, 212, 255, 0.1)'
            };
            
            const layout = {
                paper_bgcolor: '#1a1f2e',
                plot_bgcolor: '#1a1f2e',
                font: { color: '#ffffff' },
                xaxis: {
                    gridcolor: '#2d3748',
                    title: 'Time'
                },
                yaxis: {
                    gridcolor: '#2d3748',
                    title: 'Portfolio Value ($)'
                },
                margin: { t: 20, r: 20, b: 40, l: 60 }
            };
            
            Plotly.newPlot('performance-chart', [trace], layout, {responsive: true});
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.querySelector('#positions-table tbody');
            tbody.innerHTML = '';
            
            for (const [symbol, quantity] of Object.entries(positions)) {
                if (quantity !== 0) {
                    const row = tbody.insertRow();
                    row.innerHTML = `
                        <td>${symbol}</td>
                        <td>${quantity.toFixed(2)}</td>
                        <td>-</td>
                        <td>-</td>
                    `;
                }
            }
        }
        
        async function refreshData() {
            const data = await fetchData();
            if (data) {
                updateSummary(data.summary);
                updatePerformanceChart(data.performance);
                updatePositionsTable(data.positions);
            }
        }
        
        // Initial load
        refreshData();
        
        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
"""


def setup_dashboard_files():
    """Create necessary dashboard files."""
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create dashboard.html
    html_path = os.path.join(templates_dir, 'dashboard.html')
    with open(html_path, 'w') as f:
        f.write(create_dashboard_html())
    
    print(f"Dashboard template created at {html_path}")


if __name__ == "__main__":
    setup_dashboard_files()
    
    # Start dashboard server
    server = DashboardServer(debug=True)
    server.run()
