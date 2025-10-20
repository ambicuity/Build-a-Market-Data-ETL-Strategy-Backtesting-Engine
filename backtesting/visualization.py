"""Visualization utilities for backtesting results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
from pathlib import Path


class Visualizer:
    """Create visualizations for backtesting results."""
    
    def __init__(self, results: pd.DataFrame):
        """Initialize visualizer.
        
        Args:
            results: DataFrame with backtest results
        """
        self.results = results
    
    def plot_equity_curve(
        self,
        figsize: tuple = (12, 6),
        title: str = "Equity Curve",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot equity curve using matplotlib.
        
        Args:
            figsize: Figure size
            title: Plot title
            show: Whether to show the plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if "equity" in self.results.columns:
            ax.plot(self.results.index, self.results["equity"], linewidth=2)
            ax.set_ylabel("Portfolio Value ($)")
        else:
            cumulative_returns = (1 + self.results["net_returns"]).cumprod()
            ax.plot(self.results.index, cumulative_returns, linewidth=2)
            ax.set_ylabel("Cumulative Returns")
        
        ax.set_xlabel("Date")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_drawdown(
        self,
        figsize: tuple = (12, 6),
        title: str = "Drawdown",
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot drawdown chart.
        
        Args:
            figsize: Figure size
            title: Plot title
            show: Whether to show the plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate drawdown
        if "equity" in self.results.columns:
            equity = self.results["equity"]
        else:
            equity = (1 + self.results["net_returns"]).cumprod()
        
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        ax.fill_between(
            self.results.index,
            drawdown * 100,
            0,
            color="red",
            alpha=0.3,
        )
        ax.plot(self.results.index, drawdown * 100, color="red", linewidth=1)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_returns_distribution(
        self,
        figsize: tuple = (12, 6),
        bins: int = 50,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot distribution of returns.
        
        Args:
            figsize: Figure size
            bins: Number of histogram bins
            show: Whether to show the plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if "net_returns" in self.results.columns:
            returns = self.results["net_returns"]
        else:
            returns = self.results["equity"].pct_change().fillna(0)
        
        ax.hist(returns * 100, bins=bins, alpha=0.7, edgecolor="black")
        
        # Add vertical line for mean
        mean_return = returns.mean() * 100
        ax.axvline(mean_return, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_return:.2f}%")
        
        ax.set_xlabel("Returns (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Returns")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_overview(
        self,
        figsize: tuple = (15, 10),
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot overview with multiple subplots.
        
        Args:
            figsize: Figure size
            show: Whether to show the plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Equity curve
        if "equity" in self.results.columns:
            axes[0].plot(self.results.index, self.results["equity"], linewidth=2)
            axes[0].set_ylabel("Portfolio Value ($)")
        else:
            cumulative_returns = (1 + self.results["net_returns"]).cumprod()
            axes[0].plot(self.results.index, cumulative_returns, linewidth=2)
            axes[0].set_ylabel("Cumulative Returns")
        
        axes[0].set_title("Equity Curve")
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        if "equity" in self.results.columns:
            equity = self.results["equity"]
        else:
            equity = (1 + self.results["net_returns"]).cumprod()
        
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        axes[1].fill_between(
            self.results.index,
            drawdown * 100,
            0,
            color="red",
            alpha=0.3,
        )
        axes[1].plot(self.results.index, drawdown * 100, color="red", linewidth=1)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_title("Drawdown")
        axes[1].grid(True, alpha=0.3)
        
        # Position (if available)
        if "position" in self.results.columns:
            axes[2].plot(self.results.index, self.results["position"], linewidth=1, alpha=0.7)
            axes[2].set_ylabel("Position")
            axes[2].set_title("Position Over Time")
            axes[2].grid(True, alpha=0.3)
        else:
            # Show returns instead
            if "net_returns" in self.results.columns:
                returns = self.results["net_returns"] * 100
            else:
                returns = self.results["equity"].pct_change().fillna(0) * 100
            
            axes[2].bar(self.results.index, returns, width=1, alpha=0.7)
            axes[2].set_ylabel("Returns (%)")
            axes[2].set_title("Daily Returns")
            axes[2].grid(True, alpha=0.3)
        
        axes[2].set_xlabel("Date")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_interactive(
        self,
        title: str = "Backtest Results",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """Create interactive plot using Plotly.
        
        Args:
            title: Plot title
            save_path: Path to save HTML file
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Equity Curve", "Drawdown", "Position/Returns"),
        )
        
        # Equity curve
        if "equity" in self.results.columns:
            equity = self.results["equity"]
        else:
            equity = (1 + self.results["net_returns"]).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=equity,
                name="Equity",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
        
        # Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=drawdown,
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red", width=1),
            ),
            row=2,
            col=1,
        )
        
        # Position or returns
        if "position" in self.results.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results["position"],
                    name="Position",
                    line=dict(color="green", width=1),
                ),
                row=3,
                col=1,
            )
        else:
            if "net_returns" in self.results.columns:
                returns = self.results["net_returns"] * 100
            else:
                returns = equity.pct_change().fillna(0) * 100
            
            fig.add_trace(
                go.Bar(
                    x=self.results.index,
                    y=returns,
                    name="Returns",
                    marker_color="lightblue",
                ),
                row=3,
                col=1,
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=900,
            showlegend=True,
            hovermode="x unified",
        )
        
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Position/Returns", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_report(
        self,
        output_dir: str = "backtest_report",
        strategy_name: str = "Strategy",
    ) -> None:
        """Create comprehensive HTML report.
        
        Args:
            output_dir: Output directory for report
            strategy_name: Name of the strategy
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        self.plot_overview(
            show=False,
            save_path=output_path / "overview.png",
        )
        
        self.plot_equity_curve(
            show=False,
            save_path=output_path / "equity_curve.png",
        )
        
        self.plot_drawdown(
            show=False,
            save_path=output_path / "drawdown.png",
        )
        
        self.plot_returns_distribution(
            show=False,
            save_path=output_path / "returns_distribution.png",
        )
        
        # Create interactive plot
        self.plot_interactive(
            title=f"{strategy_name} - Interactive Results",
            save_path=output_path / "interactive.html",
        )
        
        print(f"Report generated in: {output_path}")
