"""
Monty Hall Problem - Interactive Streamlit Simulation

This application provides an interactive simulation of the famous Monty Hall problem,
allowing users to run simulations and see real-time updates of winning probabilities
for both "Stay" and "Switch" strategies.
"""

import random
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


class MontyHallSimulator:
    """
    A comprehensive Monty Hall problem simulator with real-time tracking capabilities.
    
    This class handles the core simulation logic, maintains statistics, and provides
    methods for running single or batch simulations.
    """
    
    def __init__(self) -> None:
        """Initialize the simulator with empty statistics."""
        self.reset_stats()
    
    def reset_stats(self) -> None:
        """Reset all simulation statistics to initial state."""
        self.stay_wins: int = 0
        self.switch_wins: int = 0
        self.total_simulations: int = 0
        self.simulation_history: List[Dict[str, any]] = []
    
    def run_single_simulation(self) -> Tuple[bool, bool, Dict[str, int]]:
        """
        Run a single Monty Hall simulation.
        
        Returns:
            Tuple containing:
            - bool: True if staying wins
            - bool: True if switching wins  
            - Dict: Game state information (doors, choices, etc.)
        """
        try:
            # Setup: 3 doors (0, 1, 2), prize behind one randomly
            prize_door: int = random.randint(0, 2)
            contestant_choice: int = random.randint(0, 2)
            
            # Host opens a door that doesn't have prize and wasn't chosen
            available_doors: List[int] = [
                door for door in range(3) 
                if door != contestant_choice and door != prize_door
            ]
            host_opens: int = random.choice(available_doors)
            
            # Determine switch choice (the remaining unopened door)
            switch_choice: int = next(
                door for door in range(3) 
                if door != contestant_choice and door != host_opens
            )
            
            # Determine winners
            stay_wins: bool = (contestant_choice == prize_door)
            switch_wins: bool = (switch_choice == prize_door)
            
            # Game state for visualization
            game_state: Dict[str, int] = {
                'prize_door': prize_door,
                'contestant_choice': contestant_choice,
                'host_opens': host_opens,
                'switch_choice': switch_choice
            }
            
            return stay_wins, switch_wins, game_state
            
        except Exception as e:
            st.error(f"Error in simulation: {str(e)}")
            return False, False, {}
    
    def run_batch_simulation(self, num_trials: int, progress_callback=None) -> Tuple[int, int]:
        """
        Run multiple simulations in batch.
        
        Args:
            num_trials: Number of simulations to run
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (stay_wins, switch_wins) for this batch
        """
        batch_stay_wins: int = 0
        batch_switch_wins: int = 0
        
        try:
            for i in range(num_trials):
                stay_win, switch_win, _ = self.run_single_simulation()
                
                if stay_win:
                    batch_stay_wins += 1
                    self.stay_wins += 1
                
                if switch_win:
                    batch_switch_wins += 1  
                    self.switch_wins += 1
                
                self.total_simulations += 1
                
                # Record history for trend analysis
                self.simulation_history.append({
                    'simulation_num': self.total_simulations,
                    'stay_win': stay_win,
                    'switch_win': switch_win,
                    'stay_win_rate': self.stay_wins / self.total_simulations,
                    'switch_win_rate': self.switch_wins / self.total_simulations,
                    'timestamp': datetime.now()
                })
                
                # Progress callback for UI updates
                if progress_callback and (i + 1) % max(1, num_trials // 20) == 0:
                    progress_callback(i + 1, num_trials)
            
            return batch_stay_wins, batch_switch_wins
            
        except Exception as e:
            st.error(f"Error in batch simulation: {str(e)}")
            return 0, 0
    
    def get_win_rates(self) -> Tuple[float, float]:
        """
        Calculate current winning rates for both strategies.
        
        Returns:
            Tuple of (stay_win_rate, switch_win_rate)
        """
        if self.total_simulations == 0:
            return 0.0, 0.0
        
        stay_rate: float = self.stay_wins / self.total_simulations
        switch_rate: float = self.switch_wins / self.total_simulations
        
        return stay_rate, switch_rate


def create_door_visualization(game_state: Dict[str, int]) -> go.Figure:
    """
    Create a visual representation of the Monty Hall game state.
    
    Args:
        game_state: Dictionary containing door information
        
    Returns:
        Plotly figure showing the game state
    """
    try:
        # Door labels and colors
        door_labels = ['Door 1', 'Door 2', 'Door 3']
        colors = ['#e6f3ff'] * 3  # Very light blue
        
        # Highlight special doors
        if 'contestant_choice' in game_state:
            colors[game_state['contestant_choice']] = '#fff2b3'  # Light gold
        
        if 'host_opens' in game_state:
            colors[game_state['host_opens']] = '#ffcccc'  # Light red
            
        if 'switch_choice' in game_state:
            colors[game_state['switch_choice']] = '#ccffcc'  # Light green
        
        # Create bar chart representation
        fig = go.Figure(data=[
            go.Bar(
                x=door_labels,
                y=[1, 1, 1],
                marker_color=colors,
                text=['🚗' if i == game_state.get('prize_door', -1) else '🐐' 
                      for i in range(3)],
                textposition='inside',
                textfont_size=40
            )
        ])
        
        fig.update_layout(
            # title="Game State Visualization",
            xaxis_title="Doors",
            yaxis_title="",
            showlegend=False,
            height=300,
            yaxis={'showticklabels': False}
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating door visualization: {str(e)}")
        return go.Figure()


def create_trend_chart(history: List[Dict]) -> go.Figure:
    """
    Create a trend chart showing win rates over time.
    
    Args:
        history: List of simulation history records
        
    Returns:
        Plotly figure showing win rate trends
    """
    try:
        if not history:
            return go.Figure()
        
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # Add stay win rate line
        fig.add_trace(go.Scatter(
            x=df['simulation_num'],
            y=df['stay_win_rate'],
            mode='lines',
            name='Stay Strategy',
            line=dict(color='red', width=2)
        ))
        
        # Add switch win rate line  
        fig.add_trace(go.Scatter(
            x=df['simulation_num'],
            y=df['switch_win_rate'],
            mode='lines',
            name='Switch Strategy',
            line=dict(color='green', width=2)
        ))
        
        # Add theoretical lines
        fig.add_hline(y=1/3, line_dash="dash", line_color="red", 
                     annotation_text="Theoretical Stay (33.33%)")
        fig.add_hline(y=2/3, line_dash="dash", line_color="green",
                     annotation_text="Theoretical Switch (66.67%)")
        
        fig.update_layout(
            title="Win Rate Trends Over Time",
            xaxis_title="Simulation Number",
            yaxis_title="Win Rate",
            yaxis_tickformat='.2%',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating trend chart: {str(e)}")
        return go.Figure()


def main() -> None:
    """
    Main Streamlit application function.
    
    Sets up the UI and handles user interactions for the Monty Hall simulation.
    """
    # Page configuration
    st.set_page_config(
        page_title="Monty Hall Problem Simulation",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .info-metric {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = MontyHallSimulator()
    
    simulator = st.session_state.simulator
    
    # Main header
    st.markdown('<h1 class="main-header">🚗 Monty Hall Problem Simulation 🐐</h1>', 
                unsafe_allow_html=True)
    
    # Problem explanation
    with st.expander("📖 **What is the Monty Hall Problem?**", expanded=False):
        st.markdown("""
        The **Monty Hall Problem** is a famous probability puzzle based on a game show scenario:
        
        1. There are 3 doors. Behind one door is a car (prize), behind the others are goats.
        2. You pick a door (but don't open it yet).
        3. The host, who knows what's behind each door, opens one of the remaining doors that has a goat.
        4. The host asks if you want to stick with your original choice or switch to the other unopened door.
        
        **The Question**: Should you stay or switch?
        """)
        
        # Example visualization
        st.markdown("**Example Game Scenario:**")
        example_game_state = {
            'prize_door': 0,      # Car is behind Door 1
            'contestant_choice': 1,  # You chose Door 2
            'host_opens': 2,      # Host opens Door 3 (has goat)
            'switch_choice': 0    # You can switch to Door 1
        }
        
        example_fig = create_door_visualization(example_game_state)
        st.plotly_chart(example_fig, use_container_width=True, key="example_door_viz")
        
        st.markdown("""
        **Legend**: 🟨 Gold = Your choice | 🟩 Green = Switch option | 🟥 Red = Host opened | 🚗 Car | 🐐 Goat
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏠 Stay Strategy:** You keep Door 2 → ❌ **LOSS**")
        with col2:
            st.markdown("**🔄 Switch Strategy:** You switch to Door 1 → 🏆 **WIN**")

    with st.expander("🔍 **Answer**", expanded=False):
        st.markdown("""
        The counterintuitive result of the Monty Hall problem is that switching doors actually doubles your chances of winning the car.
        
        - **Staying** with your initial choice gives you a 1 in 3 chance (33.33%) of winning the car.
        - **Switching** to the other unopened door gives you a 2 in 3 chance (66.67%) of winning the car.
        
        This is because when you first pick a door, there is a 2 in 3 chance that the car is behind one of the other two doors. When the host opens a door to reveal a goat, that 2 in 3 probability effectively transfers to the remaining unopened door.
        
        Try running simulations below to see this effect in action!
        """)
            
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎮 Simulation Controls")
        
        # Single simulation
        st.subheader("Single Simulation")
        if st.button("🎯 Run One Simulation", type="primary"):
            stay_win, switch_win, game_state = simulator.run_single_simulation()
            
            if stay_win:
                simulator.stay_wins += 1
            if switch_win:
                simulator.switch_wins += 1
            simulator.total_simulations += 1
            
            # Record in history
            simulator.simulation_history.append({
                'simulation_num': simulator.total_simulations,
                'stay_win': stay_win,
                'switch_win': switch_win,
                'stay_win_rate': simulator.stay_wins / simulator.total_simulations,
                'switch_win_rate': simulator.switch_wins / simulator.total_simulations,
                'timestamp': datetime.now()
            })
            
            st.session_state.last_game_state = game_state
            st.session_state.last_stay_win = stay_win
            st.session_state.last_switch_win = switch_win
            st.success(f"Simulation complete! Stay: {'Win' if stay_win else 'Loss'}, Switch: {'Win' if switch_win else 'Loss'}")
        
        # Batch simulation
        st.subheader("Batch Simulation")
        batch_size = st.selectbox(
            "Select batch size:",
            [10, 50, 100, 500, 1000, 5000],
            index=2
        )
        
        if st.button("🚀 Run Batch Simulation"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current: int, total: int) -> None:
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(f"Running simulation {current}/{total}")
            
            batch_stay, batch_switch = simulator.run_batch_simulation(
                batch_size, progress_callback
            )
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed {batch_size} simulations!")
            st.success(f"Batch complete! Stay wins: {batch_stay}, Switch wins: {batch_switch}")
        
        # Reset button
        st.subheader("Reset")
        if st.button("🔄 Reset All Data", type="secondary"):
            simulator.reset_stats()
            if 'last_game_state' in st.session_state:
                del st.session_state.last_game_state
            st.success("All data reset!")
            st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    # Statistics display
    stay_rate, switch_rate = simulator.get_win_rates()
    
    with col1:
        st.markdown('<div class="metric-container info-metric">', unsafe_allow_html=True)
        st.metric(
            label="🏠 Stay Strategy Wins",
            value=simulator.stay_wins,
            delta=f"{stay_rate:.2%}" if simulator.total_simulations > 0 else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container success-metric">', unsafe_allow_html=True)
        st.metric(
            label="🔄 Switch Strategy Wins", 
            value=simulator.switch_wins,
            delta=f"{switch_rate:.2%}" if simulator.total_simulations > 0 else None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="📊 Total Simulations",
            value=simulator.total_simulations
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Progress bars for win rates
    if simulator.total_simulations > 0:
        st.subheader("📊 Current Win Rates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Stay Strategy**")
            st.progress(stay_rate)
            st.write(f"Win Rate: {stay_rate:.2%} (Expected: 33.33%)")
        
        with col2:
            st.write("**Switch Strategy**") 
            st.progress(switch_rate)
            st.write(f"Win Rate: {switch_rate:.2%} (Expected: 66.67%)")
    
    # Game visualization
    if 'last_game_state' in st.session_state:
        st.subheader("🎪 Last Game Visualization")
        
        
        door_fig = create_door_visualization(st.session_state.last_game_state)
        st.plotly_chart(door_fig, use_container_width=True, key="last_game_door_viz")
        
        # Legend for door colors
        st.markdown("""
        **Legend**: 
        🟨 Gold = Your original choice | 🟩 Green = Switch option | 🟥 Red = Host opened | 🚗 Car | 🐐 Goat
        """)
        # Display last game outcome
        if 'last_stay_win' in st.session_state and 'last_switch_win' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                stay_result = "🏆 WIN" if st.session_state.last_stay_win else "❌ LOSS"
                st.markdown(f"**Stay Strategy Result:** {stay_result}")
            with col2:
                switch_result = "🏆 WIN" if st.session_state.last_switch_win else "❌ LOSS"
                st.markdown(f"**Switch Strategy Result:** {switch_result}")
            st.markdown("---")
    
    # Trend chart
    if len(simulator.simulation_history) > 1:
        st.subheader("📈 Win Rate Trends")
        trend_fig = create_trend_chart(simulator.simulation_history)
        st.plotly_chart(trend_fig, use_container_width=True, key="win_rate_trend_chart")
    
    # Summary statistics table
    if simulator.total_simulations > 0:
        st.subheader("📋 Detailed Statistics")
        
        stats_data = {
            'Strategy': ['Stay', 'Switch', 'Theoretical Stay', 'Theoretical Switch'],
            'Wins': [str(simulator.stay_wins), str(simulator.switch_wins), "33%", "67%"],
            'Win Rate': [f"{stay_rate:.2%}", f"{switch_rate:.2%}", "33.33%", "66.67%"],
            'Difference from Theory': [
                f"{stay_rate - 1/3:+.2%}",
                f"{switch_rate - 2/3:+.2%}",
                "0.00%",
                "0.00%"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()