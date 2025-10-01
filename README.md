# 🚗 Monty Hall Problem - Interactive Streamlit Simulation

An interactive web application built with Streamlit that demonstrates the famous Monty Hall probability puzzle through real-time simulations and visualizations.

## 📖 What is the Monty Hall Problem?

The **Monty Hall Problem** is a famous probability puzzle based on a game show scenario:

1. There are 3 doors. Behind one door is a car (prize), behind the others are goats.
2. You pick a door (but don't open it yet).
3. The host, who knows what's behind each door, opens one of the remaining doors that has a goat.
4. The host asks if you want to stick with your original choice or switch to the other unopened door.

**The Question**: Should you stay or switch?

**The Answer**: Switching doors doubles your chances of winning! 
- **Staying**: 33.33% chance of winning
- **Switching**: 66.67% chance of winning

## ✨ Features

### 🎮 Interactive Simulations
- **Single Simulation**: Run one game at a time with visual door representation
- **Batch Simulation**: Run multiple simulations (10 to 5,000) with progress tracking
- **Real-time Results**: See immediate outcomes for both Stay and Switch strategies

### 📊 Comprehensive Analytics
- **Live Statistics**: Track wins, losses, and win rates for both strategies
- **Progress Visualization**: Interactive progress bars showing current win rates vs. theoretical rates
- **Trend Analysis**: Line charts showing how win rates converge to theoretical values over time
- **Detailed Statistics Table**: Complete breakdown with differences from theoretical values

### 🎪 Visual Game Representation
- **Door Visualization**: Interactive door display showing:
  - 🟨 Your original choice (Gold)
  - 🟩 Switch option (Green)
  - 🟥 Host opened door (Red)
  - 🚗 Car location
  - 🐐 Goat locations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
```bash
git clone <repository-url>
cd streamlit-monty-hall-simulation
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in your terminal

## 📦 Dependencies

This project requires the following Python packages (see `requirements.txt` for exact versions):

- **streamlit**: Web application framework for creating interactive data apps
- **pandas**: Data manipulation and analysis library
- **plotly**: Interactive plotting library for visualizations
- **numpy**: Numerical computing library (dependency of pandas/plotly)

All other imports (`random`, `datetime`, `typing`) are part of Python's standard library.

## 🎯 Usage Guide

### Running Single Simulations
1. Use the **"🎯 Run One Simulation"** button in the sidebar
2. View the game state visualization showing door positions
3. Check the results for both Stay and Switch strategies
4. Observe how individual results affect overall statistics

### Running Batch Simulations
1. Select batch size from the dropdown (10 to 5,000 simulations)
2. Click **"🚀 Run Batch Simulation"**
3. Watch the progress bar as simulations run
4. Analyze the updated statistics and trend charts

### Understanding the Visualizations
- **Door Display**: Shows the current game state with color-coded doors
- **Win Rate Progress Bars**: Compare actual vs. theoretical win rates
- **Trend Charts**: Observe how win rates converge over multiple simulations
- **Statistics Table**: Detailed breakdown of all results
