"""
Water Demand Analytics Dashboard - Main Entry Point
This file serves as the entry point for Streamlit deployment
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import and run the dashboard app
from dashboards.app import main

main()
