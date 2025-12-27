"""
Water Demand Analytics Dashboard - Main Entry Point
This file serves as the entry point for Streamlit deployment
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import and run the dashboard app
try:
    from dashboards.app import main
    main()
except ImportError:
    # Fallback for deployment environments
    import importlib.util
    spec = importlib.util.spec_from_file_location("dashboards_app", os.path.join(os.path.dirname(__file__), "dashboards", "app.py"))
    dashboards_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboards_app)
    dashboards_app.main()
