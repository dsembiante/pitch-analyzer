# Pitch Analyzer

A computer vision tool that analyzes baseball pitching mechanics from video footage. Built as a portfolio prototype to demonstrate automated biomechanics analysis using MediaPipe pose estimation.

Given a side-angle video of a pitcher, this tool extracts pose landmarks frame-by-frame and computes 10 key mechanics metrics — including arm slot, stride length, hip-shoulder separation, and release point consistency — displayed in a clean Streamlit interface with session-over-session comparison.

**Tech stack:** Python, MediaPipe, OpenCV, Streamlit, pandas, scipy, matplotlib